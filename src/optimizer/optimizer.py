from presenter.presenter import Presenter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd

class Optimizer(object):

    def __init__(self, vectorizer):
        self.previous_clusterer = None
        self.p_overall_purity = None
        self.p_num_similar_topics = None
        self.vectorizer = vectorizer

    def examine(self, clusterer):

        if self.previous_clusterer is None:
            self.previous_clusterer = clusterer
            p = Presenter()
            self.p_overall_purity = p.purity(clusterer.labels_true,
                                             clusterer.labels_pred)
            self.calc_similar_topics(clusterer.top_terms_per_cluster(num=15,
                                                                     show=False))
            return True
        else:
            p = Presenter()
            #  overall_purity = p.purity(clusterer.labels_true,
                                      #  clusterer.labels_pred)

    def calc_similar_topics(self, top_terms, freq, unique=True, categories=True):

        num = len(top_terms)

        # Transform top term corpus in a weighted top term corpus
        #  weighted_top_terms = []
        #  for idx, topic in enumerate(top_terms):
            #  weighted_list = []
            #  for ind, term in enumerate(topic):
                #  for i in range(0, freq[idx][ind].astype(int)):
                    #  weighted_list.append(term)
            #  weighted_top_terms.append(weighted_list)

        #  for ind, wl in enumerate(weighted_top_terms):
            #  weighted_top_terms[ind] = ' '.join(weighted_top_terms[ind])

        weighted_top_terms = []
        for topic in top_terms:
            weighted_list = []
            for ind, term in enumerate(topic):
                for i in range(0, num - ind + 1):
                    weighted_list.append(term)
            weighted_top_terms.append(weighted_list)

        for ind, wl in enumerate(weighted_top_terms):
            weighted_top_terms[ind] = ' '.join(weighted_top_terms[ind])

        # Transform weighted top terms to vector space
        v_weighted_top_terms = self.vectorizer.vectorizer.transform(weighted_top_terms)

        #  print(len(top_terms))
        #  print(len(v_weighted_top_terms.shape))
        #  print(v_weighted_top_terms.dtype)
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage
        from scipy.cluster.hierarchy import fcluster
        from sklearn.metrics.pairwise import cosine_distances
        # Take cosine similarity between topics
        cos_dis = cosine_distances(v_weighted_top_terms)
        Z = linkage(cos_dis, method='complete', metric='cosine')
        t = 0.4
        kappa = fcluster(Z, t)
        print('Num categories: {}'.format(len(set(kappa))))
        return len(set(kappa))
        #  return len(set(kappa))

        #  if categories is True:
            #  categories = []
            #  for topic_n_1 in range(cos_sim.shape[0]):
                #  for topic_n_2 in range(cos_sim.shape[0]):
                    #  if topic_n_1 != topic_n_2:
                        #  sim = cos_sim[topic_n_1][topic_n_2]
                        #  if sim > 0.8:
                            #  #  found = False
                            #  for idx, c in enumerate(categories):
                                #  if topic_n_1 in c or topic_n_2 in c:
                                    #  categories[idx].update([topic_n_1,
                                                            #  topic_n_2])
                                    #  #  found = True
                            #  #  if found is False:
                            #  categories.append(set())
                            #  categories[-1].update([topic_n_1,
                                                   #  topic_n_2])
                        #  else:
                            #  continue
            #  for topic_n in range(cos_sim.shape[0]):
                #  found = False
                #  for c in categories:
                    #  if topic_n in c:
                        #  found = True
                #  if found == False:
                    #  categories.append(set())
                    #  categories[-1].update([topic_n])
            #  return len(categories)
        #  else:
            #  if unique is False:
                #  different_indices = []
                #  for idx in range(x.shape[0]):
                    #  if x[idx] != y[idx]:
                        #  different_indices.append(idx)

                #  num_similar_topics = x[different_indices].shape[0]/2
                #  return num_similar_topics
            #  else:
                #  unique_indices = []
                #  for topic in range(num):
                    #  similar_flag = False
                    #  for idx in range(x.shape[0]):
                        #  if x[idx] != y[idx]:
                            #  if (topic == x[idx] or topic == y[idx]):
                                #  similar_flag = True
                                #  break
                    #  if similar_flag is False:
                        #  unique_indices.append(topic)
                #  if not unique_indices:
                    #  return 0
                #  else:
                    #  return len(unique_indices)

def main():

    clusterers = []
    for file in os.listdir('cached_clusterers/hadoop'):
        f = open('cached_clusterers/hadoop/'+file, 'rb')
        clusterers.append(pickle.load(f))
        f.close()
    purities = {}
    for cl in clusterers:
        e = Presenter()
        purities[cl.n_clusters] = e.purity(cl.labels_true, cl.labels_pred)
    df = pd.DataFrame(list(purities.items()), columns=('Topics',
                                                       'Overall Purity'))
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    df['Overall Purity'].plot(ax=ax1, x='Topics', grid=True,
                              legend=True, figsize=(40, 25))
    val = df['Overall Purity'].values
    xpoints = np.linspace(0, val.shape[0]-1, val.shape[0])
    poly = np.polyfit(xpoints, val, 2)
    poly = np.poly1d(poly)
    ax1.plot(xpoints, poly(xpoints), 'r-', label='Fitted Polynomial')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    ax1.set_xlabel('Topics')
    #  fig.savefig('Purity.svg', format='svg')

    similar_topics = {}
    for cl in clusterers:
        o = Optimizer(cl.vect)
        similar_topics[cl.n_clusters] = o.calc_similar_topics(cl.top_terms_per_cluster())
    df1 = pd.DataFrame(list(similar_topics.items()),
                       columns=('Topics', 'Number of Similar Topics'))
    df1['Number of Similar Topics'].plot(ax=ax2, x='Topics', grid=True,
                                        legend=True, figsize=(40, 25))
    val = df1['Number of Similar Topics'].values
    xpoints = np.linspace(0, val.shape[0]-1, val.shape[0])
    poly = np.polyfit(xpoints, val, 2)
    poly = np.poly1d(poly)
    ax2.plot(xpoints, poly(xpoints), 'r-', label='Fitted Polynomial')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)
    ax2.set_xlabel('Topics')
    f.savefig('Num_similar.svg', format='svg')

if __name__ == '__main__':
    main()
