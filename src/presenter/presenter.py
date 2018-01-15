from sklearn.metrics.pairwise import cosine_similarity

from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np



def calc_purity(labels_true, labels_pred, vectorizer, top_terms):

    num = len(top_terms)

    # Transform top term corpus in a weighted top term corpus
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
    v_weighted_top_terms = vectorizer.vectorizer.transform(weighted_top_terms)

    # Take cosine similarity between topics
    sim = cosine_similarity(v_weighted_top_terms)

    df = pd.DataFrame(dict(lp=labels_pred,
                           lt=labels_true))
    groups = df.groupby('lt')

    package_purities = {}
    for lt, group in groups:

        counts = group.groupby('lp').count().sort_values(by=['lt'],
                                                         ascending=0)
        max_count = group.groupby('lp').count()['lt'].max(axis=0)
        max_topic = group.groupby('lp').count()['lt'].idxmax(axis=0)
        sum_count = group.groupby('lp').count()['lt'].sum(axis=0)

        counts.reset_index(level=0, inplace=True)
        acc = 0
        for i, row in counts.iterrows():
            if (row['lp'] != max_topic) and (sim[row['lp']][max_topic] > 0.2):
                acc += row['lt']
        package_purities[lt] = (max_count + acc)/sum_count
    temp = list(package_purities.values())
    return np.mean(temp)


def calc_topic_categories(top_terms, vectorizer):

    num = len(top_terms)

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
    v_weighted_top_terms = vectorizer.vectorizer.transform(weighted_top_terms)

    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics.pairwise import cosine_distances

    # Take cosine similarity between topics
    cos_dis = cosine_distances(v_weighted_top_terms)
    z = linkage(cos_dis, method='complete', metric='cosine')
    t = 0.4
    kappa = fcluster(z, t)
    print('Num categories: {}'.format(len(set(kappa))))
    return len(set(kappa))


def create_wordclouds(top_terms, indices):

    #  top_terms = [['layout', 'border', 'panel', 'action', 'listener', 'selected'],
                 #  ['socket', 'listener', 'open', 'close', 'network', 'open']]
    #  top_terms = top_terms[0:8]
    #  indices = indices[0:8]
    num = len(top_terms[0])

    # Transform top term corpus in a weighted top term corpus
    weighted_top_terms = []
    for topic in top_terms:
        weighted_list = []
        for ind, term in enumerate(topic):
            for i in range(0, num - ind + 1):
                weighted_list.append(term)
        weighted_top_terms.append(weighted_list)

    for ind, wl in enumerate(weighted_top_terms):
        weighted_top_terms[ind] = ' '.join(weighted_top_terms[ind])

    for ind, wl in enumerate(weighted_top_terms):
        tags = make_tags(get_tag_counts(wl), maxsize=150)
        create_tag_image(tags, '../output/topic{}.png'.format(indices[ind]), size=(1300, 1150),
                         fontname='PT Sans Regular', layout=2, rectangular=True)
    offset = len(top_terms)//3
    fig, ax = plt.subplots(3, offset, figsize=(10, 8))
    for row in range(3):
        for idx in range(offset):
            im = mpimg.imread('../output/topic{}.png'.format(indices[idx+row*offset]))
            ax[row, idx].imshow(im)
            ax[row, idx].set_xticks([])
            ax[row, idx].set_yticks([])
            ax[row, idx].set_title('Topic #{}'.format(indices[idx+row*offset]))
    fig.tight_layout()
    fig.savefig('../output/tags.eps', format='eps')
