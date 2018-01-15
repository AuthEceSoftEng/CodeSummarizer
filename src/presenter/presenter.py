import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
