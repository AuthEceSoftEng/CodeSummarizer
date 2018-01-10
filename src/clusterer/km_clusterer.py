from __future__ import print_function
from __future__ import absolute_import

from time import time
import logging

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn import metrics

import numpy as np
import pandas as pd

from clusterer.abstract_clusterer import AbsClust
from clusterer.cluster_tools import (cohesion_separation_per_cluster,
                                     by_topic,
                                     similarity_matrix,
                                     silhouette_per_cluster)

logger = logging.getLogger('KMClust')


class KMClust(AbsClust):

    '''Clusters a set of documents with the KMeans algorithm

    Parameters
    ----------
    vect        : Object of type *Vect.
    labels_true : True labels to be used for metrics evaluation.
    '''

    def __init__(self, vect, labels_true, init='k-means++',
                 max_iter=1000, n_init=10, tol=1e-6):

        super(KMClust, self).__init__('KMeans', vect, labels_true)
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol

    def cluster(self, n_clusters):

        self.n_clusters = n_clusters

        km_clusterer = KMeans(n_clusters=n_clusters, init=self.init,
                              max_iter=self.max_iter, n_init=self.n_init,
                              verbose=False, n_jobs=4, tol=self.tol)

        logger.info('Clustering sparse data with {}'.format(km_clusterer))

        t0 = time()
        km_clusterer.fit(self.vect.vec_data)
        logger.info('done in {:0.3f}s'.format((time() - t0)))
        self.labels_pred = km_clusterer.labels_
        self.cluster_centers = km_clusterer.cluster_centers_

    def metrics(self, show_graphs=False, search=False):

        names = ['Homogeneity', 'Completeness', 'V-measure',
                 'Adjusted Rand-Index', 'Silhouette', 'Cohesion',
                 'Separation']

        results = np.zeros(shape=len(names), dtype=np.double)

        results[0] = metrics.homogeneity_score(self.labels_true,
                                               self.labels_pred)
        results[1] = metrics.completeness_score(self.labels_true,
                                                self.labels_pred)
        results[2] = metrics.v_measure_score(self.labels_true,
                                             self.labels_pred)
        results[3] = metrics.adjusted_rand_score(self.labels_true,
                                                 self.labels_pred)
        results[4] = metrics.silhouette_score(self.vect.vec_data,
                                              self.labels_pred,
                                              sample_size=1000)
        coh_sep = cohesion_separation_per_cluster(self.vect.vec_data,
                                                  self.labels_pred,
                                                  self.n_clusters,
                                                  self.cluster_centers,
                                                  show_graphs=False)
        results[5] = coh_sep.Cohesion.sum()
        results[6] = coh_sep.Separation.sum()

        logger.info('Number of labels: {}'.format(len(set(self.labels_true))))
        for index, name in enumerate(names):
            logger.info('{0:20}: {1:.3f}'.format(name, results[index]))

        logger.info('Top terms per cluster:')

        self.top_terms_per_cluster(num=10, show=True)

        if show_graphs:
            #  similarity_matrix(cosine_similarity(self.vectorizer.vec_data),
                              #  self.labels_pred)
            #  similarity_matrix(cosine_similarity(self.cluster_centers))
            #  silhouette_per_cluster(self.vectorizer.vec_data,
                                   #  self.labels_pred,
                                   #  self.n_clusters)
            cohesion_separation_per_cluster(self.vect.vec_data,
                                            self.labels_pred,
                                            self.n_clusters,
                                            self.cluster_centers,
                                            show_graphs=show_graphs)

        df = pd.DataFrame(columns=names)
        df.loc[self.n_clusters] = results
        return df

    def top_terms_per_cluster(self, num=15, show=False):
        '''
        Calculate the top terms per cluster center.

        Parameters
        ----------
        num             : Number of top terms to calculate
        show            : Show the top terms in log

        Returns
        -------
        top_terms       : List of size num, with the top num terms
        '''
        order_centroids = self.cluster_centers.argsort()[:, ::-1]
        top_terms = []
        for i in range(len(self.cluster_centers)):
            top_terms.append([self.vect.feature_names[ind]
                              for ind in order_centroids[i, :num]])
        if show:
            for index, clust in enumerate(top_terms):
                logger.info('Cluster {}: {}'.format(index, ' '.join(clust)))

        return top_terms
