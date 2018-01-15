from presenter.presenter import Presenter
import logging

logger = logging.getLogger('Optimizer')


class Optimizer(object):

    def __init__(self):
        self.clusters = []
        self.overall_purity = []
        self.num_similar_topics = []

    def examine(self, clusterer):

        self.clusterers.append(clusterer.n_clusters)
        p = Presenter()
        self.overall_purity.append(p.purity(clusterer.labels_true, clusterer.labels_pred,
                                            vectorizer=clusterer.vectorizer,
                                            top_terms=clusterer.top_terms_per_cluster(show=False)))

        self.num_similar_topics.append(p.calc_similar_topics(clusterer.top_terms_per_cluster(), clusterer.vectorizer))
        logger.info('Clustered with {}. Overall purity: {}. Number of categories {}'.format(clusterer.n_clusters,
                                                                                            self.overall_purity[-1],
                                                                                            self.num_similar_topics[-1]))

    def optimize(self, clusterer, n_clusters, one_run=False):

        if one_run is True:
            logger.info('OneRun Enabled. Clustering with {} clusters...'.format(clusterer.n_clusters))
            clusterer.cluster(n_clusters=n_clusters)
            self.examine(clusterer)
        else:
            for i in range(10, n_clusters):
                logger.info('Clustering with {} clusters...'.format(i))
                clusterer.cluster(n_clusters=i)
                self.examine(clusterer)
