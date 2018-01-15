from __future__ import print_function
from __future__ import absolute_import

import logging
import numpy as np
from time import time
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation

from clusterer.abstract_clusterer import AbsClust

logger = logging.getLogger('LDAClust')


class LDAClust(AbsClust):

    '''Clusters a set of documents with the LDA algorithm

    Parameters
    ----------

    labels_true : True labels to be used for metrics evaluation.
    vect        : Object of type *Vect.
    '''

    def __init__(self, vect, labels_true, max_iter=1000,
                 learning_method='online', learning_offset=50.,
                 dtp=500, twp=.001):

        super(LDAClust, self).__init__('LDA', vect, labels_true)

        self.labels_ = None
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.learning_offset = learning_offset
        self.dtp = dtp
        self.twp = twp

    def cluster(self, n_clusters):
        self.n_clusters = n_clusters

        lda_clusterer = LatentDirichletAllocation(n_components=n_clusters,
                                                  doc_topic_prior=self.dtp/n_clusters,
                                                  topic_word_prior=self.twp,
                                                  max_iter=self.max_iter,
                                                  learning_method=self.learning_method,
                                                  learning_offset=self.learning_offset,
                                                  n_jobs=1)
        logger.info('Performing LDA on vectorized data')
        logger.info('Parameters:')
        logger.info('Doc_topic_prior(a) = {}'.format(500/n_clusters))
        logger.info('Topic_word_prior(b) = {}'.format(0.001))

        t0 = time()
        self.lda_vec_data = lda_clusterer.fit_transform(self.vect.vec_data)
        self.lda_topic_word = lda_clusterer.components_
        self.labels_pred = self.gen_labels()
        logger.info("Done in %0.3fs" % (time() - t0))

    def metrics(self, show_graphs=False):
        self.top_terms_per_cluster(num=10, show=True)
        by_folder(self.labels_pred, self.labels_true)

    def top_terms_per_cluster(self,  num=15, show=False, freq=False):
        top_terms = []
        frequency = []
        for topic_idx, topic in enumerate(self.lda_topic_word):
            top_terms.append([self.vect.feature_names[i]
                              for i in topic.argsort()[:-num - 1:-1]])
            frequency.append([topic[i]
                              for i in topic.argsort()[:-num - 1:-1]])
            if show is True:
                logger.info("Topic #%d: " % topic_idx +
                            " ".join([self.vect.feature_names[i]
                                     for i in topic.argsort()[:-num - 1:-1]]))
        if freq is True:
            return top_terms, freq
        else:
            return top_terms

    def gen_labels(self):
        return np.argmax(self.lda_vec_data, axis=1)

    def export_csv_topic_word(self):
        tt_dict = {}
        for t_idx, topic in enumerate(self.lda_topic_word):
            tt_dict['Topic_'+str(t_idx)] = [str(self.vect.feature_names[i]) + '=' +
                                            str(topic[i]) for i in topic.argsort()[:-10 - 1:-1]]
        df = pd.DataFrame(tt_dict)
        file_name = self.name + '_' + self.vect.name + '_' + str(self.n_clusters) + '.csv'
        df.transpose().to_csv('../output/topic_word_' + file_name, sep=';')
        return file_name
