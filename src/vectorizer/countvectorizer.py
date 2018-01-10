from __future__ import print_function
from __future__ import absolute_import

from time import time
import logging

from sklearn.feature_extraction.text import CountVectorizer

from vectorizer.abstract_vectorizer import AbsVect

logger = logging.getLogger('CountVect')


class CountVect(AbsVect):

    def __init__(self, corpus):

        '''Vectorize the corpus provided.

        Parameters
        ----------
        corpus : 2D list. Rows are documents and Columns are words.

        '''
        super(CountVect, self).__init__("Count", corpus)

        self.corpus = [' '.join(d) for d in self.corpus]

        logger.info("Extracting features from the training corpus using" +
                    " a count vectorizer")

        t0 = time()

        count_vectorizer = CountVectorizer()
        self.vec_data = count_vectorizer.fit_transform(self.corpus)
        self.feature_names = count_vectorizer.get_feature_names()
        self.vectorizer = count_vectorizer

        logger.info("done in %fs" % (time() - t0))
        logger.info("n_samples: %d, n_features: %d" % self.vec_data.shape)
