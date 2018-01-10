from __future__ import print_function
from __future__ import absolute_import

from time import time
import logging

from sklearn.feature_extraction.text import TfidfVectorizer

from vectorizer.abstract_vectorizer import AbsVect

logger = logging.getLogger('TfidfVect')


class TfidfVect(AbsVect):

    def __init__(self, corpus):

        '''Vectorize the corpus provided.

        Parameters
        ----------
        corpus : 2D list. Rows are documents and Columns are words.

        '''
        super(TfidfVect, self).__init__("Tfidf", corpus)

        self.corpus = [' '.join(d) for d in self.corpus]

        logger.info("Extracting features from the training corpus using" +
                    " a sparse tf-idf vectorizer")

        t0 = time()

        tfidf_vectorizer = TfidfVectorizer(max_df=0.70, max_features=None,
                                           min_df=0.08,
                                           use_idf=True, sublinear_tf=True)
        self.vec_data = tfidf_vectorizer.fit_transform(self.corpus)
        self.feature_names = tfidf_vectorizer.get_feature_names()
        self.vectorizer = tfidf_vectorizer

        logger.info("done in %fs" % (time() - t0))
        logger.info("n_samples: %d, n_features: %d" % self.vec_data.shape)
