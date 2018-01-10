

class AbsVect(object):

    def __init__(self, name, corpus):

        self.name = name
        self.corpus = corpus

        self.vec_data = None
        self.feature_names = None
        self.vectorizer = None
