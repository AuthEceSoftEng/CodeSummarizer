#! /usr/bin/python3

import pandas as pd

class AbsClust(object):

    def __init__(self, name, vect, labels_true):

        self.name = name
        self.labels_true = labels_true
        self.labels_pred = None

        self.vect = vect

        self.n_clusters = None

    def cluster(self, n_clusters):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError

    def top_terms_per_cluster(self, num, show):
        raise NotImplementedError

    def export_csv_topic_word(self):
        raise NotImplementedError

    def export_csv_doc_topic(self):


        lp = ['Topic_' + str(x) for x in self.labels_pred.tolist()]
        df = pd.DataFrame(dict(labels_true=self.labels_true, labels_pred=lp))
        file_name = self.name + '_' + self.vect.name + '_' + str(self.n_clusters) + '.csv'
        df.to_csv('../output/document_topic_' + file_name, sep=';')
        return file_name

