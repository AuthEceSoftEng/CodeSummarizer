import matplotlib.pyplot as plt
from presenter.presenter import Presenter
import numpy as np
import pandas as pd
import pdb
import os
import pickle


class PurityOptimizer(object):

    def __init__(self, clusterer):
        self.clust = clusterer

    def optimize_cluster_size(self, max_cl, file_name, min_cl=10):

        df = pd.DataFrame(columns=['Overall Purity'])
        for i in range(min_cl, max_cl):
            self.clust.cluster(i, n_init=10)
            e = Presenter()
            e.purity(self.clust.labels_true,
                     self.clust.labels_pred)
            df.loc[i] = e.overall_purity

        # Write dataframe to file
        f = open(file_name, 'wb')
        pickle.dump(df, f)
        f.close()

    def plot(self, file_name, path='optimizer/results/'):

        # Load the dataframe from the file
        f = open(file_name, 'rb')
        df = pickle.load(f)
        f.close()

        # Overall Purity
        fig, ax = plt.subplots(1)
        ax = df['Overall Purity'].plot(grid=True, legend=True, figsize=(40, 25))
        ax.set_xlabel('Clusters')
        fig.savefig(path+'Purity.svg', format='svg')


def main():
    clusterers = []
    for file in os.listdir('cached_clusterers'):
        f = open('cached_clusterers/'+file, 'rb')
        clusterers.append(pickle.load(f))
        f.close()
    purities = {}
    for cl in clusterers:
        e = Presenter()
        purities[cl.n_clusters] = e.purity(cl.labels_true, cl.labels_pred)
    df = pd.DataFrame(list(purities.items()), columns=('Topics',
                                                       'Overall Purity'))
    fig, ax = plt.subplots(1)
    pu_ax = df['Overall Purity'].plot(ax=ax, x='Topics', grid=True,
                                   legend=True, figsize=(40, 25))
    val = df['Overall Purity'].values
    xpoints = np.linspace(0, val.shape[0]-1, val.shape[0])
    poly = np.polyfit(xpoints, val, 2)
    poly = np.poly1d(poly)
    pol_ax = plt.plot(xpoints, poly(xpoints), 'r-', label='Fitted Polynomial')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_xlabel('Topics')
    fig.savefig('Purity.svg', format='svg')
    pdb.set_trace()

if __name__ == '__main__':
    main()
