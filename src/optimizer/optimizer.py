import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

from presenter.presenter import (calc_purity, calc_topic_categories)

logger = logging.getLogger('Optimizer')


class Optimizer(object):

    def __init__(self):
        self.purities = {}
        self.topic_categories = {}
        self.latest_clusterer = None

    def examine(self, clusterer):

        self.latest_clusterer = clusterer
        self.purities[clusterer.n_clusters] = calc_purity(clusterer.labels_true, clusterer.labels_pred,
                                                          vectorizer=clusterer.vectorizer,
                                                          top_terms=clusterer.top_terms_per_cluster(show=False))

        self.topic_categories[clusterer.n_clusters] = calc_topic_categories(clusterer.top_terms_per_cluster(),
                                                                            clusterer.vectorizer)
        logger.info('Clustered with {}. Overall purity: {}. Number of categories {}'.
                    format(clusterer.n_clusters, self.purities[clusterer.n_clusters],
                           self.topic_categories[clusterer.n_clusters]))


    def optimize(self, clusterer, n_clusters, one_run=False):

        if one_run is True:
            logger.info('OneRun Enabled. Clustering with {} clusters...'.format(n_clusters))
            clusterer.cluster(n_clusters=n_clusters)
            self.examine(clusterer)
        else:
            for i in range(10, n_clusters):
                logger.info('Clustering with {} clusters...'.format(i))
                clusterer.cluster(n_clusters=i)
                self.examine(clusterer)
                self.plot_current()

    def plot_current(self):

        df = pd.DataFrame(list(self.purities.items()), columns=('Topics', 'Overall Purity'))
        df1 = pd.DataFrame(list(self.topic_categories.items()), columns=('Topics', 'Number of Topic Categories'))

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # Plot Purities
        df['Overall Purity'].plot(ax=ax1, x='Topics', grid=True, legend=True, figsize=(40, 25))

        # Calculate and plot fitted polynomial
        val = df['Overall Purity'].values
        x_points = np.linspace(0, val.shape[0]-1, val.shape[0])
        poly = np.polyfit(x_points, val, 2)
        poly = np.poly1d(poly)
        ax1.plot(x_points, poly(x_points), 'r-', label='Fitted Polynomial')

        # Subplot maintenance
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, fontsize=40, loc=0)
        ax1.set_xlabel('Topics', fontsize=40)
        ax1.set_ylabel('Overall Purity', fontsize=40)
        ax1.tick_params(axis='x', labelsize=40)
        ax1.tick_params(axis='y', labelsize=40)

        # Plot Topic Categories
        df1['Number of Topic Categories'].plot(ax=ax2, x='Topics', grid=True, legend=True, figsize=(40, 25))

        # Calculate and plot fitted polynomial
        val = df1['Number of Topic Categories'].values
        xpoints = np.linspace(0, val.shape[0]-1, val.shape[0])
        poly = np.polyfit(xpoints, val, 2)
        poly = np.poly1d(poly)
        ax2.plot(xpoints, poly(xpoints), 'r-', label='Fitted Polynomial')

        # Subplot maintenance
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels, fontsize=40, loc=0)
        ax2.set_xlabel('Topics', fontsize=40)
        ax2.set_ylabel('Topic categories', fontsize=40)
        ax2.tick_params(axis='x', labelsize=40)
        ax2.tick_params(axis='y', labelsize=40)

        final = pd.merge(df, df1, on='Topics', how='outer')
        final.to_csv('../output/Latest_Optimization_data.csv', sep=';')
        plt.show()

        # fig.savefig('../output/Latest_Optimization_plot.eps', format='eps')
