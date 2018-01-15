from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score, silhouette_samples

# from grapher.graph_json import GraphJSON

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('Cluster_tools')


def change_label_depth(labels_true, depth=0):

    # Remove maximum common subsequence
    flag = True
    while(True):
        check = labels_true[0].split('.')[0]
        for i in labels_true:
            if not(i.split('.')[0] == check):
                flag = False
        if flag:
            labels_true = ['.'.join(i.split('.')[1:]) for i in labels_true]
            continue
        else:
            break
    # Remove the file name from the label path
    #  labels_true = ['.'.join(l.split('.')[1:]) for l in labels_true]

    # Make label length equal to depth
    if depth <= 0:
        labels_true = [l.split('.')[0] for l in labels_true]
    elif depth > 0:
        for idx, label in enumerate(labels_true):
            if len(label.split('.')) > depth+1:
                labels_true[idx] = '.'.join(label.split('.')[0:depth])

    return labels_true


def calc_cluster_centers(vec_data, labels_pred, n_clusters):

    cluster_centers = np.zeros(shape=(n_clusters, vec_data.shape[1]))
    for i in range(n_clusters):
        ith_cluster_members = vec_data[labels_pred == i]
        cluster_centers[i] = np.mean(ith_cluster_members, axis=0)
    return cluster_centers


def cohesion_per_cluster(vec_data, cluster_centers, labels_pred, n_clusters):

    '''
    Calculate the cohesion per cluster using the euclidead distance of each
    member of a cluster to their cluster center.

    Parameters
    ----------
    vec_data        : Vectorized samples.
    cluster_centers : The centers of the clusters in vector space
    n_clusters      : The total number of clusters

    Returns
    -------
    sse : Numpy array of size n_clusters containing the cohesion for
          each cluster.
    '''

    sse = np.zeros(shape=n_clusters, dtype=np.double)
    for i in range(0, n_clusters):

        ith_cluster_vec_data = vec_data[labels_pred == i]
        dist = euclidean_distances(ith_cluster_vec_data,
                                   cluster_centers[i].reshape(1, -1))
        dist = dist**2
        sse[i] = sum(dist)
    return sse


def separation_per_cluster(cluster_centers, n_clusters):

    '''
    Calculate the separation per cluster using the euclidead distance of
    each cluster center to the centroid mean.

    Parameters
    ----------
    cluster_centers : The centers of the clusters in vector space
    n_clusters      : The total number of clusters

    Returns
    -------
    ssb : Numpy array of size n_clusters containing the separation for
          each cluster.
    '''

    ssb = np.zeros(shape=n_clusters, dtype=np.double)
    centroid_mean = np.mean(cluster_centers, axis=0).reshape(1, -1)

    for i in range(0, n_clusters):

        dist = euclidean_distances(cluster_centers[i].reshape(1, -1),
                                   centroid_mean)
        dist = dist**2
        ssb[i] = dist
    return ssb


def cohesion_separation_per_cluster(vec_data, labels_pred, n_clusters,
                                    cluster_centers, normalization='min_max',
                                    show_graphs=False, path='plots/'):
    '''
    Calculate cohesion and separation for each cluster, preprocess it and return
    it to a DataFrame.

    Parameters
    ----------
    vec_data        : Vectorized samples.
    labels_pred     : Predicted labels per sample.
    cluster_centers : The centers of the clusters in vector space.
    n_clusters      : The total number of clusters.
    normalization   : Technique to normalize sse and ssb (min_max, None).
    show_graphs     : Boolean indicating whether to show graphs.
    path            : Path to save the graphs (relative to main).

    Returns
    -------
    df  : Pandas Dataframe containing cohesion and separation per cluster,
          properly normalized.
    '''

    sse = cohesion_per_cluster(vec_data, cluster_centers, labels_pred,
                               n_clusters)
    ssb = separation_per_cluster(cluster_centers, n_clusters)

    if normalization is 'min_max':
        scaler = MinMaxScaler()
        sse = scaler.fit_transform(sse.reshape(-1, 1))
        ssb = scaler.fit_transform(ssb.reshape(-1, 1))
    else:
        sse = sse.reshape(-1, 1)
        ssb = ssb.reshape(-1, 1)

    sse = sse.transpose()
    ssb = ssb.transpose()
    df = pd.DataFrame(dict(Cohesion=sse[0], Separation=ssb[0]))

    if show_graphs is True:
        df.plot(kind='bar', subplots=True, sharex=False,
                grid=True, figsize=(40, 25))
        fig = plt.gcf()
        fig.savefig(path+'coh_sep_per_cluster_c' +
                    str(n_clusters)+'.svg', format='svg')

    return df


def similarity_matrix(dist):
    #  arg = np.argsort(labels_pred)
    #  dist = dist[arg, :]
    #  dist = dist[:, arg]
    my_xticks = ['Topic #{}'.format(idx) for idx in range(len(dist))]
    plt.xticks(range(len(my_xticks)), my_xticks, rotation=45)
    plt.yticks(range(len(my_xticks)), my_xticks, rotation=45)
    fig = plt.figure()
    plt.imshow(dist, interpolation='nearest')
    plt.set_cmap('autumn')
    fig.show()


def by_topic_percentage(labels_pred, labels_true, num_topics=5,
                        show_zeros=False):

    max_cl = max(labels_pred)

    df = pd.DataFrame(dict(labels_pred=labels_pred,
                           labels_true=labels_true))
    groups = df.groupby('labels_true')

    name_fmt = '{:<35}'
    num_fmt = '{0:3.0f}'
    ending = ' | '

    graph = GraphJSON()
    for i in range(max_cl+1):
        graph.add_vertex('TC' + str(i))

    # Print label_true and how it was distributed to the labels_pred
    for label_tr, group in groups:

        graph.add_vertex(label_tr)

        lp = list(group.labels_pred)
        num_lp = len(lp)

        pct_array = np.zeros(shape=max_cl+1)
        for ind in range(0, max_cl+1):
            pct_array[ind] = (lp.count(ind)/num_lp)*100
        sort = pct_array.argsort()[::-1][:]

        out_line = ''
        # put padding with -- if label_true is a subdirectory
        for b in range(0, len(label_tr.split('.'))-1):
            out_line = out_line + '--'
        out_line = name_fmt.format(out_line + label_tr) + ending

        # itterate over topic clusters
        for ind in sort[:num_topics]:
            pct = pct_array[ind]
            if pct < 0:
                continue
            else:
                # display the percent and the Topic Cluster
                out_line = out_line + num_fmt.format(pct) + '% TC' + \
                           '{0:3}'.format(ind) + ending
                graph.add_edge(label_tr, 'TC'+str(ind), pct)
        logger.info(out_line)
    graph.save('grapher/graph.json')
    return graph.data


def by_topic(labels_pred, labels_true, show_zeros=False):

    max_cl = max(labels_pred)

    distribution = {}

    df = pd.DataFrame(dict(labels_pred=labels_pred,
                           labels_true=labels_true))
    groups = df.groupby('labels_true')

    name_fmt = '{:<30}'
    num_fmt = '{:<3}'
    ending = '|'

    # Print cluster indices at the top
    str = ''
    str = str + name_fmt.format('Cluster Paths') + ending
    for ind in range(0, max_cl+1):
        str = str + num_fmt.format(ind) + ending
    logger.info(str)

    # Print separator between cluster indices and results
    str = ''
    str = name_fmt.format(str + '=================') + ending
    for ind in range(0, max_cl+1):
        str = str + num_fmt.format('===') + ending
    logger.info(str)

    # Print label_true and how it was distributed to the labels_pred
    for label_tr, group in groups:

        distribution[label_tr] = np.zeros(shape=max_cl+1)
        lp = list(group.labels_pred)
        str = ''

        # put padding with -- if label_true is a subdirectory
        for b in range(0, len(label_tr.split('.'))-1):
            str = str + '--'
        str = name_fmt.format(str + label_tr) + ending

        for ind in range(0, max_cl+1):
            count = lp.count(ind)
            distribution[label_tr][ind] = count
            if count == 0 and show_zeros is False:
                str = str + num_fmt.format(' ') + ending
            else:
                str = str + num_fmt.format(lp.count(ind)) + ending
        logger.info(str)
    return distribution

def silhouette_per_cluster(vec_data, labels_pred,
                           n_clusters, path='plots/'):

    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1)
    fig.set_size_inches(50, 30)

    # The silhouette coefficient can range from -1, 1
    # but in this example all lie within [-0.1, 1]
    ax1.set_xlim([-0.3, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(labels_pred) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(vec_data, labels_pred)
    print('For n_clusters =', n_clusters,
          'The average silhouette_score is :', silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(vec_data, labels_pred)

    df = pd.DataFrame(columns=['Size', 'Silhouette'])

    y_lower = 0
    for i in range(0, n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels_pred == i]

        df.loc[i] = [len(ith_cluster_silhouette_values),
                     ith_cluster_silhouette_values.mean()]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their
        # cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title('The silhouette plot for the various clusters.')
    ax1.set_xlabel('The silhouette coefficient values')
    ax1.set_ylabel('Cluster label')

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
    ax1.axvline(x=0.2, color='blue', linestyle='--')
    ax1.axvline(x=0.4, color='blue', linestyle='--')
    ax1.axvline(x=0.6, color='blue', linestyle='--')

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    fig.savefig(path+'Silhouette_per_cluster_fancy_c' +
                str(n_clusters)+'.svg', format='svg')

    fig = plt.figure()
    df.plot(kind='bar', legend=True, grid=True,
            subplots=True, figsize=(40, 25))
    plt.savefig(path+'Silhouette_per_cluster_c' +
                str(n_clusters)+'.svg', format='svg')

    fig = plt.figure()
    df['Silhouette'].plot(kind='hist', legend=True, figsize=(40, 25))
    fig.savefig(path+'Silhouette_histogram_c' +
                str(n_clusters)+'.svg', format='svg')
