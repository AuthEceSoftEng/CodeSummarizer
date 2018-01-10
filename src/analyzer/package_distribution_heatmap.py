from __future__ import division

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from collections import defaultdict


# From https://stackoverflow.com/a/34959983
def round_to_100(percents):
    def error_gen(tactual, trounded):
        return (abs(trounded - tactual) ** 2 / sqrt(1.0 if tactual < 1.0 else tactual)) if tactual > 0 else 0

    n = len(percents)
    rounded = [int(x) for x in percents]
    up_count = 100 - sum(rounded)
    errors = [(error_gen(percents[i], rounded[i] + 1) - error_gen(percents[i], rounded[i]), i) for i in range(n)]
    rank = sorted(errors)
    for i in range(up_count):
        rounded[rank[i][1]] += 1
    return rounded


# From https://stackoverflow.com/a/18926541
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def generate_heatmap(file_name):

    """Print plots from the run

    Parameters
    ----------
    file_name: string
              The file name of the csv containing the run results.
              Typically of the form: Algorithm_Vectorizer_NumberOfTopics.csv

    """

    plot_topic_participation = True

    ttype = file_name.strip("_")[1]    # vectorizer name

    # Read top terms for each topic
    top_terms = {}
    with open("topic_word_" + file_name) as infile:
        for line in infile:
            if line:
                line = line.strip().split(';')
                top_terms[line[0]] = []
                for l in line[1:]:
                    term = l.split('=')[0]
                    term = term[2:] if term.startswith('m_') else term
                    if term not in top_terms[line[0]]:
                        top_terms[line[0]].append(term)
    # top_terms['Topic_73'][4] = "follow"  # for visualization purposes
    # top_terms['Topic_127'][4] = "class"  # for visualization purposes

    # Read data in a dict, so that data[package][topic] contains the number of classes
    # of the package that are in this topic.
    data = defaultdict(lambda: defaultdict(int))
    with open("document_topic_" + file_name) as infile:
        next(infile)
        for line in infile:
            if line:
                line = line.strip().split(';')
                if line[2]:
                    data[line[2]][line[1]] += 1
    packages = data.keys()

    # Sort topics according to participation and keep only the first
    topics_limits = 20
    topics = set()
    for package in packages:
        topics |= set(data[package].keys())
    topic_participation = [(topic, sum([data[package][topic] for package in packages])) for topic in topics]
    topic_participation = sorted(topic_participation, key=lambda x: x[1], reverse=True)
    topics = [topic for topic, total_participation in topic_participation if total_participation > 0]

    if plot_topic_participation:
        # Plot topic participation
        fig = plt.figure(figsize=(5.0, 3.25))
        ax = plt.subplot(111)
        topicscores = [total_participation for topic, total_participation in topic_participation
                       if total_participation > 0]
        topicscores = [s / sum(topicscores) for s in topicscores]  # [:topics_limits]
        width = 0.85
        ax.bar(np.arange(len(topicscores)), topicscores, width, align='center', color="blue",
               linewidth=0.5)  # , edgecolor='black')
        box = ax.get_position()
        ax.set_position([box.x0 + 0.025, box.y0, box.width, box.height])
        # plt.xticks(np.arange(0, len(topicscores), 1), [t.split('_')[1] for t in topics])
        plt.xticks(np.arange(0, len(topicscores) + 6, 10), np.arange(0, len(topicscores) + 6, 10))
        plt.yticks(np.arange(0, 0.16, 0.03), [str(100 * s) + "%" for s in np.arange(0, 0.16, 0.03)])
        plt.axis((-0.75, len(topicscores) + 6 - 0.25, 0, 0.15))

        ax.tick_params(axis='both', labelsize=10)
        plt.xlabel("Number of Topics", fontsize=10)
        plt.ylabel("Participation", fontsize=10)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.9)
        plt.savefig("topic_distribution_" + ttype + ".eps", format='eps')
        plt.savefig("topic_distribution_" + ttype + ".pdf", format='pdf')
    plt.show()
    # exit()

    topics = topics[0:topics_limits]

    # for i, filename in enumerate(["document_topic_" + file_name.split('.')[0], "document_topic_" +
    #                                                                             file_name.split('.')[0] + "_gui"]):
    for i, filename in enumerate(["document_topic_" + file_name.split('.')[0]]):
        participation_threshold = 0.02

        # Read data in a dict, so that data[package][topic] contains the number of classes
        # of the package that are in this topic.
        data = defaultdict(lambda: defaultdict(int))
        with open(filename + ".csv") as infile:
            next(infile)
            for line in infile:
                if line:
                    line = line.strip().split(';')
                    if line[2]:
                        # if line[2] == "ensembleLibraryEditor": line[2] = "ensLibraryEditor"
                        #  for visualization purposes
                        data[line[2]][line[1]] += 1
        packages = data.keys()

        # For each package remove all topics with less than a class participation threshold
        # and all topics that are not in the top ones
        # and normalize all topics in a package to 1
        for package in packages:
            total_number_of_classes = sum(data[package].values())
            for topic, participation in data[package].items():
                if (topic not in topics) or (participation / total_number_of_classes < participation_threshold):
                    data[package][topic] = 0
            # if i == 0 and topic == "Topic_129" and package == "gui": # for visualization purposes
            # data[package][topic] *= 2
            total_number_of_classes = sum(data[package].values())
            if total_number_of_classes > 0:
                for topic, participation in data[package].items():
                    data[package][topic] = participation / total_number_of_classes
        packages = sorted([package for package in packages if sum(data[package].values()) > 0], reverse=True)

        # remove topics with very low participation
        finaltopics_participation = dict([(topic, sum([data[package][topic] for package in packages])) for topic in topics])
        finaltopics = [t for t in topics if finaltopics_participation[t] > 0]

        # print packages
        # print finaltopics
        npdata = np.zeros(shape=(len(packages), len(finaltopics)))
        for k, package in enumerate(packages):
            for l, topic in enumerate(finaltopics):
                npdata[k, l] = 100 * data[package][topic]
            npdata[k] = round_to_100(npdata[k])
        # print npdata
        fig = plt.figure(figsize=(12.5, 5.25))
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0 - 0.01, box.y0, box.width * 0.67, box.height])
        ax.tick_params(axis='both', labelsize=9.75)
        bplot = ax.pcolor(npdata, cmap=truncate_colormap(plt.get_cmap('YlOrRd'), 0.0, 0.95))
        x1, x2, y1, y2 = plt.axis()
        plt.axis((0, len(finaltopics), 0, len(packages)))
        plt.yticks(np.arange(0.5, len(packages) + 0.5, 1), packages)
        ax.xaxis.tick_top()
        ax.tick_params(axis=u'both', which=u'both', length=0)
        plt.xticks(np.arange(0.45, len(finaltopics) + 0.45, 1), [topic.split('_')[1] for topic in finaltopics])

        # Add percentages
        for y in range(npdata.shape[0]):
            for x in range(npdata.shape[1]):
                if npdata[y, x] == 100:
                    # plt.text(x + 0.5, y + 0.5, '%d%%' % npdata[y, x], fontsize=8.8, horizontalalignment='center',
                    #  verticalalignment='center')
                    plt.text(x + 0.10, y + 0.5, '1' % npdata[y, x], fontsize=8.9, horizontalalignment='center',
                             verticalalignment='center')
                    plt.text(x + 0.30, y + 0.5, '0' % npdata[y, x], fontsize=8.9, horizontalalignment='center',
                             verticalalignment='center')
                    plt.text(x + 0.525, y + 0.5, '0' % npdata[y, x], fontsize=8.9, horizontalalignment='center',
                             verticalalignment='center')
                    plt.text(x + 0.80, y + 0.5, '%%' % npdata[y, x], fontsize=8.9, horizontalalignment='center',
                             verticalalignment='center')
                elif npdata[y, x] == 0:
                    plt.text(x + 0.45, y + 0.525, '-' % npdata[y, x], fontsize=9.35, horizontalalignment='center',
                             verticalalignment='center')
                else:
                    plt.text(x + 0.5, y + 0.5, '%d%%' % npdata[y, x], fontsize=9.35, horizontalalignment='center',
                             verticalalignment='center')

        topics_terms = [
            ("  " + topic.split('_')[1] if len(topic.split('_')[1]) == 2 else topic.split('_')[1]) + " (" + ", ".join(
                top_terms[topic][:5]) + ")" for topic in finaltopics]
        # topics_terms = [("  " + topic.split('_')[1] if len(topic.split('_')[1]) == 2 else topic.split('_')[1])
        # + ": " + ", ".join(top_terms[topic][:5]) for topic in finaltopics]
        if i == 0:
            plt.legend([plt.Rectangle((0, 0), 0, 0, color="white", alpha=0)] * len(topics_terms), topics_terms, \
                       handlelength=-0.6, handleheight=0, ncol=1, bbox_to_anchor=(1.74, 1.018), prop={'size': 9.75})
        else:
            plt.legend([plt.Rectangle((0, 0), 0, 0, color="white", alpha=0)] * len(topics_terms), topics_terms, \
                       handlelength=-0.6, handleheight=0, ncol=1, bbox_to_anchor=(1.985, 1.018), prop={'size': 9.75})
        cbar = plt.colorbar(bplot, shrink=1.0 if i == 0 else 1.065, ticks=range(0, 101, 10))
        cbar.ax.tick_params(labelsize=9.75)
        cbar.ax.set_yticklabels(["%d%%" % s for s in range(0, 101, 10)])
        if i == 0:
            plt.subplots_adjust(left=0.1, right=0.80)
            cbox = cbar.ax.get_position()
            box = ax.get_position()
            cbar.ax.set_position([cbox.x0 - 0.061, cbox.y0, cbox.width * 0.67, cbox.height])
            ax.set_position([box.x0 + 0.02, box.y0, box.width * 0.90, box.height])
        else:
            plt.subplots_adjust(left=0.12, right=0.84)
            cbox = cbar.ax.get_position()
            box = ax.get_position()
            cbar.ax.set_position([cbox.x0 - 0.2065, cbox.y0 + 0.0245, cbox.width * 0.67, cbox.height])
            ax.set_position([box.x0 + 0.02, box.y0, box.width * 0.65, box.height * 1.065])
        plt.savefig(filename + "_heatmap.eps", format='eps')
        plt.savefig(filename + "_heatmap.pdf", format='pdf')
    plt.show()
