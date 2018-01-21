from __future__ import division
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import re
from wordcloud import WordCloud


def calc_purity(labels_true, labels_pred, vectorizer, top_terms):

    num = len(top_terms)

    # Transform top term corpus in a weighted top term corpus
    weighted_top_terms = []
    for topic in top_terms:
        weighted_list = []
        for ind, term in enumerate(topic):
            for i in range(0, num - ind + 1):
                weighted_list.append(term)
        weighted_top_terms.append(weighted_list)

    for ind, wl in enumerate(weighted_top_terms):
        weighted_top_terms[ind] = ' '.join(weighted_top_terms[ind])

    # Transform weighted top terms to vector space
    v_weighted_top_terms = vectorizer.vectorizer.transform(weighted_top_terms)

    # Take cosine similarity between topics
    sim = cosine_similarity(v_weighted_top_terms)

    df = pd.DataFrame(dict(lp=labels_pred,
                           lt=labels_true))
    groups = df.groupby('lt')

    package_purities = {}
    for lt, group in groups:

        counts = group.groupby('lp').count().sort_values(by=['lt'],
                                                         ascending=0)
        max_count = group.groupby('lp').count()['lt'].max(axis=0)
        max_topic = group.groupby('lp').count()['lt'].idxmax(axis=0)
        sum_count = group.groupby('lp').count()['lt'].sum(axis=0)

        counts.reset_index(level=0, inplace=True)
        acc = 0
        for i, row in counts.iterrows():
            if (row['lp'] != max_topic) and (sim[row['lp']][max_topic] > 0.2):
                acc += row['lt']
        package_purities[lt] = (max_count + acc)/sum_count
    temp = list(package_purities.values())
    return np.mean(temp)


def calc_topic_categories(top_terms, vectorizer):

    num = len(top_terms)

    weighted_top_terms = []
    for topic in top_terms:
        weighted_list = []
        for ind, term in enumerate(topic):
            for i in range(0, num - ind + 1):
                weighted_list.append(term)
        weighted_top_terms.append(weighted_list)

    for ind, wl in enumerate(weighted_top_terms):
        weighted_top_terms[ind] = ' '.join(weighted_top_terms[ind])

    # Transform weighted top terms to vector space
    v_weighted_top_terms = vectorizer.vectorizer.transform(weighted_top_terms)

    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics.pairwise import cosine_distances

    # Take cosine similarity between topics
    cos_dis = cosine_distances(v_weighted_top_terms)
    z = linkage(cos_dis, method='complete', metric='cosine')
    t = 0.4
    kappa = fcluster(z, t)
    print('Num categories: {}'.format(len(set(kappa))))
    return len(set(kappa))


def create_wordclouds(tag_csv, topic_word_csv):

    random_state = 1
    pref = '../output/'
    tag_csv = pref + tag_csv

    topic_word_csv = pref + 'topic_word_' + topic_word_csv
    use_frequencies = True
    print_full = False
    if print_full:
        img_size = (100, 150)
        figsize = (20, 10)
    else:
        figsize = (7, 3.5)
        img_size = (int(300 * figsize[0]), int(300 * figsize[1]))
    # Select the topics to print (only if print full is false
    selected = []

    top_tags = {}
    topics = []
    # with open(".." + os.sep + ".." + os.sep + "topic_tags_5_weka.csv") as infile:
    with open(tag_csv) as infile:
        for line in infile:
            if line:
                line = line.strip().split(';')
                top_tags[line[0]] = []
                topics.append(line[0])
                if line[1:] is not []:
                    selected.append(line[0])
                    for l in line[1:]:
                        term = l.split('=')
                        top_tags[line[0]].append(term)

    top_terms_str = {}
    with open(topic_word_csv) as infile:
        next(infile)
        for line in infile:
            if line:
                line = line.strip().split(';')
                top_terms_str[line[0]] = []
                for l in line[1:]:
                    print(l)
                    term = l.split('=')
                    print(term)
                    term = (term[0][2:] if term[0].startswith('m_') else term[0], term[1])
                    top_terms_str[line[0]].append(term)

    top_terms = {}
    for topic in topics:
        top_tags[topic] = [(re.sub(r'[^a-zA-Z0-9]', '', t[0]), int(t[1])) for t in top_tags[topic] if t[0] != 'java' and t[0] != 'weka'][:5]
        top_terms[topic] = [(re.sub(r'[^a-zA-Z0-9]', '', t[0]), float(t[1])) for t in top_terms_str['Topic_' + str(topic)][:5]]

    if print_full:
        f, axarr = plt.subplots(4, 5, figsize = figsize)
    else:
        f, ax = plt.subplots(1, 1, figsize = figsize)
    for tn, topic in enumerate([topic for topic in topics if print_full or topic in selected]):
        # print topic
        # print top_tags[topic]
        # print top_terms[topic]
        # print
        if use_frequencies:
            top_terms_frequencies = dict([(t[0], t[1] / max(s[1] for s in top_terms[topic])) for t in top_terms[topic]])
            top_tags_frequencies = dict([(t[0], t[1] / max(s[1] for s in top_tags[topic])) for t in top_tags[topic]])
            frequencies = top_terms_frequencies
            frequencies.update(top_tags_frequencies)
            ccc = lambda word, font_size, position, orientation, font_path, random_state: 'rgb(255, 0, 0)' \
                if word in [top_tag[0] for top_tag in top_tags[topic]] else 'rgb(0, 0, 255)'
            wc = WordCloud(background_color="white", prefer_horizontal = 1, stopwords = set(), relative_scaling = 0.4,
                           min_font_size = 100, max_font_size = 400, color_func = ccc, width = img_size[0],
                           height = img_size[1], random_state = random_state)
            wc.generate_from_frequencies(frequencies)
        else:
            top_terms_frequencies = dict([(t[0], (5 - i) / 5) for i, t in enumerate(top_terms[topic])])
            top_tags_frequencies = dict([(t[0], (5 - i) / 5) for i, t in enumerate(top_tags[topic])])
            frequencies = top_terms_frequencies
            frequencies.update(top_tags_frequencies)
            ccc = lambda word, font_size, position, orientation, font_path, random_state: 'rgb(255, 0, 0)' \
                if word in [top_tag[0] for top_tag in top_tags[topic]] else 'rgb(0, 0, 255)'
            wc = WordCloud(background_color="white", prefer_horizontal = 1, stopwords = set(), relative_scaling = 0.4,
                           min_font_size = 100, max_font_size = 400, color_func = ccc, width = img_size[0],
                           height = img_size[1], random_state = random_state)
            wc.generate_from_frequencies(frequencies)

        if print_full:
            axarr[tn % 4, tn // 4].imshow(wc, interpolation="bilinear")
            axarr[tn % 4, tn // 4].set_title("Topic " + topic)
            axarr[tn % 4, tn // 4].axis("off")
        else:
            ax.imshow(wc, interpolation="bilinear")
            # ax.set_title("Topic " + topic)
            ax.axis("off")
            plt.subplots_adjust(left = 0.03, bottom = 0.03, right = 0.97, top = 0.97)
            plt.savefig(pref + 'wordcloud_' + topic + '.png', format='png')#, bbox_inches='tight', pad_inches = 0)
            plt.savefig(pref + 'wordcloud_' + topic + '.eps', format='eps')#, bbox_inches='tight', pad_inches = 0)
            plt.savefig(pref + 'wordcloud_' + topic + '.pdf', format='pdf')#, bbox_inches='tight', pad_inches = 0)

    if print_full:
        plt.savefig(pref + 'wordclouds_full.eps', format='eps')
        plt.savefig(pref + 'wordclouds_full.pdf', format='pdf')
        plt.show()
