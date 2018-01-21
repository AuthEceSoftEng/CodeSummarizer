#! /usr/bin/python3

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os.path
import sys
import csv
from time import time, strftime

import matplotlib

python_version = (sys.version_info.major, sys.version_info.minor)
required_version = (3, 5)
if python_version >= required_version:
    matplotlib.use('Qt5Agg')

from extractor.extractor import Extractor
from preprocessor.preprocessor import Preprocessor

from vectorizer.tfidfvectorizer import TfidfVect
from vectorizer.countvectorizer import CountVect

from clusterer.cluster_tools import change_label_depth
from clusterer.km_clusterer import KMClust
from clusterer.lda_clusterer import LDAClust

from optimizer.optimizer import Optimizer

from presenter.presenter import create_wordclouds
from presenter.package_distribution_heatmap import generate_heatmap

from searcher.official_googler import OfficialGoogler


parser = argparse.ArgumentParser()
parser.add_argument('--reload_extraction', action='store_true', help='force to redo the extraction of the dataset (do not use previous run)')
parser.add_argument('--reload_preprocessing', action='store_true', help='force to redo the preprocessing on the extracted (do not use previous run)')
parser.add_argument('--dataset', '-d', type=str, required=True, help='The folder contraining the dataset')
parser.add_argument('--algorithm', '-a', type=str, required=True, help='The algorithm to be used for clustering. ' +
                    'Available options are \'km\' and \'lda\'')
parser.add_argument('--pkg_start', type=str, required=False, help='Package start to keep (useful for excluding certain subpackages of a project)')
parser.add_argument('--vectorizer', '-v', type=str, required=True, help='The vectorizer to be used. ' +
                    'Available options are \'tfidf\' and \'count\'')
parser.add_argument('--n_clusters', '-n', type=int, required=True, help='The number of clusters')
parser.add_argument('--ldepth', '-l', type=int, required=True, help='The depth of the package structure to be used as ground truth labels')
parser.add_argument('--search', '-s', action='store_true', help='Enable search of tags for \'good\' clusters')
parser.add_argument('--optimize', '-o', action='store_true', help='Run optimizer module from 10 to N_CLUSTERS')
parser.add_argument('--verbose', '-V', action='store_true', help='Show output on screen (alt only saved in log file)')
# Parse given arguments.
args = parser.parse_args()

# Setup logging parameters.
timestr = strftime('%m.%d-%H.%M.%S')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s%(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/main '+timestr+'.log',
                    filemode='w')

# If verbose is set, output logs to sonsole as well.
if args.verbose is True:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-2s: %(levelname)-2s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# Extract the classes from the dataset. Use cached version if available or not specified otherwise.
logging.info('Using dataset {}'.format(args.dataset))

t0 = time()
a = Extractor()
cache_path = '../dataset/cache/' + args.dataset + '.pckl'

if os.path.isfile(cache_path) and args.reload_extraction is False:
    logging.info('########## LOADING DATASET FROM CACHE ##########')
    a.load(cache_path)
else:
    logging.info('##########     EXTRACTING DATASET     ##########')
    dataset_path = '../dataset/raw/' + args.dataset
    if not os.path.exists(dataset_path):
        sys.exit('Specified dataset not found in dataset folder. Aborting')
    a.clean_dataset(dataset_path)
    a.extr_folder_classes(dataset_path)
    a.save(cache_path)
logging.info('Finished extracting {0:.4f}s'.format(time()-t0))

# Preprocess extracted dataset. Use cached version if available or not specified otherwise.
t0 = time()
cache_path = '../dataset/cache/' + args.dataset + '_prep.pckl'
if os.path.isfile(cache_path) and (args.reload_preprocessing is False and args.reload_extraction is False):
    logging.info('########## LOADING PRERPOCESSED DATA FROM CACHE ##########')
    b = Preprocessor()
    b.load(cache_path)
else:
    logging.info('########## PREPROCESSING DATASET ##########')
    b = Preprocessor(a.classes, type='class', pkg_start=args.pkg_start)
    b.save(cache_path)
logging.info('Finished preprocessing {0:.4f}s'.format(time()-t0))

# Choose appropriate vectorizer and initialize
if args.vectorizer == 'tfidf':
    v = TfidfVect(b.corpus)
elif args.vectorizer == 'count':
    v = CountVect(b.corpus)
else:
    logging.error('Vectorizer name not recognized. Exiting...')
    sys.exit()

# Change label depth according to argument given by user. Purposely kept in main (instead of preprocessor to be able to
# change without redoing the preprocessing)
labels_true = change_label_depth(b.labels, depth=args.ldepth)

# Choose appropriate clusterer and initialize.
if args.algorithm == 'km':
    c = KMClust(v, labels_true)
elif args.algorithm == 'lda':
    c = LDAClust(v, labels_true)
else:
    logging.error('Clusterer name not recognized. Exiting...')
    sys.exit()

logging.info('Clustering...')
o = Optimizer()
o.optimize(c, args.n_clusters, one_run=(not args.optimize))

o.latest_clusterer.export_csv_doc_topic()
file_name = o.latest_clusterer.export_csv_topic_word()

# Generate the heatmap for the last clusterer
generate_heatmap(file_name)

# Plot the Purities and Topic categories so far
if args.optimize:
    o.plot_current()

if args.search:
    # Compute wordclouds for the latest clusterer
    og = OfficialGoogler()
    tags = []
    with open('../output/tags.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, c.n_clusters):
            og.search(' '.join(c.top_terms_per_cluster(num=5)[i]) + ' ' + args.dataset, '../output/result{}.json'.format(i))
            tags.append(og.fetch_tags('../output/result{}.json'.format(i)))
            csv_writer.writerow([str(i)] + [str(y)+'='+str(x) for (x, y) in tags[-1]])

    top_tags = {}
    topics = []
    with open("../output/tags.csv") as infile:
        for line in infile:
            if line:
                line = line.strip().split(';')
                top_tags[line[0]] = []
                topics.append(line[0])
                for l in line[1:]:
                    term = l.split('=')[0]
                    top_tags[line[0]].append(term)

    tags = []
    topic_ids = []
    for topic in topics:
        if top_tags[topic]:
            tags.append(top_tags[topic][:5])
            topic_ids.append(topic)

create_wordclouds('tags.csv', file_name)
logging.info('Finished execution')
