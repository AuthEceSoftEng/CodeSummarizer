#!/usr/bin/python3
from __future__ import absolute_import

import logging
import re
import pickle
import sys
import six
import pdb

from extractor.extractor import Extractor
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, porter
from nltk.tokenize import WordPunctTokenizer

logger = logging.getLogger('Preprocessor')


class Preprocessor(object):

    def __init__(self, class_list=None, type='class', pkg_start=None):

        self.regexp = re.compile('([A-Z]+[a-z]|[a-z]+)([A-Z])')  # decamelcase
        if class_list is not None:
            if type == 'class':
                self.class_corpus(class_list, pkg_start=pkg_start)
            elif type == 'method':
                self.method_corpus(class_list, pkg_start=pkg_start)
            else:
                logger.error("Incorrect type declaration for Preprocessor")
        else:
            self.corpus = []
            self.labels = []
            self.paths = []

    def class_corpus(self, class_list, pkg_start=None):

        '''Create class level corpus

        Every document in the corpus consist of the following words:
        -- class, method and variable name (decamelcased)
        -- class, method and variable javadocs (cleaned)
        -- method block simplenames (decamelcased)
        The label for each document is the path to the class.
        '''
        self.corpus = []
        self.labels = []
        self.paths = []
        logger.info('Creating Class level corpus')
        logger.info('Label: Package Declaration')
        logger.info('Adding variable names(d) and javadocs(c)')
        logger.info('Adding class javadoc')
        logger.info('Adding method names(d), javadocs(c) and blocks(d)')
        for cl in class_list:
            words = []
            words.extend(self.clean_javadoc(' '.join(cl.get_javadoc())))
            for index, var_j in enumerate(cl.get_var_javadoc()):
                var_name = cl.get_var_names()[index][0]
                words.extend(self.break_camel_case(var_name))
                words.extend(self.clean_javadoc(' '.join(var_j)))
            for index, method_j in enumerate(cl.get_method_javadoc()):
                method_name = cl.get_method_names()[index][0]
                words.extend(self.break_camel_case(method_name))
                words.extend(self.clean_javadoc(' '.join(method_j)))
                method_b = cl.get_method_block()[index]
                words.extend(self.break_camel_case(' '.join(method_b)))
            self.corpus.append(words)
            self.paths.append(cl.path)  # cl.path() returns string
            if isinstance(cl.package, six.string_types):
                self.labels.append(cl.package)
            else:
                logger.error('Package Declaration {} is list in file {}'
                             .format(cl.package, cl.path))
                sys.exit(-1)
            # self.corpus[-1] = ' '.join(self.corpus[-1])
        self.clean_packages(pkg_start=pkg_start)
        self.get_low()
        self.replace_common()
        self.tokenize()
        self.lemmatize()
        self.remove_stopwords()

    def clean_packages(self, pkg_start):

        if pkg_start is not None:
            logger.info('Cleaning Packages not starting with {}'
                        .format(pkg_start))
            keep = []
            for index, label in enumerate(self.labels):
                if label.split('.')[0] == pkg_start:
                    keep.append(index)
            self.labels = [lab for index, lab in enumerate(self.labels)
                           if index in keep]
            self.corpus = [corp for index, corp in enumerate(self.corpus)
                           if index in keep]

    def get_low(self):

        logger.info('Getting very very low')
        for index, cor in enumerate(self.corpus):
            self.corpus[index] = [word.lower() for word in cor]

    def method_corpus(self, class_list):

        '''Create method level corpus

        Every document in the corpus consists of the following words:
        method names (decamelcased) and the method javadocs (cleaned).
        The label for each document is a list with the class path
        of the class the method belongs to and the name of the method.
        '''

        self.corpus = []
        self.labels = []
        logger.info('Creating Method level corpus')
        logger.info('Label: [Path to class, Method Name]')
        logger.info('Perform: replace_common, tokenize(remove_tokens = True),' +
                    'remove_stopwords, lemmatize')

        for cl in class_list:
            for index, method_j in enumerate(cl.get_method_javadoc()):
                words = []
                name = []

                name.extend(cl.path.split())
                name.extend(cl.get_method_names()[index][0].split())
                words.extend(self.clean_javadoc(' '.join(method_j)))
                words.extend(self.break_camel_case(
                                    cl.get_method_names()[index][0]))

                self.corpus.append(words)
                self.labels.append(name)
                # self.corpus[-1] = ' '.join(self.corpus[-1])
        self.replace_common()
        self.tokenize()
        self.remove_stopwords()
        self.lemmatize()

    def break_camel_case(self, word):

        '''Break all camel case words in a string

        word -- string of words seperated by spaces
        returns -- list of all words in the string
        '''

        return self.regexp.sub(r'\1 \2', word).split()

    def clean_javadoc(self, jdoc):

        # break camel case
        jdoc = self.regexp.sub(r'\1 \2', jdoc)
        # seperate same words seperated by /
        jdoc = re.sub('([a-zA-Z])/([a-zA-Z])', r'\1 \2', jdoc)
        # seperate words seperated by _ (not tested)
        # jdoc = re.sub('([a-zA-Z])_([a-zA-Z])', r'\1 \2', jdoc)
        # remove preformatted code
        jdoc = re.sub('<pre>(.*?)</pre>', r'', jdoc)
        # remove $Revision$ statements
        jdoc = re.sub('\$(.*?)\$', r'', jdoc)
        # remove email declarations
        jdoc = re.sub('\((.*?)at(.*?)dot(.*?)\)', r'', jdoc)
        jdoc = re.sub('\((.*?)\@(.*?)\)', r'', jdoc)
        # remove the <code> sandwich
        jdoc = re.sub('<( *)CODE>(.*?)<(.*?)CODE>', r'\2', jdoc)
        # remove trailing () from method names in comments
        jdoc = re.sub('(.*?)\(\)', r'\1', jdoc)
        # remove (* or *)
        jdoc = re.sub('(.*?)\)', r'\1', jdoc)
        jdoc = re.sub('\((.*?)', r'\1', jdoc)
        # remove all <*>
        return re.sub('<(.*?)>', r' ', jdoc).split()

    def replace_common(self):

        logger.info('Replacing common abbreviations')
        replacement_patterns = [
                (r'won\'t', 'will not'),
                (r'can\'t', 'cannot'),
                (r'i\'m', 'i am'),
                (r'ain\'t', 'is not'),
                (r'(\w+)\'ll', '\g<1> will'),
                (r'(\w+)n\'t', '\g<1> not'),
                (r'(\w+)\'ve', '\g<1> have'),
                (r'(\w+)\'s', '\g<1> is'),
                (r'(\w+)\'re', '\g<1> are'),
                (r'(\w+)\'d', '\g<1> would')
                ]

        patterns = [(re.compile(regex), rep1)
                    for (regex, rep1) in replacement_patterns]

        for index, co in enumerate(self.corpus):
            text = ' '.join(co)
            for (patt, rep1) in patterns:
                text = re.sub(patt, rep1, text)
            self.corpus[index] = text.split()

    def tokenize(self, remove_tokens=True):

        logger.info('Tokenizing')
        tokens = ['.', ',', '-', '\'', ':', ';', '&', '\"', ']', '[', '][',
                  ';]\"', '[&', '].', '],', '*', '=', '/', '+', '.,', ';-&',
                  '\"\"', '\"&', '{[', ']}', '[]']
        tokenizer = WordPunctTokenizer()
        for index, cor in enumerate(self.corpus):
            self.corpus[index] = [tokenizer.tokenize(word.lower())
                                  for word in cor]
            self.corpus[index] = [word for sublist in self.corpus[index]
                                  for word in sublist]
            if remove_tokens is True:
                self.corpus[index] = [word for word in self.corpus[index]
                                      if word not in tokens]
                self.corpus[index] = [word for word in self.corpus[index]
                                      if len(word) > 1]

    def remove_stopwords(self, english=True, java=True, conventions=True):

        logger.info('Removing Stopwords')
        stops = []
        if (english is True):
            stops.extend(stopwords.words('english'))
            logger.info('Removing English Stopwords')
        if (java is True):
            stops.extend(open('./preprocessor/java_stopwords.txt')
                         .read().splitlines())
            logger.info('Removing Java Stopwords')
        if (conventions is True):
            stops.extend(open('./preprocessor/conventions.txt').read().splitlines())
            logger.info('Removing Java Conventions')
        for index, cor in enumerate(self.corpus):
            self.corpus[index] = [word.lower() for word in cor
                                  if word.lower() not in stops]

    def lemmatize(self):

        logger.info('Lemmatizing')
        lemmatizer = WordNetLemmatizer()
        for index, cor in enumerate(self.corpus):
            self.corpus[index] = [lemmatizer.lemmatize(word.lower())
                                  for word in cor]

    def stem(self):

        logger.info('Stemming')
        stemmer = porter.PorterStemmer()
        for index, cor in enumerate(self.corpus):
            self.corpus[index] = [stemmer.stem(word.lower())
                                  for word in cor]

    def save(self, file_name):

        path = file_name
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()

    def load(self, file_name):

        path = file_name
        f = open(path, 'rb')
        a = pickle.load(f)
        self.corpus = a.corpus
        self.labels = a.labels
        f.close()

    def export_to_files(self, dir_path):

        corpus = [' '.join(d) for d in self.corpus]
        file_name = [i.split('/')[-1] for i in self.path]
        for idx, item in enumerate(corpus):
            with open(dir_path+file_name[idx]+".txt", 'w') as file_handler:
                file_handler.write("{}\n".format(item))


def main():

    e = Extractor()
    e.load(sys.argv[1])
    p = Preprocessor(e.classes, type='class', pkg_start=sys.argv[2])
    save_path = 'dataset/cache/'+sys.argv[1].split('/')[-1].split('.')[0] + \
                '_prep.pckl'
    p.save(save_path)
    for label in p.labels:
        print(label)
    print('Number of classes:  {}'.format(len(p.labels)))
    print('Number of packages: {}'.format(len(set(p.labels))))

if __name__ == '__main__':
    main()
