import os
import sys
import json
import pprint
import urllib3
import certifi
from bs4 import BeautifulSoup

from googleapiclient.discovery import build
import logging

logger = logging.getLogger('OfficialGoogler')


class OfficialGoogler(object):

    def __init__(self):
        fp = open('../config/api_key.txt', 'r')
        key = fp.readline()
        if key:
            self.service = build("customsearch", "v1",
                                 developerKey=key)
        else:
            logger.error('No API key found. Please consult the Readme.')
            sys.exit(1)


    def extract_links(self, res_file):

        '''
        Extracts the links returned by a google search.
        Supports a JSON file as input.

        Parameters
        ----------
        res_file    : JSON file, created by google search.

        Return
        ------
        url_list   : list of urls.
        '''

        print(res_file)
        try:
            f = open(res_file, 'r')
            res = json.load(f)
            f.close()
        except ValueError:
            sys.exit('Invalid file. File is not JSON.', res_file)
        except IOError:
            sys.exit('Cannot open file', res_file)

        url_list = []
        if 'items' in res.keys():
            for item in res['items']:
                url_list.append(item['link'])
        return url_list

    def fetch_tags(self, res_file):

        '''
        Fetches a list of tags, contained in StackOverflow urls.

        Parameters
        ----------
        url_list : list of StackOverflow urls to fetch

        Return
        ------
        tag_list_sorted : list of tags, included in all urls and sorted
                          based on occurence.
        '''
        url_list = self.extract_links(res_file)

        tag_list = []
        for url in url_list:
            http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                                       ca_certs=certifi.where())
            r = http.request('GET', url)
            soup = BeautifulSoup(r.data, 'html.parser')
            for tag in soup.find_all('a'):
                if tag.get('class') is not None:
                    if tag.get('class')[0] == 'post-tag':
                        tag_list.append(tag.get_text())
        tag_set = set(tag_list)
        count = [tag_list.count(x) for x in tag_set]
        tag_list_sorted = sorted(zip(count, tag_set), reverse=True)

        return tag_list_sorted

    def search(self, query, res_file):

        '''
        Searches the internet using the customsearch Google API. The
        customsearch API is configured to search StackOverflow.com.

        Parameters
        ----------
        query    : String containing the query to be searched.
        res_file : Path of file to store the results.

        Return
        ------
        res : JSON structure containing  the result of the Google search.
        '''

        res = self.service.cse().list(q=query,
                                      cx='017297631459647283051' +
                                      ':rsleiavvpd0').execute()
        f = open(res_file, 'w')
        json.dump(res, f)
        f.close()
        return res

    def tags_per_cluster(self, clusterer, labels_true, search=True, show=True):

        top_terms = clusterer.top_terms_per_cluster(num=7, show=False)

        labels_true = [label.split('.')[-1] for label in labels_true]
        best = [index for index, value in enumerate(labels_true)]

        if search:
            for index in best:
                path = 'searcher/Cluster'+str(index)+'_tags.txt'
                if os.path.isfile(path):
                    continue
                else:
                    query = labels_true[index]+' '+' '.join(top_terms[index])
                    logger.info("Query used: {}".format(query))
                    self.search(query, path)
        tags = []
        for index in best:
            tags.append(self.fetch_tags('../output/Cluster' +
                                        str(index)+'_tags.txt'))
        if show:
            count = 0
            for index, clust in enumerate(top_terms):
                if index in best:
                    logger.info('Cluster {}: {}'.format(index, ' '.join(clust)))
                    logger.info('Cluster {}: {}'.format(index,
                                                        tags[count].__str__()))
                    count = count + 1
        return tags


def main():
    og = OfficialGoogler()
    og.search('plt show ax plot', 'result.json')
    tags = og.fetch_tags('result.json')
    pprint.pprint(tags)

if __name__ == '__main__':
    main()
