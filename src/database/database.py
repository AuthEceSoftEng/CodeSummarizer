import sqlite3
from sqlite3 import Error
import logging

logger = logging.getLogger('Database')


class Database(object):
    """Class to interface with the sqlite database.

    """

    def __init__(self, db_name):
        try:
            self.conn = sqlite3.connect(db_name)
            logger.info('Database \"'+db_name+'\" Opened successfully')
        except Error as e:
            logger.error(e)

    def __del__(self):
        logger.info('Closing Database')
        self.conn.commit()


    def get_dataset_id(self, dataset):

        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM Dataset WHERE Dataset.Name=?', (dataset, ))
        result = cursor.fetchall()
        logger.info('Result array: {}'.format(result))
        if result.__len__() == 0:
            logger.info('No dataset found in DB with that name. Adding it...')
            cursor.execute('INSERT INTO Dataset(Name) VALUES (?)', (dataset, ))
            self.conn.commit()
            logger.info('DatasetID: {}'.format(cursor.lastrowid))
            return cursor.lastrowid
        elif result.__len__() == 1:
            logger.info('Dataset \"'+dataset+'\" found')
            logger.info('DatasetID: {}'.format(result[0][0]))
            return result[0][0]
        else:
            logger.error('Multiple instances of Dataset \"'+dataset+'\" found in database. Something is wrong.'
                         ' Exiting...')
            exit(1)

    def get_vectorizer_id(self, vectorizer):

        cursor = self.conn.cursor()
        if vectorizer.name == 'Count':
            cursor.execute('SELECT * FROM Count')
            result = cursor.fetchall()
            logger.info('Result array: {}'.format(result))
            if result.__len__() == 0:
                logger.info('No Vectorizer found in DB with that configuration. Adding it...')
                cursor.execute('SELECT max(VectorizerID) FROM Vectorizer')
                vectorizer_id = cursor.fetchone()[0] + 1
                cursor.execute('INSERT INTO Vectorizer(VectorizerID, VectorizerTypeID) VALUES (?, ?)',
                               (vectorizer_id, 1))
                cursor.execute('INSERT INTO Count(VectorizerID, VectorizerTypeID) VALUES (?, ?)',
                               (vectorizer_id, 1))
                self.conn.commit()
                logger.info('VectorizerID: {}:{}'.format(vectorizer_id, cursor.lastrowid))
                return vectorizer_id
            elif result.__len__() == 1:
                logger.info('Vectorizer found: {}'.format(result))
                logger.info('VectorizerID: {}'.format(result[0][0]))
                return result[0][0]
            else:
                logger.error('Multiple instances of Count Vectorizer found in database. Something is wrong. Exiting...')
                exit(1)
        elif vectorizer.name == 'Tfidf':
            cursor.execute('SELECT * FROM Tfidf')
            result = cursor.fetchall()
            logger.info('Result array: {}'.format(result))
            if result.__len__() == 0:
                logger.info('No Vectorizer found in DB with that configuration. Adding it...')
                cursor.execute('SELECT max(VectorizerID) FROM Vectorizer')
                vectorizer_id = cursor.fetchone()[0] + 1
                cursor.execute('INSERT INTO Vectorizer(VectorizerID, VectorizerTypeID) VALUES (?, ?)',
                               (vectorizer_id, 2))
                cursor.execute('INSERT INTO Tfidf(VectorizerID, VectorizerTypeID) VALUES (?, ?)',
                               (vectorizer_id, 2))
                self.conn.commit()
                logger.info('VectorizerID: {}:{}'.format(vectorizer_id, cursor.lastrowid))
                return vectorizer_id
            elif result.__len__() == 1:
                logger.info('Vectorizer found: {}'.format(result))
                logger.info('VectorizerID: {}'.format(result[0][0]))
                return result[0][0]
            else:
                logger.error('Multiple instances of Tfidf Vectorizer found in database. Something is wrong. Exiting...')
                exit(1)

    def get_algorithm_id(self, algorithm, n_clusters):

        cursor = self.conn.cursor()
        if algorithm.name == 'KMeans':
            cursor.execute('SELECT * FROM KMeans WHERE n_clusters=? and init=? and max_iter=? and n_init=? and tol=?',
                           (n_clusters, algorithm.init, algorithm.max_iter, algorithm.n_init, algorithm.tol))
            result = cursor.fetchall()
            logger.info('Result array: {}'.format(result))
            if result.__len__() == 0:
                logger.info('No Algorithm found in DB with that configuration. Adding it...')
                cursor.execute('SELECT max(AlgorithmID) FROM Algorithm')
                algorithm_id = cursor.fetchone()[0] + 1
                cursor.execute('INSERT INTO Algorithm(AlgorithmID, AlgorithmTypeID) VALUES (?, ?)',
                               (algorithm_id, 1))
                cursor.execute('INSERT INTO '
                               'KMeans(AlgorithmID, AlgorithmTypeID, n_clusters, init, max_iter, n_init, tol)'
                               'VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (algorithm_id, 1, n_clusters, algorithm.init, algorithm.max_iter, algorithm.n_init,
                                algorithm.tol))
                self.conn.commit()
                logger.info('AlgorithmID: {}:{}'.format(algorithm_id, cursor.lastrowid))
                return algorithm_id
            elif result.__len__() == 1:
                logger.info('Algorithm found: {}'.format(result))
                logger.info('AlgorithmID: {}'.format(result[0][0]))
                return result[0][0]
            else:
                logger.error('Multiple instances of KMeans Algorithm found in database. Something is wrong. Exiting...')
                exit(1)

        elif algorithm.name == 'LDA':
            cursor.execute('SELECT * FROM LDA WHERE n_clusters=? and doc_topic_prior=? and topic_word_prior=? and'
                           ' max_iter=? and learning_method=? and learning_offset=?',
                           (n_clusters, algorithm.dtp/n_clusters, algorithm.twp, algorithm.max_iter, algorithm.learning_method,
                            algorithm.learning_offset))
            result = cursor.fetchall()
            logger.info('Result array: {}'.format(result))
            if result.__len__() == 0:
                logger.info('No Algorithm found in DB with that configuration. Adding it...')
                cursor.execute('SELECT max(AlgorithmID) FROM Algorithm')
                algorithm_id = cursor.fetchone()[0] + 1
                cursor.execute('INSERT INTO Algorithm(AlgorithmID, AlgorithmTypeID) VALUES (?, ?)',
                               (algorithm_id, 2))
                cursor.execute('INSERT INTO '
                               'LDA(AlgorithmID, AlgorithmTypeID, n_clusters, doc_topic_prior, topic_word_prior, '
                               'max_iter, learning_method, learning_offset)'
                               'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                               (algorithm_id, 2, n_clusters, algorithm.dtp/n_clusters, algorithm.twp, algorithm.max_iter,
                                algorithm.learning_method, algorithm.learning_offset))
                self.conn.commit()
                logger.info('AlgorithmID: {}:{}'.format(algorithm_id, cursor.lastrowid))
                return algorithm_id
            elif result.__len__() == 1:
                logger.info('Algorithm found: {}'.format(result))
                logger.info('AlgorithmID: {}'.format(result[0][0]))
                return result[0][0]
            else:
                logger.error('Multiple instances of LDA Algorithm found in database. Something is wrong. Exiting...')
                exit(1)

    def get_results_id(self, dataset_id, vectorizer_id, algorithm_id):

        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM Run WHERE AlgorithmID=? and VectorizerID=? and DatasetID=?',
                       (algorithm_id, vectorizer_id, dataset_id))
        result = cursor.fetchall()
        logger.info('Result array: {}'.format(result))
        if result.__len__() == 0:
            logger.info('No Results found in DB for the selected configuration.')
            return None
        elif result.__len__() == 1:
            logger.info('Reults found: {}'.format(result))
            logger.info('ResultID: {}'.format(result[0][-1]))
            return result[0][0]
        else:
            logger.error('Multiple instances of Results found in database. Something is wrong. Exiting...')
            exit(1)
