"""
Tools facilitating the cleanup and preparation of the dataset for extraction.
"""
import os
import logging

logger = logging.getLogger('Extractor Tools')


def clean_dataset(folder_name):
    """
    Traverse the directory structure under folder_name and remove empty files and files not ending with .java.
    Be carefull as the files are permanently deleted and cannot be recovered.

    :param folder_name: Absolute path of the root folder, where a java project is located.
    :return: None.
    """
    for root, dirs, files in os.walk(folder_name):
        for index, file in enumerate(files):
            if not file.endswith(".java"):
                logger.debug('Removing non .java file: ', os.path.join(root, file))
                os.remove(os.path.join(root, file))
            elif os.path.getsize(os.path.join(root, file)) == 0:
                logger.debug('Removing empty file: ', os.path.join(root, file))
                os.remove(os.path.join(root, file))

    logger.debug('Total number of files remaining: ', sum([len(files) for r, d, files in os.walk(folder_name)]))