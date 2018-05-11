"""
Class facilitating the extraction of classes from a Java project or library (located in a folder). For every class in
the project, a ClassObj object is created, containing its methods, variables, docstring, comments and various other
data in an organized and easy-to-access manner.
"""

from __future__ import absolute_import

import json
import logging
import pickle
import os

import six

from lib.astextractor.astextractor import ASTExtractor
from extractor.class_obj import ClassObj

ASTE_JAR = "lib/astextractor/ASTExtractor-0.4.jar"
ASTE_PROPERTIES = "lib/astextractor/ASTExtractor.properties"
logger = logging.getLogger('Extractor')


class Extractor(object):

    """
    Extract a ClassObj objects from a file, or a folder containing class files in java.
    """

    def __init__(self):

        """
        Initialize the variables of the object.
        """

        self.classes = []

    def extract_classes_from_project(self, folder_name):

        """
        Extract ClassObj objects for a project contained in a folder, and consisting of Java class files.

        :param folder_name: Absolute path to folder, where a Java project is contained
        :return None
        """
        # Trying to use the parse_folder method, as it is much faster. If parse_folder returns an empty ast due to
        # lack of memory (because dataset is too large), iterate over each file and use parse_file instead.

        logger.info('Using ASTExtractor to produce the JSON')
        ast_extractor = ASTExtractor(ASTE_JAR, ASTE_PROPERTIES)
        ast = ast_extractor.parse_folder(folder_name, "JSON")

        if ast == '':
            logger.warning('ASTExtractor does not have enough memory to create a JSON for the whole project. Using' +
                           ' ASTExtractor per java file.')
            for root, dirs, files in os.walk(folder_name):
                for index, file in enumerate(files):

                    file_path = os.path.join(root, file)
                    ast = ast_extractor.parse_file(file_path, "JSON")
                    try:
                        json_ast = json.loads(ast)
                    except ValueError:
                        logger.error('ValueError in generated JSON for file: {}'.format(file_path))
                        continue
                    self.extract_class_from_json(json_ast, file_path=file_path)
        else:
            logger.info('JSON produced successfully')
            json_ast = json.loads(ast)

            logger.info('Extracting classes from JSON')
            files = json_ast['folder']['file']
            for file in files:
                self.extract_class_from_json(file['ast'], file_path=file['path'])

    def extract_class_from_json(self, json_str, file_path=None, omit_embedded_classes=True):

        """
        Extract the ClassObj for a class contained in a JSON string.

        :param json_str: A JSON string to be parsed
        :param file_path: The path to the file, to which the JSON string refers
        :param omit_embedded_classes: Discard embedded classes (Java feature not yet examined)
        :return:
        """

        compilation_unit = json_str['CompilationUnit']
        if not isinstance(compilation_unit, dict):
            logger.warning('CompilationUnit is not a dictionary in File: {}'.format(file_path))
            return

        if 'PackageDeclaration' in compilation_unit.keys():
            package_dec = compilation_unit['PackageDeclaration']
        else:
            # if the class has no packageDeclaration return without adding the ClassObj to the list
            logger.warning('WARNING 1 PackageDeclaration not in File: {}'.format(file_path))
            return

        if 'TypeDeclaration' in compilation_unit.keys():
            type_dec = compilation_unit['TypeDeclaration']
        elif 'AnnotationTypeDeclaration' in compilation_unit.keys():
            logger.warning('WARNING 2 AnnotationTypeDeclaration is in file: {}'.format(file_path))
            type_dec = compilation_unit['AnnotationTypeDeclaration']
        else:
            logger.warning('WARNING 3 No TypeDeclaration in File: {}'.format(file_path))
            return

        # Can't handle list type declarations yet!
        if isinstance(type_dec, dict):

            # Create a ClassObj with the class name and add it to the list
            cl = ClassObj(type_dec['SimpleName'])
            cl.set_path(file_path)

            # Add the package information to the ClassObj first argument of the arrays is always the word package
            cl.set_package(str(package_dec).split()[1].strip(';'))

            # Append the import declaration to the ClassObj
            if 'ImportDeclaration' in compilation_unit.keys():
                import_dec = compilation_unit['ImportDeclaration']
                temp_words = []
                self.json2words(import_dec, temp_words)
                temp_words = list(filter('import'.__ne__, temp_words))
                temp_words = [w.strip(';') for w in temp_words]
                cl.add_import(temp_words)

            # Add modifier to the ClassObj
            if 'Modifier' in type_dec.keys():
                modif = []
                self.json2words(type_dec['Modifier'], modif)
                cl.add_modifier(modif)

            # Add the Javadoc to the ClassObj
            if 'Javadoc' in type_dec.keys():
                temp_words = []     # init to null,bc json2words is recursive
                self.json2words(type_dec['Javadoc'], temp_words)
                cl.add_javadoc(temp_words)

            # Add any class variables to the ClassObj
            if 'FieldDeclaration' in type_dec.keys():
                field_dec = type_dec['FieldDeclaration']
                if isinstance(field_dec, list):
                    for var in field_dec:
                        cl = self.extract_variable(var, cl)
                else:
                    cl = self.extract_variable(field_dec, cl)

            # Append the method names and method bows to the classobj
            if 'MethodDeclaration' in type_dec.keys():
                method_dec = type_dec['MethodDeclaration']
                if isinstance(method_dec, list):
                    for methods in method_dec:
                        cl = self.extract_method(methods, cl)
                else:
                    cl = self.extract_method(method_dec, cl)
        elif isinstance(type_dec, list):
            # if the class has embedded classes return without adding the ClassObj to the list
            logger.warning('WARNING 4 TypeDeclaration is List in File: {}, omit is set to {}'
                           .format(file_path, omit_embedded_classes))
            return

        self.classes.append(cl)

    def json2words(self, json_str, words):

        """
        Navigate through a JSON structure recursively and append the words contained in every leaf node to an list.

        :param json_str: the JSON structure to be navigated
        :param words: the list to which the words will be appended.

        :return words: list of words
        """

        if isinstance(json_str, dict):
            for key in json_str:
                self.json2words(json_str[key], words)
        elif isinstance(json_str, list):
            for entry in json_str:
                self.json2words(entry, words)
        elif isinstance(json_str, six.string_types):
            return words.extend(json_str.split())
        return words

    def extract_method(self, json_str, cl):

        #TODO create a tool function for the extraction of every sub-field (i.e. Javadoc, Modifier etc.)
        name = []
        if 'SimpleName' in json_str.keys():
            self.json2words(json_str['SimpleName'], name)

            if name is []:
                logger.warning('A method has no name (probably enum) in class: {}. Excluding it'.format(cl.path))
                return cl

            javadoc = []
            if 'Javadoc' in json_str.keys():
                self.json2words(json_str['Javadoc'], javadoc)

            modifier = []
            if 'Modifier' in json_str.keys():
                self.json2words(json_str['Modifier'], modifier)

            block = []
            if 'Block' in json_str.keys():
                self.json2words(json_str['Block'], block)
                block = [word for word in block if word is not None]

            cl.add_method(name, javadoc, modifier, block)

        return cl

    def extract_variable(self, json_str, cl):

        name = []
        if 'VariableDeclarationFragment' in json_str.keys():
            self.json2words(json_str['VariableDeclarationFragment'], name)

        javadoc = []
        if 'Javadoc' in json_str.keys():
            self.json2words(json_str['Javadoc'], javadoc)

        modifier = []
        if 'Modifier' in json_str.keys():
            self.json2words(json_str['Modifier'], modifier)

        variable_type = []
        if 'SimpleType' in json_str.keys():
            self.json2words(json_str['SimpleType'], variable_type)

        cl.add_variable(name, javadoc, modifier, variable_type)

        return cl

    def save(self, file_name):

        path = file_name
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()

    def load(self, file_name):

        path = file_name
        f = open(path, 'rb')
        a = pickle.load(f)
        self.classes = a.classes
        f.close()


def main():

    import sys
    from extractor.tools import clean_dataset

    a = Extractor()
    clean_dataset(sys.argv[1])
    a.extract_classes_from_project(sys.argv[1])
    for cla in a.classes:
        print(cla)
    a.save('test.pckl')


if __name__ == '__main__':
    main()
