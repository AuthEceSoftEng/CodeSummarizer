''' ... '''

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

    '''Extract a ClassObj objects from a file, or a folder
    containing class files in java.
    '''
    def __init__(self):

        '''Initialize the arrays of the object.'''

        self.classes = []

    def clean_dataset(self, folder_name):

        for root, dirs, files in os.walk(folder_name):
            for index, file in enumerate(files):
                if not file.endswith(".java"):
                    print(os.path.join(root, file))
                    os.remove(os.path.join(root, file))
                elif os.path.getsize(os.path.join(root, file)) == 0:
                    print(os.path.join(root, file), '...EMPTY FILE')
                    os.remove(os.path.join(root, file))

        print(sum([len(files) for r, d, files in os.walk(folder_name)]))

    def extr_folder_classes(self, folder_name):

        '''Extract the class_objs for a project contained in a folder,
        and consisting of java class files.
        '''
        # Trying to use the parse_folder method.
        # if parse_folder returns empty ast due to lack of memory
        # use parse_file.

        logger.info('Using ASTExtractor to produce the JSON')
        ast_extractor = ASTExtractor(ASTE_JAR, ASTE_PROPERTIES)
        ast = ast_extractor.parse_folder(folder_name, "JSON")

        if ast == '':
            logger.warning('ASTExtractor does not have enough memory to' +
                           ' create a JSON for the whole project. Using' +
                           ' ASTExtractor per java file.')
            for root, dirs, files in os.walk(folder_name):
                for index, file in enumerate(files):
                    if not file.endswith(".java"):
                        print(os.path.join(root, file))
                        os.remove(os.path.join(root, file))
                    elif os.path.getsize(os.path.join(root, file)) == 0:
                        print(os.path.join(root, file), '...EMPTY FILE')
                        os.remove(os.path.join(root, file))
                    else:
                        file_path = os.path.join(root, file)
                        ast = ast_extractor.parse_file(file_path, "JSON")
                        try:
                            json_ast = json.loads(ast)
                        except ValueError:
                            print(file_path)
                            continue
                        self.extr_class(json_ast,
                                        file_path=file_path)

        else:
            logger.info('JSON produced successfully')
            json_ast = json.loads(ast)

            logger.info('Processing files in json')
            files = json_ast['folder']['file']
            for file in files:
                self.extr_class(file['ast'], file_path=file['path'])

    def extr_class(self, json_str, file_path=None,
                   omit_embedded_classes=True):

        '''Extract the class_obj for a class contained in a json.'''

        # set useful json location variables
        compilation_unit = json_str['CompilationUnit']
        if not isinstance(compilation_unit, dict):
            logger.warning('CompilationUnit is not a dictionary in File: {}'
                           .format(file_path))
            return

        if 'PackageDeclaration' in compilation_unit.keys():
            package_dec = compilation_unit['PackageDeclaration']
        else:
            # if the class has no packageDeclaration return without
            # adding the ClassObj to the list
            logger.warning('WARNING 1 PackageDeclaration not in File: {}'
                           .format(file_path))
            return

        if 'TypeDeclaration' in compilation_unit.keys():
            type_dec = compilation_unit['TypeDeclaration']
        elif 'AnnotationTypeDeclaration' in compilation_unit.keys():
            logger.warning('WARNING 2 AnnotationTypeDeclaration is in file: {}'
                           .format(file_path))
            type_dec = compilation_unit['AnnotationTypeDeclaration']
        else:
            logger.warning('WARNING 3 No TypeDeclaration in File: {}'
                           .format(file_path))
            return

        # Can't handle list type declarations yet!
        if isinstance(type_dec, dict):

            # Create a ClassObj with the class name and add it to the list
            cl = ClassObj(type_dec['SimpleName'])
            cl.set_path(file_path)

            # Add the package information to the ClassObj
            # first argument of the arrays is always the word package
            cl.set_package(str(package_dec).split()[1].strip(';'))

            # Append the import declaration to the ClassObj
            if 'ImportDeclaration' in compilation_unit.keys():
                import_dec = compilation_unit['ImportDeclaration']
                temp_words = []
                self.json2words(import_dec, temp_words)
                temp_words = list(filter(('import').__ne__, temp_words))
                temp_words = [w.strip(';') for w in temp_words]
                cl.add_import(temp_words)

            # Add modifier to the ClassObj
            if 'Modifier' in type_dec.keys():
                modif = []
                self.json2words(type_dec['Modifier'], modif)
                cl.add_modifier(modif)

            # Add the javadoc to the ClassObj
            if 'Javadoc' in type_dec.keys():
                temp_words = []     # init to null,bc json2words is recursive
                self.json2words(type_dec['Javadoc'], temp_words)
                cl.add_javadoc(temp_words)

            # Add any class variables to the ClassObj
            if 'FieldDeclaration' in type_dec.keys():
                field_dec = type_dec['FieldDeclaration']
                if isinstance(field_dec, list):
                    for var in field_dec:
                        cl = self.extr_var(var, cl)
                else:
                    cl = self.extr_var(field_dec, cl)

            # Append the method names and method bows to the classobj
            if 'MethodDeclaration' in type_dec.keys():
                method_dec = type_dec['MethodDeclaration']
                if isinstance(method_dec, list):
                    for methods in method_dec:
                        cl = self.extr_method(methods, cl)
                else:
                    cl = self.extr_method(method_dec, cl)

        elif isinstance(type_dec, list):
            # if the class has embedded classes return without
            # adding the ClassObj to the list
            logger.warning('WARNING 4 TypeDeclaration is List in File: ' +
                           '{}, omit is set to {}'
                           .format(file_path,
                                   omit_embedded_classes))
            return

        self.classes.append(cl)

    def json2words(self, json_str, words):

        '''Navigate through a JSON structure and append the
        words contained in every leaf node to an array.

        Parameters
        ----------
        json_str : the JSON structure to be navigated
        words : the list to which the words will be appended.

        Returns
        -------
        words : list of words
        '''

        if isinstance(json_str, dict):
            for key in json_str:
                self.json2words(json_str[key], words)
        elif isinstance(json_str, list):
            for entry in json_str:
                self.json2words(entry, words)
        elif isinstance(json_str, six.string_types):
            return words.extend(json_str.split())
        return words

    def extr_block(self, json_str, names):

        if isinstance(json_str, dict):
            for key in json_str:
                if key == 'SimpleName' or key == 'ParameterizedType' or \
                        key == 'VariableDeclarationFragment':
                    names.extend(self.extract_list(json_str[key]))
                self.extr_block(json_str[key], names)
        elif isinstance(json_str, list):
            for entry in json_str:
                self.extr_block(entry, names)
        return names

    def extr_method(self, json_str, cl):

        if 'SimpleName' in json_str.keys():
            name = self.extract_list(json_str['SimpleName'])

            if name == []:
                logger.warning('A method has no name (probably enum)' +
                               ' in class: {}. Excluding it'
                               .format(cl.path))
                return cl
            jdoc = []
            if 'Javadoc' in json_str.keys():
                self.json2words(json_str['Javadoc'], jdoc)

            modif = []
            if 'Modifier' in json_str.keys():
                modif = self.extract_list(json_str['Modifier'])

            block = []
            if 'Block' in json_str.keys():
                self.extr_block(json_str['Block'], block)
                block = [word for word in block if word is not None]

            cl.add_method(name, jdoc, modif, block)

        return cl

    def extr_var(self, json_str, cl):

        if 'VariableDeclarationFragment' in json_str.keys():
            name = self.extract_list(json_str['VariableDeclarationFragment'])

        jdoc = []
        if 'Javadoc' in json_str.keys():
            self.json2words(json_str['Javadoc'], jdoc)

        modif = []
        if 'Modifier' in json_str.keys():
            modif = self.extract_list(json_str['Modifier'])

        type = []
        if 'SimpleType' in json_str.keys():
            self.json2words(json_str['SimpleType'], type)

        cl.add_var(name, jdoc, modif, type)

        return cl

    def extract_list(self, json_str):

        if isinstance(json_str, list):
            return [word for word in json_str if not isinstance(word, bool)]
        elif isinstance(json_str, six.string_types):
            return json_str.split()
        elif json_str is None:
            return []
        else:
            return []

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

    a = Extractor()
    a.clean_dataset(sys.argv[1])
    a.extr_folder_classes(sys.argv[1])
    for cla in a.classes:
        print(cla.package)
    a.save('dataset/cache/' + sys.argv[1].split('/')[-1] + '.pckl')


if __name__ == '__main__':
    main()
