from __future__ import print_function
from __future__ import unicode_literals


class ClassObj(object):

    def __init__(self, name):

        self.package = " "
        self.path = " "
        self.import_dec = []
        self.name = name
        self.javadoc = []
        self.modif = []

        self.var_names = []
        self.var_modif = []
        self.var_javadoc = []
        self.var_type = []

        self.method_names = []
        self.method_modif = []
        self.method_javadoc = []
        self.method_block = []

    def __str__(self):
        print('Package: ', self.package)
        print('Path:    ', self.path)
        print('Imports: ', self.import_dec)
        print('Modifier:', self.modif)
        print('Name:    ', self.name)
        print()
        print('Javadoc: ', ' '.join(self.javadoc))
        print()

        for index, var_j in enumerate(self.var_javadoc):
            print('Var Modifier:    ', self.var_modif[index])
            print('Var Type:        ', self.var_type[index])
            print('Var Name:        ', self.var_names[index])
            print('Var Javadoc:     ', ' '.join(var_j))
            print()

        for index, method_j in enumerate(self.method_javadoc):
            print('Metho Modifier: ', self.method_modif[index])
            print('Method Name:    ', self.method_names[index])
            print('Method Javadoc: ', ' '.join(method_j))
            print('Method Block:   ', self.method_block[index])
            print()
        return ' '

    def set_package(self, pack):
        self.package = pack

    def set_path(self, path):
        self.path = path

    def add_modifier(self, modif):
        self.modif.extend(modif)

    def add_import(self, imp):
        self.import_dec.extend(imp)

    def add_javadoc(self, jdoc):
        self.javadoc.extend(jdoc)

    def add_method(self, name, jdoc, modif, block):
        self.method_names.append(name)
        self.method_modif.extend(modif)
        self.method_javadoc.append(jdoc)
        self.method_block.append(block)

    def add_var(self, name, jdoc, modif, type):

        self.var_names.extend(name)
        self.var_modif.append(modif)
        self.var_javadoc.append(jdoc)
        self.var_type.append(type)

    def get_package(self):
        return self.package

    def get_import_dec(self):
        return self.import_dec

    def get_name(self):
        return self.name

    def get_javadoc(self):
        return self.javadoc

    def get_method_names(self):
        return self.method_names

    def get_method_javadoc(self):
        return self.method_javadoc

    def get_method_modif(self):
        return self.method_modif

    def get_method_block(self):
        return self.method_block

    def get_var_names(self):
        return self.var_names

    def get_var_javadoc(self):
        return self.var_javadoc

    def get_var_modif(self):
        return self.var_modif

    def get_var_type(self):
        return self.var_type
