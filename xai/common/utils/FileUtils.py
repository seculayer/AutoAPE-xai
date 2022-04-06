# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import os


# class : FileUtils
class FileUtils(object):
    @staticmethod
    def get_realpath(file=None):
        return os.path.dirname(os.path.realpath(file))

    @staticmethod
    def mkdir(dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def is_exist(dir_path):
        return os.path.exists(dir_path)

    @classmethod
    def search_package(cls, dirname, exclude_list):
        result_list = list()
        try:
            filenames = os.listdir(dirname)
            for filename in filenames:
                full_filename = os.path.join(dirname, filename)
                if os.path.isdir(full_filename):
                    result_list += cls.search_package(full_filename, exclude_list)
                else:
                    ext = os.path.splitext(full_filename)[-1]
                    if ext == '.py' or ext == ".pyc":
                        if dirname not in result_list and filename not in exclude_list:
                            result_list.append(dirname)

        except PermissionError:
            pass

    @staticmethod
    def file_pointer(filename, mode):
        return open(filename, mode, encoding='UTF-8', errors='ignore')
