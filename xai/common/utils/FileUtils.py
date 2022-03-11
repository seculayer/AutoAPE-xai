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
