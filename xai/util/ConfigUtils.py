# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2017 AI-TF Team
from typing import Dict

from dataanalyzer.util.XMLUtils import XMLUtils


######################################################################################
# class : ConfUtils
# XML Configuration file handling
class ConfigUtils(object):
    @staticmethod
    def load(filename) -> Dict:
        configuration = XMLUtils.xml_load(filename=filename)

        conf_dict = dict()

        for _property in configuration.findall("property"):
            conf_dict[_property.find("name").text] = _property.find("value").text

        return conf_dict
