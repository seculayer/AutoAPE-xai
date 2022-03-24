# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import json

# ---- automl packages
from xai.common.Singleton import Singleton
from xai.common.logger.MPLogger import MPLogger
from xai.common.utils.FileUtils import FileUtils
from xai.common.Constants import Constants


# class : class_name
class Common(metaclass=Singleton):
    # make directories
    FileUtils.mkdir(Constants.DIR_DATA_ROOT)
    FileUtils.mkdir(Constants.DIR_LOG)

    # LOGGER
    LOGGER: MPLogger = MPLogger(log_dir=Constants.DIR_LOG, log_level=Constants.LOG_LEVEL,
                                log_name=Constants.LOG_NAME)

    CNVRTR_PACK_LIST = [
        "xai.core.data.cnvrtr.functions",
        Constants.CUSTOM_PACK_NM
    ]

    with open(Constants.DIR_RESOURCES + "/rest_url_info.json", "r") as f:
        REST_URL_DICT = json.load(f)
