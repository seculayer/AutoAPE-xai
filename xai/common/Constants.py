# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from xai.common.Singleton import Singleton
from xai.common.utils.ConfigUtils import ConfigUtils
from xai.common.utils.FileUtils import FileUtils

import os
os.chdir(FileUtils.get_realpath(__file__) + "/../../")


# class : Constants
class Constants(metaclass=Singleton):
    # load config xml file
    _CONFIG = ConfigUtils.load(filename=os.getcwd() + "/conf/xai-conf.xml")

    # Directories
    DIR_DATA_ROOT = _CONFIG.get("dir_data_root", "/eyeCloudAI/data")
    DIR_DIVISION_PATH = DIR_DATA_ROOT + "/processing/ape/division"
    DIR_JOB_PATH = DIR_DATA_ROOT + "/processing/ape/jobs"
    DIR_STORAGE = DIR_DATA_ROOT + _CONFIG.get("dir_storage", "/storage/ape")
    DIR_ML_TMP = DIR_DATA_ROOT + "/processing/ape/temp"
    CUSTOM_PACK_NM = _CONFIG.get("user_custom_converter_package_nm", "cnvrtr")
    DIR_RESOURCES = (
        FileUtils.get_realpath(file=__file__)
        + "/.."
        + _CONFIG.get("dir_resources", "/resources")
    )
    DIR_RESOURCES_CNVRTR = DIR_RESOURCES + "/cnvrtr"

    # Logs
    DIR_LOG = _CONFIG.get("dir_log", "./logs")
    LOG_LEVEL = _CONFIG.get("log_level", "INFO")
    LOG_NAME = _CONFIG.get("log_name", "XAI")

    # Hosts
    MRMS_SVC = _CONFIG.get("mrms_svc", "mrms-svc")
    MRMS_USER = _CONFIG.get("mrms_username", "HE12RmzKHQtH3bL7tTRqCg==")
    MRMS_PASSWD = _CONFIG.get("mrms_password", "jTf6XrqcYX1SAhv9JUPq+w==")
    MRMS_SFTP_PORT = int(_CONFIG.get("mrms_sftp_port", "10022"))
    MRMS_REST_PORT = int(_CONFIG.get("mrms_rest_port", "9200"))

    REST_URL_ROOT = "http://{}:{}".format(
        MRMS_SVC, MRMS_REST_PORT
    )

    LIB_TYPE_TF = "TF"
    LIB_TYPE_SKL = "SKL"

    DATASET_FORMAT_TEXT = "1"
    DATASET_FORMAT_IMAGE = "2"


if __name__ == '__main__':
    print(Constants.DIR_DATA_ROOT)

