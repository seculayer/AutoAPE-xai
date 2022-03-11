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

    # STATUS


if __name__ == '__main__':
    print(Constants.DIR_DATA_ROOT)

