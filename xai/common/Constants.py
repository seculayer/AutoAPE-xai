# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from pycmmn.Singleton import Singleton
from pycmmn.utils.ConfUtils import ConfUtils
from pycmmn.utils.FileUtils import FileUtils
from pycmmn.tools.VersionManagement import VersionManagement

import os


# class : Constants
class Constants(metaclass=Singleton):
    _working_dir = os.getcwd()
    _data_cvt_dir = _working_dir + "/../xai"
    _conf_xml_filename = _data_cvt_dir + "/conf/xai-conf.xml"

    _MODE = "deploy"

    if not FileUtils.is_exist(_conf_xml_filename):
        _MODE = "dev"

        if _working_dir != "/eyeCloudAI/app/ape/xai":
            os.chdir(FileUtils.get_realpath(__file__) + "/../../")

        _working_dir = os.getcwd()
        _data_cvt_dir = _working_dir + "/../xai"
        _conf_xml_filename = _working_dir + "/conf/xai-conf.xml"

    # load config xml file
    _CONFIG = ConfUtils.load(filename=os.getcwd() + "/conf/xai-conf.xml")

    try:
        VERSION_MANAGER = VersionManagement(app_path=_working_dir)
    except Exception as e:
        # DEFAULT
        VersionManagement.generate(
            version="1.0.0",
            app_path=_working_dir,
            module_nm="xai",
        )
        VERSION_MANAGER = VersionManagement(app_path=_working_dir)
    VERSION = VERSION_MANAGER.VERSION
    MODULE_NM = VERSION_MANAGER.MODULE_NM

    # Directories
    DIR_DATA_ROOT = _CONFIG.get("dir_data_root", "/eyeCloudAI/data")
    DIR_PROCESSING = DIR_DATA_ROOT + _CONFIG.get("dir_processing", "/processing/ape")
    DIR_JOB = DIR_PROCESSING + _CONFIG.get("dir_job", "/jobs")
    DIR_STORAGE = DIR_DATA_ROOT + _CONFIG.get("dir_storage", "/storage/ape")
    DIR_TEMP = DIR_DATA_ROOT + "/processing/ape/temp"
    DIR_RESOURCES = (
        FileUtils.get_realpath(file=__file__) + "/resources"
    )
    DIR_RESULT = DIR_PROCESSING + _CONFIG.get("dir_result", "/results_xai")
    DIR_WEB_FILE = _CONFIG.get("ape_web_file_dir", "/eyeCloudAI/app/www/store/upload")

    # Logs
    DIR_LOG = _CONFIG.get("dir_log", "./logs")
    LOG_LEVEL = _CONFIG.get("log_level", "INFO")
    LOG_NAME = _CONFIG.get("log_name", "XAI")

    # Hosts
    MRMS_SVC = _CONFIG.get("mrms_svc", "mrms-svc")
    MRMS_SFTP_PORT = int(_CONFIG.get("mrms_sftp_port", "10022"))
    MRMS_REST_PORT = int(_CONFIG.get("mrms_rest_port", "9200"))
    MRMS_USER = _CONFIG.get("mrms_username", "HE12RmzKHQtH3bL7tTRqCg==")
    MRMS_PASSWD = _CONFIG.get("mrms_password", "jTf6XrqcYX1SAhv9JUPq+w==")

    STORAGE_SVC = _CONFIG.get("storage_svc", "ape-storage-svc")
    STORAGE_SFTP_PORT = int(_CONFIG.get("storage_sftp_port", "10122"))
    STORAGE_USER = _CONFIG.get("storage_username", "HE12RmzKHQtH3bL7tTRqCg==")
    STORAGE_PASSWD = _CONFIG.get("storage_password", "jTf6XrqcYX1SAhv9JUPq+w==")

    REST_URL_ROOT = "http://{}:{}".format(
        MRMS_SVC, MRMS_REST_PORT
    )

    JOB_TYPE = "xai"

    LIB_TYPE_TF_SINGLE = "TF_SINGLE"
    LIB_TYPE_TF = "TF"
    LIB_TYPE_GS = "GS"
    LIB_TYPE_SKL = "SKL"
    LIB_TYPE_LGBM = "LGBM"
    LIB_TYPE_XGB = "XGB"

    DATASET_FORMAT_TEXT = "1"
    DATASET_FORMAT_IMAGE = "2"
    DATASET_FORMAT_TABLE = "3"

    XAI_ALG_GRAD_SCAM = "gram_scam"
    XAI_ALG_LIME = "lime"

    STATUS_XAI_COMPLETE = "6"
    STATUS_XAI_ERROR = "7"

    BATCH_SIZE = int(_CONFIG.get("inference_batch_size", "512"))

    # TABLE FIELD TYPE
    FIELD_TYPE_NULL = "null"
    FIELD_TYPE_INT = "int"
    FIELD_TYPE_FLOAT = "float"
    FIELD_TYPE_STRING = "string"
    FIELD_TYPE_IMAGE = "image"
    FIELD_TYPE_DATE = "date"
    FIELD_TYPE_LIST = "list"


if __name__ == '__main__':
    print(Constants.DIR_DATA_ROOT)
