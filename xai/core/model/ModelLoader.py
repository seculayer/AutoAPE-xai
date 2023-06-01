from typing import Callable
import joblib
import tensorflow as tf
from xgboost import XGBClassifier
import pickle

from xai.common.Common import Common
from xai.common.Constants import Constants
from pycmmn.sftp.SFTPClientManager import SFTPClientManager
from pycmmn.utils.FileUtils import FileUtils


class ModelLoader(object):
    LOGGER = Common.LOGGER.getLogger()
    MRMS_SFTP_MANAGER: SFTPClientManager = SFTPClientManager(
        "{}:{}".format(Constants.MRMS_SVC, Constants.MRMS_SFTP_PORT),
        Constants.MRMS_USER, Constants.MRMS_PASSWD, LOGGER
    )

    @classmethod
    def load(cls, lib_type, model_id):
        case_fn: Callable = {
            Constants.LIB_TYPE_TF: ModelLoader._get_tf_model,
            Constants.LIB_TYPE_SKL: ModelLoader._get_skl_model,
            Constants.LIB_TYPE_XGB: ModelLoader._get_xgb_model,
            Constants.LIB_TYPE_LGBM: ModelLoader._get_lgbm_model
        }.get(lib_type)

        ModelLoader._scp_model_from_storage(model_id)
        dir_model = '{}/{}/0'.format(
            Constants.DIR_TEMP, model_id
        )
        if FileUtils.is_exist(dir_model):
            try:
                cls.LOGGER.info("model load ....")
                cls.LOGGER.info("model dir : {}".format(dir_model))

                return case_fn(dir_model)
            except Exception as e:
                cls.LOGGER.error(e, exc_info=True)
        else:
            cls.LOGGER.warn("MODEL FILE IS NOT EXIST : [{}]".format(dir_model))

    @classmethod
    def _get_xgb_model(cls, dir_model):
        model = XGBClassifier()
        try:
            model.load_model(dir_model + "/model.h5")
            return model
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e

    @classmethod
    def _get_lgbm_model(cls, dir_model):
        try:
            f = open(dir_model + "/apeflow.model", "rb")
            return pickle.load(f)
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e

    @classmethod
    def _get_tf_model(cls, dir_model):
        try:
            return tf.keras.models.load_model(dir_model)
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e

    @classmethod
    def _get_skl_model(cls, dir_model):
        try:
            return joblib.load("{}/skl_model.joblib".format(dir_model))
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e

    @classmethod
    def _scp_model_from_storage(cls, model_id) -> None:
        remote_path = f"{Constants.DIR_STORAGE}/{model_id}"
        try:
            cls.MRMS_SFTP_MANAGER.scp_from_storage(
                remote_path, Constants.DIR_TEMP
            )
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
