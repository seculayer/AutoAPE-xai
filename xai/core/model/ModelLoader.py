from typing import Callable
import joblib

from xai.common.Common import Common
from xai.common.Constants import Constants
from xai.core.SFTPClientManager import SFTPClientManager
from xai.common.utils.FileUtils import FileUtils


class ModelLoader(object):
    LOGGER = Common.LOGGER.get_logger()
    MRMS_SFTP_MANAGER: SFTPClientManager = SFTPClientManager(
        "{}:{}".format(Constants.MRMS_SVC, Constants.MRMS_SFTP_PORT), Constants.MRMS_USER, Constants.MRMS_PASSWD
    )

    @classmethod
    def load(cls, lib_type, alg_cls, model_id):
        case_fn: Callable = {
            Constants.LIB_TYPE_TF: ModelLoader._get_tf_model,
            Constants.LIB_TYPE_SKL: ModelLoader._get_skl_model
        }.get(lib_type)

        ModelLoader._scp_model_from_storage(model_id)
        dir_model = '{}/{}/0'.format(
            Constants.DIR_ML_TMP, model_id
        )
        if FileUtils.is_existed(dir_model):
            try:
                case_fn(alg_cls, dir_model)

                cls.LOGGER.info("model load ....")
                cls.LOGGER.info("model dir : {}".format(dir_model))
            except Exception as e:
                cls.LOGGER.error(e, exc_info=True)
        else:
            cls.LOGGER.warn("MODEL FILE IS NOT EXIST : [{}]".format(dir_model))

    @classmethod
    def _get_tf_model(cls, alg_cls, dir_model):
        try:
            alg_cls.model.load_weights(dir_model + '/weights.h5')
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e

    @classmethod
    def _get_skl_model(cls, alg_cls, dir_model):
        try:
            joblib.dump(alg_cls.model, "{}/skl_model.pkl".format(dir_model))
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e

    @classmethod
    def _scp_model_from_storage(cls, model_id) -> None:
        remote_path = f"{Constants.DIR_STORAGE}/{model_id}"
        try:
            cls.MRMS_SFTP_MANAGER.scp_from_storage(
                remote_path, Constants.DIR_ML_TMP
            )
        except Exception as e:
            pass
