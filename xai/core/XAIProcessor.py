#  -*- coding: utf-8 -*-
#  Author : Manki Baek
#  e-mail : manki.baek@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.

from typing import Union
from datetime import datetime
import os
import tensorflow as tf
from typing import List, Dict

from pycmmn.sftp.SFTPClientManager import SFTPClientManager
from pycmmn.rest.RestManager import RestManager
from xai.common.Common import Common
from xai.common.Constants import Constants
from xai.info.XAIJobInfo import XAIJobInfo, XAIJobInfoBuilder
from xai.core.data.DataManager import DataManager, DataManagerBuilder
from xai.core.model.ModelLoader import ModelLoader
from xai.core.algorithm.Lime import Lime
from xai.core.data.datawriter.ResultWriter import ResultWriter


class XAIProcessor(object):
    LOGGER = Common.LOGGER.getLogger()

    def __init__(self, hist_no: str, task_idx: str, job_type: str) -> None:
        self.mrms_sftp_manager: SFTPClientManager = SFTPClientManager(
            "{}:{}".format(Constants.MRMS_SVC, Constants.MRMS_SFTP_PORT),
            Constants.MRMS_USER, Constants.MRMS_PASSWD, self.LOGGER
        )

        self.job_info: XAIJobInfo = XAIJobInfoBuilder() \
            .set_hist_no(hist_no=hist_no) \
            .set_task_idx(task_idx) \
            .set_job_dir(Constants.DIR_JOB) \
            .set_job_type(job_type=job_type) \
            .set_logger(self.LOGGER) \
            .set_sftp_client(self.mrms_sftp_manager) \
            .build()

        self.job_key: str = self.job_info.get_key()
        self.job_type: str = job_type
        self.task_idx: str = task_idx

        self.lib_type = self.job_info.get_lib_type()
        self._set_backend(task_idx)

        self.model = None
        self.data_loader_manager: DataManager = DataManagerBuilder() \
            .set_job_info(job_info=self.job_info) \
            .set_sftp_client(self.mrms_sftp_manager) \
            .build()
        self.xai_cls: Union[Lime, None] = None

    def run(self) -> None:
        try:
            self.LOGGER.info(f"-- XAI start. [{self.job_key}]")

            self.data_loader_manager.run()
            self.model = self.model_load()
            self.xai_cls = self.set_xai_cls()
            data = self.data_loader_manager.get_inference_data()
            json_data = self.data_loader_manager.get_json_data()
            self.LOGGER.info(f"Data Length : [{len(data['x'])}]")
            result: list = self.xai_cls.run(data, json_data)
            self.result_write(result)

            RestManager.update_xai_status_cd(
                Constants.REST_URL_ROOT,
                self.LOGGER,
                Constants.STATUS_XAI_COMPLETE,
                self.job_info.get_hist_no(),
                "0", "-"
            )
            self.LOGGER.info("-- XAI end. [{}]".format(self.job_key))

        except Exception as e:
            self.LOGGER.error(e, exc_info=True)
            RestManager.update_xai_status_cd(
                Constants.REST_URL_ROOT,
                self.LOGGER,
                Constants.STATUS_XAI_ERROR,
                self.job_info.get_hist_no(),
                "0", "-"
            )

    def set_xai_cls(self):
        return {
            # Constants.XAI_ALG_GRAD_SCAM: GradScam(self.model, self.job_info),
            Constants.XAI_ALG_LIME: Lime(self.model, self.job_info)
        }.get(self.job_info.get_xai_alg(), Constants.XAI_ALG_GRAD_SCAM)

    def model_load(self):
        return ModelLoader.load(
            lib_type=self.lib_type, model_id=self.job_info.get_model_id()
        )

    def result_write(self, result_list):
        json_data = self.data_loader_manager.get_json_data()
        json_data = self._insert_xai_info(json_data, result_list)

        ResultWriter.result_file_write(
            result_path=Constants.DIR_RESULT,
            results=json_data,
            result_type=Constants.JOB_TYPE
        )

    def _insert_xai_info(self, json_data, result_dict_list: List[Dict]):
        curr_time = datetime.now().strftime('%Y%m%d%H%M%S')

        for line_idx, jsonline in enumerate(json_data):
            result_dict_keys = result_dict_list[line_idx].keys()
            for key in result_dict_keys:
                jsonline[key] = result_dict_list[line_idx][key]
            jsonline["eqp_dt"] = curr_time
            jsonline["xai_hist_no"] = self.job_key
            jsonline["infr_hist_no"] = self.job_info.get_infr_hist_no()
            json_data[line_idx] = jsonline

        return json_data

    def _set_backend(self, task_idx):
        if self.lib_type == Constants.LIB_TYPE_TF:
            self.LOGGER.info(f"TF_CONFIG : {os.environ['TF_CONFIG']}")

            if os.environ.get("CUDA_VISIBLE_DEVICES", None) is "-1":
                self.LOGGER.info("Running CPU MODE")
            else:
                physical_devices = tf.config.list_physical_devices('GPU')

                if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("NVIDIA_COM_GPU_MEM_IDX", "0")

                if len(physical_devices) != 0:
                    # allow growth GPU memory
                    tf.config.set_visible_devices(physical_devices[0], 'GPU')

                    self.LOGGER.info(
                        f"gpu_no : {os.environ['CUDA_VISIBLE_DEVICES']}, task_idx : {task_idx}, \
                        physical devices: {physical_devices}, \
                        NVIDIA_COM_GPU_MEM_IDX : {os.environ.get('NVIDIA_COM_GPU_MEM_IDX', 'no variable!')}"
                    )

                    # 메모리 제한
                    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
                    mem_limit = int(int(os.environ.get("NVIDIA_COM_GPU_MEM_POD", 1024)) * 0.35)
                    self.LOGGER.info("GPU Memory Limit Size : {}".format(mem_limit))
                    tf.config.experimental.set_memory_growth(physical_devices[0], False)
                    tf.config.set_logical_device_configuration(
                        physical_devices[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)])

                else:
                    self.LOGGER.debug("Physical Devices(GPU) are None")
