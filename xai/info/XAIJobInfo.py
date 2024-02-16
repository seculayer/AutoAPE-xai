# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import logging
from typing import Dict

from pycmmn.Singleton import Singleton
from xai.common.Constants import Constants
from pycmmn.exceptions.FileLoadError import FileLoadError
from pycmmn.exceptions.JsonParsingError import JsonParsingError
from xai.info.DatasetInfo import DatasetInfo
from pycmmn.sftp.SFTPClientManager import SFTPClientManager
from pycmmn.utils.StringUtil import StringUtil


class XAIJobInfo(object, metaclass=Singleton):
    def __init__(self, hist_no, task_idx, job_type, job_dir, logger, sftp_client):
        self.job_type: str = job_type
        self.hist_no: str = hist_no
        self.task_idx: str = task_idx
        self.job_dir: str = job_dir
        self.LOGGER = logger
        self.sftp_client: SFTPClientManager = sftp_client

        self.info_dict: dict = self._load()
        self.LOGGER.debug(self.info_dict)

        self.dataset_info: DatasetInfo = self._create_dataset(self.info_dict.get("datasets"))

    # ---- loading
    def _create_job_filename(self) -> str:
        return self.job_type + "_" + self.hist_no + ".job"

    def _load(self) -> dict:
        filename = self._create_job_filename()
        try:
            path = f"{self.job_dir}/xai/{filename}"
            job_dict: Dict = self.sftp_client.load_json_data(path)
            self.LOGGER.info(f"--------JOB INFO(dataset excluded)----------")
            for key, value in job_dict.items():
                if key == "datasets":
                    continue
                self.LOGGER.info(f"{key} : {value}")
            self.LOGGER.info(f"job load...")
        except FileNotFoundError as e:
            self.LOGGER.error(str(e), exc_info=True)
            raise FileLoadError(file_name=filename)
        except Exception as e:
            self.LOGGER.error(str(e), exc_info=True)
            raise JsonParsingError()

        return job_dict

    def _create_dataset(self, dataset_dict) -> DatasetInfo:
        dataset = DatasetInfo(dataset_dict, self.get_target_field())
        self.LOGGER.debug(str(dataset))

        return dataset

    # ---- get
    def get_hist_no(self) -> str:
        return self.hist_no

    def get_dataset_info(self) -> DatasetInfo:
        return self.dataset_info

    def get_task_idx(self) -> str:
        return self.task_idx

    def get_fields(self):
        return self.dataset_info.get_fields()

    def get_key(self) -> str:
        # key format : jobType_HistNo
        return self.info_dict.get("key", "")

    def get_param_dict_list(self) -> list:
        return [self.info_dict.get("algorithms", dict())]

    def set_input_units(self, input_units):
        for param_dict in self.get_param_dict_list():
            param_dict["params"]["input_units"] = input_units

    def set_output_units(self, output_units):
        for param_dict in self.get_param_dict_list():
            param_dict["params"]["output_units"] = output_units

    def get_num_worker(self) -> int:
        return int(self.info_dict.get("num_worker", "1"))

    def get_project_id(self) -> str:
        return self.info_dict.get("project_id")

    def get_target_field(self) -> str:
        return self.info_dict.get("target_field")

    def get_file_list(self) -> list:
        return self.info_dict.get("datasets", {}).get("metadata_json", {}).get("file_list")

    def get_dataset_lines(self) -> list:
        if self.get_dataset_format() == Constants.DATASET_FORMAT_TEXT:
            return self.info_dict.get("datasets", {}).get("metadata_json", {}).get("file_num_line")
        elif self.get_dataset_format() == Constants.DATASET_FORMAT_IMAGE:
            return self.info_dict.get("datasets", {}).get("metadata_json", {}).get("file_num")

    def get_dist_yn(self) -> bool:
        return StringUtil.get_boolean(self.info_dict.get("algorithms", {}).get("dist_yn", "").lower())

    def get_dataset_format(self) -> str:
        return self.info_dict.get("dataset_format")

    def get_xai_alg(self):
        # TODO : 알고리즘 추가시 분기점
        return Constants.XAI_ALG_LIME

    def get_lib_type(self):
        rst = {
            "1": Constants.LIB_TYPE_TF_SINGLE,
            "2": Constants.LIB_TYPE_TF,
            "4": Constants.LIB_TYPE_GS,
            "5": Constants.LIB_TYPE_SKL
        }.get(self.info_dict["algorithms"]["lib_type"])

        if rst == Constants.LIB_TYPE_TF_SINGLE:
            rst = {
                "XGBoost": Constants.LIB_TYPE_XGB,
                "LightGBM": Constants.LIB_TYPE_LGBM
            }.get(self.info_dict["algorithms"]["algorithm_code"])

        return rst

    def get_model_id(self):
        return self.info_dict.get("learn_hist_no", None)

    def get_infr_hist_no(self):
        return self.info_dict.get("infr_hist_no", None)


class XAIJobInfoBuilder(object):
    def __init__(self):
        self.job_type = None
        self.hist_no = None
        self.job_type = None
        self.task_idx = None
        self.job_dir = None
        self.logger = logging.getLogger()
        self.sftp_client = None

    def set_hist_no(self, hist_no):
        self.hist_no = hist_no
        return self

    def set_job_dir(self, job_dir):
        self.job_dir = job_dir
        return self

    def set_job_type(self, job_type):
        self.job_type = job_type
        return self

    def set_task_idx(self, task_idx):
        self.task_idx = task_idx
        return self

    def set_logger(self, logger):
        self.logger = logger
        return self

    def set_sftp_client(self, sftp_client):
        self.sftp_client = sftp_client
        return self

    def build(self) -> XAIJobInfo:
        return XAIJobInfo(
            hist_no=self.hist_no, task_idx=self.task_idx,
            job_type=self.job_type, job_dir=self.job_dir,
            logger=self.logger, sftp_client=self.sftp_client
        )
