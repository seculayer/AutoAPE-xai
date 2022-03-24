# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from multiprocessing import Queue
from typing import List

from xai.common.Singleton import Singleton
from xai.common.info.JobInfo import JobInfo
from xai.common.info.FieldInfo import FieldInfo

from xai.common.Common import Common
from xai.common.decorator.CalTimeDecorator import CalTimeDecorator
from xai.common.info.DatasetInfo import DatasetInfo
from xai.core.SFTPClientManager import SFTPClientManager
from xai.core.data.DataLoaderFactory import DataloaderFactory


class DataManager(object, metaclass=Singleton):

    def __init__(self, job_info: JobInfo, sftp_client: SFTPClientManager) -> None:
        # threading.Thread.__init__(self)
        self.LOGGER = Common.LOGGER.get_logger()
        self.job_info: JobInfo = job_info
        self.data_queue: Queue = Queue()
        self.sftp_client = sftp_client

        self.dataset_info: DatasetInfo = self.job_info.get_dataset_info()
        self.dataset = {}

    @CalTimeDecorator("Data Manager")
    def run(self) -> None:
        try:
            self.LOGGER.info("DataManager Start.")

            # ---- data load
            data_list = self.read_files(self.dataset_info.get_fields())
            self.dataset = data_list

            self.LOGGER.info("DataManager End.")
        except Exception as e:
            self.LOGGER.error(e, exc_info=True)

            raise e

    def read_files(self, fields: List[FieldInfo]) \
            -> List:
        # ---- prepare
        # 분산이 되면 워커마다 파일 1개씩, 아니면 워커1개가 모든 파일을 읽는다
        file_list = list()
        if self.job_info.get_dist_yn() \
                and (len(self.job_info.get_file_list()) == self.job_info.get_num_worker()):
            idx = int(self.job_info.get_task_idx())
            file_list.append(self.job_info.get_file_list()[idx])
        else:
            file_list = self.job_info.get_file_list()

        # data_list = self.read_subproc(file_list, fields)
        data_list = DataloaderFactory.create(
                        dataset_format=self.job_info.get_dataset_format(),
                        job_info=self.job_info,
                        sftp_client=self.sftp_client
                    ).read(file_list, fields)

        return data_list

    def get_learn_data(self) -> dict:
        return {"x": self.dataset[0][0], "y": self.dataset[0][1]}

    def get_eval_data(self) -> dict:
        return {"x": self.dataset[1][0], "y": self.dataset[1][1]}

    def get_json_data(self) -> list:
        return self.dataset[2]

    def get_inference_data(self) -> dict:
        return {"x": self.dataset[0], "y": self.dataset[1]}


# ---- builder Pattern
class DataManagerBuilder(object):
    def __init__(self):
        self.job_info = None
        self.sftp_client = None

    def set_job_info(self, job_info):
        self.job_info = job_info
        return self

    def set_sftp_client(self, sftp_client):
        self.sftp_client = sftp_client
        return self

    def build(self) -> DataManager:
        return DataManager(job_info=self.job_info, sftp_client=self.sftp_client)
