# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from multiprocessing import Queue
from typing import List

from pycmmn.Singleton import Singleton
from xai.info.XAIJobInfo import XAIJobInfo
from xai.info.FieldInfo import FieldInfo

from xai.common.Common import Common
from pycmmn.decorator.CalTimeDecorator import CalTimeDecorator
from xai.info import DatasetInfo
from pycmmn.sftp.SFTPClientManager import SFTPClientManager
from xai.core.data.DataLoaderFactory import DataloaderFactory


class DataManager(object, metaclass=Singleton):
    LOGGER = Common.LOGGER.getLogger()

    def __init__(self, job_info: XAIJobInfo, sftp_client: SFTPClientManager) -> None:
        # threading.Thread.__init__(self)
        self.job_info: XAIJobInfo = job_info
        self.data_queue: Queue = Queue()
        self.sftp_client = sftp_client

        self.dataset_info: DatasetInfo = self.job_info.get_dataset_info()
        self.dataset = {}

    @CalTimeDecorator("Data Manager", LOGGER)
    def run(self) -> None:
        try:
            self.LOGGER.info("DataManager Start.")

            # ---- data load
            self.dataset = self.read_files(self.dataset_info.get_fields())

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
