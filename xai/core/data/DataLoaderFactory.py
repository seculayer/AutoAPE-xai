# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team

from xai.common.Common import Common
from xai.common.Constants import Constants
from xai.core.data.dataloader.DataLoaderText import DataLoaderText
from xai.core.data.dataloader.DataLoaderImage import DataLoaderImage
from xai.core.SFTPClientManager import SFTPClientManager
from xai.common.info.JobInfo import JobInfo


class DataloaderFactory(object):
    LOGGER = Common.LOGGER.get_logger()

    @staticmethod
    def create(dataset_format: str, job_info: JobInfo, sftp_client: SFTPClientManager):
        case = {
            Constants.DATASET_FORMAT_TEXT: "DataLoaderText",
            Constants.DATASET_FORMAT_IMAGE: "DataLoaderImage"
        }.get(dataset_format)
        return eval(case)(job_info, sftp_client)


if __name__ == '__main__':
    pass
