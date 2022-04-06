#  -*- coding: utf-8 -*-
#  Author : Manki Baek
#  e-mail : manki.baek@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.

from typing import Dict, List
import numpy as np

from xai.common.Common import Common
from xai.common.Constants import Constants
from xai.common.info.JobInfo import JobInfo
from xai.core.data.cnvrtr.ConvertAbstract import ConvertAbstract
from xai.core.data.dataloader.DataLoaderAbstract import DataLoaderAbstract


class AlgAbstract(object):
    LOGGER = Common.LOGGER.get_logger()

    def __init__(self, model, job_info: JobInfo):
        self.model = model
        self.job_info = job_info
        self.functions: List[List[ConvertAbstract]] = DataLoaderAbstract.build_functions(
            self.job_info.get_dataset_info().get_fields()
        )
        self.fields = self.job_info.get_dataset_info().get_fields()

    def run(self, data: Dict, json_data: Dict):
        raise NotImplementedError

    def model_inference(self, x: list):
        start = 0
        results = None
        len_x = len(x)

        while start < len_x:
            end = start + Constants.BATCH_SIZE
            batch_x = x[start: end]
            try:
                # case tensorflow
                tmp_results = self.model.predict(batch_x).numpy()
            except:
                # case sklearn
                tmp_results = self.model.predict(batch_x)
            if start == 0:
                results = tmp_results
            else:
                results = np.concatenate((results, tmp_results), axis=0)

            start += Constants.BATCH_SIZE

        return results
