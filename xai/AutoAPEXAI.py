# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2021-2022 AI Service Model Team, R&D Center.

from xai.common.Common import Common
from xai.common.Constants import Constants
from xai.core.XAIProcessor import XAIProcessor


class AutoAPEXAI(object):
    LOGGER = Common.LOGGER.get_logger()

    def __init__(self, key, task_idx):
        self.key: str = key
        self.task_idx: str = task_idx

        self.LOGGER.info(Constants.VERSION_MANAGER.print_version())
        self.processor = XAIProcessor(key, task_idx, Constants.JOB_TYPE)

    def run(self) -> None:
        self.processor.run()


if __name__ == '__main__':
    import sys
    _key = sys.argv[1]
    _task_idx = sys.argv[2]

    xai = AutoAPEXAI(_key, _task_idx)
    xai.run()
