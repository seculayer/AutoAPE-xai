# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import requests as rq
import json

from xai.common.Common import Common
from xai.common.Constants import Constants
from xai.common.Singleton import Singleton


class RestManager(object, metaclass=Singleton):

    @staticmethod
    def get(url) -> str:
        response = rq.get(url)
        Common.LOGGER.get_logger().debug("GET {}".format(url))

        return response.text

    @staticmethod
    def post(url: str, data: dict) -> rq.Response:
        response = rq.post(url, json=data)
        Common.LOGGER.get_logger().debug("POST {}".format(url))
        Common.LOGGER.get_logger().debug("POST DATA: {}".format(data))

        return response

    @staticmethod
    def get_cnvr_dict() -> dict:
        url = Constants.REST_URL_ROOT + Common.REST_URL_DICT.get("cnvr_list", [])
        return json.loads(RestManager.get(url))
