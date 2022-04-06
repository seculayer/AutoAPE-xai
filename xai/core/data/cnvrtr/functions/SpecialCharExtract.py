# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2017-2018 AI Core Team, Intelligence R&D Center.

import re
import urllib.parse as decode
import numpy as np

from xai.core.data.cnvrtr.ConvertAbstract import ConvertAbstract


class SpecialCharExtract(ConvertAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_len = int(self.arg_list[0])
        self.num_feat = self.max_len

    def apply(self, data):
        # URL Decode
        try:
            data = data.replace("\\/", "/")
            dec_data = decode.unquote(data)
        except:
            dec_data = data

        # replace
        try:
            rep_data = re.findall(r'[\W_]', dec_data)
        except:
            rep_data = dec_data

        result = list()
        for i, ch in enumerate(rep_data):
            if i >= self.max_len:
                break
            try:
                result.append(float(ord(ch)))
            except:
                result.append(255.)

        result_len = len(result)
        # padding
        if result_len < self.max_len:
            padding = [255.]*(self.max_len - result_len)
            result.extend(padding)
            return result
        else:
            return result[:self.max_len]

    def get_num_feat(self):
        return self.max_len

    def reverse(self, data):
        rst_list = list()
        for i in range(len(data)):
            if type(data[i]) == float or type(data[i]) == int:
                if data[i] == 255:
                    rst_list.append(f"{i}_PADDING")
                elif data[i] == 0:
                    rst_list.append("NULL")
                else:
                    rst_list.append(chr(int(data[i])))

        return rst_list

    def get_original_idx(self, cvt_data, original_data):
        rst_list = list()
        find_from = 0
        data = original_data.replace("\\/", "/")
        for _ in range(5):
            data = decode.unquote(data)

        for token in self.reverse(cvt_data):
            if token == "NULL":
                token = chr(0)
            s_idx = data.find(token, find_from)

            if s_idx == -1:
                if "PADDING" not in token:
                    self.LOGGER.error(f"Can't find token : [{token}], original_data : [{data}]")
                continue

            e_idx = s_idx + len(token) - 1
            rst_list.append([s_idx, e_idx])
            find_from = e_idx + 1

        return rst_list, data


if __name__ == '__main__':
    _data = "https://stackoverflow.com/questions/16566069/url-decode-utf-8-in-python白萬基"
    cvt_fn = SpecialCharExtract(stat_dict=None, arg_list=[1000])

    print(cvt_fn.apply(data=_data))
