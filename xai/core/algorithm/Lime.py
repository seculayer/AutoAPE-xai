#  -*- coding: utf-8 -*-
#  Author : Manki Baek
#  e-mail : manki.baek@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.

from lime import lime_image, lime_tabular
from lime.wrappers.scikit_image import SegmentationAlgorithm
from typing import Dict, Callable, List
import numpy as np
from datetime import datetime

from pycmmn.rest.RestManager import RestManager
from xai.common.Constants import Constants
from xai.info.XAIJobInfo import XAIJobInfo
from xai.core.algorithm.AlgAbstract import AlgAbstract


class Lime(AlgAbstract):
    def __init__(self, model, job_info: XAIJobInfo):
        super().__init__(model, job_info)
        self.data_type = self.job_info.get_dataset_format()
        if self.job_info.get_lib_type() == Constants.LIB_TYPE_TF:
            self.predict_fn = self.model.predict
        else:  # Constants.LIB_TYPE_SKL
            self.predict_fn = self.model.predict_proba

        if self.data_type == Constants.DATASET_FORMAT_IMAGE:
            self.explainer = lime_image.LimeImageExplainer()
            # 이미지 분할 알고리즘 종류 slic, quickshift, felzenszwalb
            self.segmenter = SegmentationAlgorithm(
                'slic',         # 분할 알고리즘 이름
                n_segments=20,   # 이미지 분할 조각 개수
                compactness=3,  # 유사한 파트를 합치는 함수
                sigma=1         # 스무딩 역할: 0과 1사이의 float
            )

    def run(self, data: Dict, json_data: List):
        x = data['x']

        result_list = list()
        progress_pct = 0

        total_start_time = datetime.now()
        for idx, line in enumerate(x):
            loop_start_time = datetime.now()

            try:
                case: Callable = {
                    Constants.DATASET_FORMAT_TEXT: self.text_data_run,
                    Constants.DATASET_FORMAT_IMAGE: self.image_data_run
                }.get(self.data_type, self.text_data_run)

                line_rst_dict = case(x, json_data, idx)

                result_list.append(line_rst_dict)

                self.LOGGER.info(f"Line [{idx}] is finished..")

                # 진행률 표시
                progress_pct = (idx + 1) / len(x) * 100
                self.LOGGER.info(f"{progress_pct} % completed...")

                RestManager.send_xai_progress(
                    Constants.REST_URL_ROOT, self.LOGGER, self.job_info.get_hist_no(), progress_pct
                )

            except Exception as e:
                self.LOGGER.error(e, exc_info=True)
                self.LOGGER.error(f"idx : {idx}, data : {line}")
                self.LOGGER.error("append {} at result_list")
                result_list.append({})

            self.LOGGER.info(f"Loop excution time : [{datetime.now() - loop_start_time}]")

        self.LOGGER.info(f"Total excution time : [{datetime.now() - total_start_time}]")

        RestManager.send_xai_progress(
            Constants.REST_URL_ROOT, self.LOGGER, self.job_info.get_hist_no(), 100.0, "delete"
        )

        return result_list

    def text_data_run(self, cvt_data, json_data, line_idx):

        inferenced_y = self.model_inference(cvt_data)

        column_list = list()
        original_idx_dict = dict()
        cvt_dict = dict()
        line_rst_dict = dict()
        s_idx = 0
        unique_labels = 0
        for field_idx, field in enumerate(self.fields):
            if field.is_label:
                unique_labels = field.stat_dict["unique_count"]
                continue

            max_len = self.functions[field_idx][-1].get_num_feat()
            e_idx = s_idx + max_len

            reversed_data = cvt_data[line_idx][s_idx:e_idx]
            for cvt_idx in range(len(self.functions[field_idx]) - 1, -1, -1):  # 역순
                reversed_data = self.functions[field_idx][cvt_idx].reverse(reversed_data, json_data[line_idx][field.field_name])
            tmp_idx_list, cvt_origin = self.functions[field_idx][-1].get_original_idx(
                cvt_data=cvt_data[line_idx][s_idx:e_idx], original_data=json_data[line_idx][field.field_name]
            )
            original_idx_dict[field.field_name] = tmp_idx_list
            cvt_dict[field.field_name] = cvt_origin
            column_list.extend(reversed_data)

        explainer = lime_tabular.LimeTabularExplainer(np.array(cvt_data), feature_names=column_list)
        exp = explainer.explain_instance(
            np.array(cvt_data[line_idx]), predict_fn=self.predict_fn,
            num_samples=1000, labels=range(unique_labels), num_features=6
        )
        """
            exp.domain_mapper.feature_names : feature name
            exp.local_exp : (feature idx, 점수)
            exp.domain_mapper.feature_values : feature value
            exp.domain_mapper.discretized_feature_names : 점수위에 설명???
            exp.calss_names : class name
        """
        # line_rst_dict["feature_name"] = exp.domain_mapper.feature_names
        temp_effect_val = dict()
        for key in exp.local_exp.keys():
            temp_effect_val[key] = [(exp.domain_mapper.feature_names[f_idx], val) for (f_idx, val) in exp.local_exp[key]]
        line_rst_dict["effect_val"] = temp_effect_val
        line_rst_dict["class_names"] = exp.class_names
        line_rst_dict["predict_proba"] = exp.predict_proba.tolist()
        line_rst_dict["inference_result"] = int(inferenced_y[line_idx])
        line_rst_dict["origin_idx_dict"] = original_idx_dict
        line_rst_dict["cvt_dict"] = cvt_dict

        # exp.save_to_file(f"./temp_data/rrmm_{line_idx}.html")

        return line_rst_dict

    def image_data_run(self, cvt_data, json_data, line_idx):
        inferenced_y = self.model_inference(cvt_data)

        line_rst_dict = dict()
        unique_labels = 0
        reversed_data = None
        for field_idx, field in enumerate(self.fields):
            if field.is_label:
                unique_labels = field.stat_dict["unique_count"]
                continue

            reversed_data = cvt_data[line_idx]
            for cvt_idx in range(len(self.functions[field_idx]) - 1, -1, -1):  # 역순
                reversed_data = self.functions[field_idx][cvt_idx].reverse(reversed_data)

        exp = self.explainer.explain_instance(
            np.array(reversed_data),
            classifier_fn=self.predict_fn,  # 각 class 확률 반환
            num_samples=1000,              # sample space
            top_labels=1,                   # 확률 기준 1위
            segmentation_fn=self.segmenter  # 분할 알고리즘
        )
        img, mask = exp.get_image_and_mask(exp.top_labels[0])
        self.LOGGER.info(mask)
        self.LOGGER.info(f"inference rst : [{exp.top_labels[0]}]")
        line_rst_dict["inference_result"] = int(inferenced_y[line_idx])
        line_rst_dict["mask"] = mask.tolist()

        return line_rst_dict
