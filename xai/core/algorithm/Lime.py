#  -*- coding: utf-8 -*-
#  Author : Manki Baek
#  e-mail : manki.baek@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.

from lime import lime_image, lime_tabular, lime_text
from lime.wrappers.scikit_image import SegmentationAlgorithm
from typing import Dict, Callable, List, AnyStr, Union
import numpy as np
from sklearn.pipeline import make_pipeline
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pdpbox import pdp

from pycmmn.sftp.SFTPClientManager import SFTPClientManager
from pycmmn.rest.RestManager import RestManager
from pycmmn.utils.CV2Utils import CV2Utils
from pycmmn.utils.FileUtils import FileUtils
from xai.common.Constants import Constants
from xai.info.XAIJobInfo import XAIJobInfo
from xai.core.algorithm.AlgAbstract import AlgAbstract


class Lime(AlgAbstract):
    def __init__(self, model, job_info: XAIJobInfo, sftp_manager: SFTPClientManager):
        super().__init__(model, job_info)
        self.data_type = self.get_dataset_type()
        self.sftp_manager = sftp_manager

        self.predict_fn: Union[Callable, str] = self.define_predict_fn()

        self.explainer = None
        self.segmenter = None

    def get_dataset_type(self) -> str:
        dataset_format = self.job_info.get_dataset_format()

        # 현재, 이미지가 아닐경우 무조건 dataset format을 text로 분류
        if dataset_format == Constants.DATASET_FORMAT_TEXT:
            str_flag = False
            for field in self.fields:
                if field.is_label:
                    continue
                if field.field_type == "string":
                    str_flag = True
                    break

            if not str_flag:
                dataset_format = Constants.DATASET_FORMAT_TABLE

        return dataset_format

    def define_predict_fn(self) -> Union[Callable, str]:
        predict_fn: Union[Callable, str, None] = None
        if self.job_info.get_lib_type() == Constants.LIB_TYPE_TF:
            if self.data_type == Constants.DATASET_FORMAT_TEXT:
                predict_fn = "predict"
            else:
                predict_fn = self.model.predict
        else:  # Constants.LIB_TYPE_SKL, LIB_TYPE_LGBM, LIB_TYPE_XGB
            if self.data_type == Constants.DATASET_FORMAT_TEXT:
                predict_fn = "predict_proba"
            else:
                predict_fn = self.model.predict_proba

        return predict_fn

    def define_explainer(self, cvt_data) -> None:
        if self.data_type == Constants.DATASET_FORMAT_IMAGE:
            self.explainer = lime_image.LimeImageExplainer()
            # 이미지 분할 알고리즘 종류 slic, quickshift, felzenszwalb
            self.segmenter = SegmentationAlgorithm(
                'slic',         # 분할 알고리즘 이름
                n_segments=30,   # 이미지 분할 조각 개수
                compactness=3,  # 유사한 파트를 합치는 함수
                sigma=1         # 스무딩 역할: 0과 1사이의 float
            )
        elif self.data_type == Constants.DATASET_FORMAT_TEXT:
            self.explainer = lime_text.LimeTextExplainer(random_state=42, split_expression=r' ')
            # self.explainer = lime_text.LimeTextExplainer(random_state=42, split_expression=r'[\W_0-9]+')

        elif self.data_type == Constants.DATASET_FORMAT_TABLE:
            column_list = list()
            for field_idx, field in enumerate(self.fields):
                if field.is_label:
                    continue
                column_list.append(field.field_name)

            self.explainer = lime_tabular.LimeTabularExplainer(np.array(cvt_data), feature_names=column_list)

    def run(self, data: Dict, json_data: List):
        x = data['x']
        self.define_explainer(x)

        result_list = list()

        total_start_time = datetime.now()
        for idx, line in enumerate(x):
            loop_start_time = datetime.now()

            try:
                case: Callable = {
                    Constants.DATASET_FORMAT_TEXT: self.text_data_run,
                    Constants.DATASET_FORMAT_IMAGE: self.image_data_run,
                    Constants.DATASET_FORMAT_TABLE: self.tabular_data_run
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

        if self.data_type == Constants.DATASET_FORMAT_TABLE:
            result_list.append(self.make_table_statistics(json_data, x))

        self.scp_image_rst()

        self.LOGGER.info(f"Total excution time : [{datetime.now() - total_start_time}]")

        RestManager.send_xai_progress(
            Constants.REST_URL_ROOT, self.LOGGER, self.job_info.get_hist_no(), 100.0, "delete"
        )

        return result_list

    def make_table_statistics(self, json_data: List[Dict], cvt_data: List) -> Dict:
        df_data = pd.DataFrame(json_data)
        rst_dict = dict()

        try:
            df_data.pop('key')
            df_data.pop('dataset_id')
            df_data.pop('proc_dt')
        except KeyError:
            pass

        FileUtils.mkdir(f"{self.job_info.get_hist_no()}")

        line_plot_list = self.create_line_plot(df_data)
        rst_dict["line_plot_list"] = line_plot_list

        pdp_isolate_list = self.create_pdp_isolate(cvt_data)
        rst_dict["pdp_isolate_list"] = pdp_isolate_list

        pdp_interact_list = self.create_pdp_interact(cvt_data)
        rst_dict["pdp_interact_list"] = pdp_interact_list

        return rst_dict

    def create_line_plot(self, df_data) -> List[str]:
        features_list = [field.field_name for field in self.fields]
        target_field = features_list.pop(0)

        color_list = ["blue", "green", "orange", "red"]
        file_name_list = list()

        for idx, feature in enumerate(features_list):
            plt.figure(figsize=(16, 9))
            sns.set()
            sns.lineplot(data=df_data, x=target_field, y=feature, errorbar=('ci', 95), color=color_list[idx % len(color_list)])
            plt.subplots_adjust(top=0.93)
            plt.suptitle(f"{feature} and {target_field} plot", fontsize=16)
            file_name = f"{feature}_{target_field}_line_plot"
            file_name_list.append(f"/xai/table_statistics/{self.job_info.get_hist_no()}/{file_name}.png")
            plt.savefig(f"{self.job_info.get_hist_no()}/{file_name}", bbox_inched='tight', dpi=400)

        return file_name_list

    def create_pdp_isolate(self, cvt_data: List, cluster_flag=False, nb_clusters=None, lines_flag=False) -> List[str]:
        if not self.job_info.get_lib_type() in [Constants.LIB_TYPE_XGB, Constants.LIB_TYPE_LGBM, Constants.LIB_TYPE_SKL]:
            return []

        features_list = [field.field_name for field in self.fields]
        features_list.pop(0)
        x = pd.DataFrame(cvt_data, columns=features_list)
        file_name_list = list()

        for feature in features_list:
            # Create the data that we will plot
            pdp_goals = pdp.pdp_isolate(model=self.model, dataset=x, model_features=features_list, feature=feature)
            pdp.pdp_plot(pdp_goals, feature, cluster=cluster_flag, n_cluster_centers=nb_clusters, plot_lines=lines_flag)
            file_name = f"{feature}_pdp_isolate.png"
            plt.savefig(f"{self.job_info.get_hist_no()}/{file_name}", dpi=400, bbox_inches="tight")
            file_name_list.append(f"/xai/table_statistics/{self.job_info.get_hist_no()}/{file_name}")

        return file_name_list

    def create_pdp_interact(self, cvt_data: List) -> List[str]:
        if not self.job_info.get_lib_type() in [Constants.LIB_TYPE_XGB, Constants.LIB_TYPE_LGBM, Constants.LIB_TYPE_SKL]:
            return []

        features_list = [field.field_name for field in self.fields]
        features_list.pop(0)
        x = pd.DataFrame(cvt_data, columns=features_list)
        file_name_list = list()

        for start in range(len(features_list)):
            if start == len(features_list) - 1 or len(features_list) < 2:
                break
            for end in range(start + 1, len(features_list)):
                features_to_plot = [features_list[start], features_list[end]]
                inter = pdp.pdp_interact(model=self.model, dataset=x, model_features=features_list, features=features_to_plot)
                pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=features_to_plot, plot_type='contour')
                file_name = f"{'_'.join(features_to_plot)}_pdp_interact.png"
                plt.savefig(f"{self.job_info.get_hist_no()}/{file_name}", dpi=400, bbox_inches="tight")
                file_name_list.append(f"/xai/table_statistics/{self.job_info.get_hist_no()}/{file_name}")

        return file_name_list

    def scp_image_rst(self) -> None:
        if self.data_type == Constants.DATASET_FORMAT_IMAGE:
            self.sftp_manager.scp_to_storage(
                local_path=f"{self.job_info.get_hist_no()}",
                remote_path=f"{Constants.DIR_WEB_FILE}/xai/masked_image"
            )
            self.sftp_manager.scp_to_storage(
                local_path=f"{self.job_info.get_hist_no()}_thumbnail",
                remote_path=f"{Constants.DIR_WEB_FILE}/xai/masked_thumbnail_image"
            )
            FileUtils.remove_dir(f"{self.job_info.get_hist_no()}")
            FileUtils.remove_dir(f"{self.job_info.get_hist_no()}_thumbnail")
        elif self.data_type == Constants.DATASET_FORMAT_TABLE:
            self.sftp_manager.scp_to_storage(
                local_path=f"{self.job_info.get_hist_no()}",
                remote_path=f"{Constants.DIR_WEB_FILE}/xai/table_statistics"
            )
            FileUtils.remove_dir(f"{self.job_info.get_hist_no()}")

    def text_data_run(self, cvt_data, json_data, line_idx):
        inferenced_y = self.model_inference(cvt_data)

        pipe = make_pipeline(self.functions[-1][-1], self.model)
        line_rst_dict = dict()
        original_idx_dict = dict()
        cvt_dict = dict()

        s_idx = 0
        reversed_data = list()
        for field_idx, field in enumerate(self.fields):
            if field.is_label:
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
        exp = self.explainer.explain_instance(" ".join(reversed_data), eval(f"pipe.{self.predict_fn}"), num_samples=1000)

        # store result
        temp_effect_val = dict()
        for key in exp.local_exp.keys():
            temp_effect_val[key] = [(int(f_idx), val) for (f_idx, val) in exp.local_exp[key]]
        line_rst_dict["lime_effect_val"] = temp_effect_val
        line_rst_dict["lime_class_names"] = exp.class_names
        line_rst_dict["lime_predict_proba"] = exp.predict_proba.tolist()
        line_rst_dict["inference_result"] = int(inferenced_y[line_idx])
        line_rst_dict["origin_idx_dict"] = original_idx_dict
        line_rst_dict["cvt_dict"] = cvt_dict
        line_rst_dict["lime_cvt_text"] = " ".join(reversed_data)
        line_rst_dict["lime_text_highlight_idx"] = exp.domain_mapper.indexed_string.positions
        line_rst_dict["lime_inverse_vocab"] = exp.domain_mapper.indexed_string.inverse_vocab

        return line_rst_dict

    def text_data_run_deprecated(self, cvt_data, json_data, line_idx):

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

    def tabular_data_run(self, cvt_data, json_data, line_idx):
        inferenced_y = self.model_inference(cvt_data)
        line_rst_dict = dict()

        exp = self.explainer.explain_instance(
            np.array(cvt_data[line_idx]), predict_fn=self.predict_fn,
            num_samples=1000, num_features=6
        )
        """
            exp.domain_mapper.feature_names : feature name
            exp.local_exp : (feature idx, 점수)
            exp.domain_mapper.feature_values : feature value
            exp.domain_mapper.discretized_feature_names : 값의 범위
            exp.calss_names : class name
        """
        temp_effect_val = dict()
        for key in exp.local_exp.keys():
            temp_effect_val[key] = [(int(f_idx), val) for (f_idx, val) in exp.local_exp[key]]
        line_rst_dict["lime_effect_val"] = temp_effect_val
        line_rst_dict["lime_discretized_feature_names"] = exp.domain_mapper.discretized_feature_names
        line_rst_dict["lime_class_names"] = exp.class_names
        line_rst_dict["lime_predict_proba"] = exp.predict_proba.tolist()
        line_rst_dict["lime_feature_names"] = exp.domain_mapper.feature_names
        line_rst_dict["lime_feature_values"] = exp.domain_mapper.feature_values
        line_rst_dict["inference_result"] = int(inferenced_y[line_idx])
        line_rst_dict["cvt_data"] = cvt_data[line_idx]

        return line_rst_dict

    def image_data_run(self, cvt_data, json_data, line_idx):
        inferenced_y = self.model_inference(cvt_data)

        line_rst_dict = dict()

        exp = self.explainer.explain_instance(
            np.array(cvt_data[line_idx]),
            classifier_fn=self.predict_fn,  # 각 class 확률 반환
            num_samples=1000,              # sample space
            segmentation_fn=self.segmenter  # 분할 알고리즘
        )
        img, mask = exp.get_image_and_mask(exp.top_labels[0])
        self.LOGGER.info(mask)
        self.LOGGER.info(f"inference rst : [{exp.top_labels[0]}]")

        mask = np.expand_dims(mask, axis=2)
        masked_img: np.ndarray = img * mask
        masked_thumbnail_img = CV2Utils.resize(masked_img.astype("float32"), (256, 256))
        line_json = json_data[line_idx]

        self.make_rst_image(line_idx, line_json, masked_img, masked_thumbnail_img)

        file_path: AnyStr = line_json["file_path"]
        thumbnail_path: AnyStr = line_json["thumbnail_path"]
        line_rst_dict["file_path"] = file_path.replace(Constants.DIR_WEB_FILE, "")
        line_rst_dict["thumbnail_path"] = thumbnail_path.replace(Constants.DIR_WEB_FILE, "")
        line_rst_dict["masked_file_path"] = f"/xai/masked_image/{self.job_info.get_hist_no()}"
        line_rst_dict["masked_thumbnail_path"] = f"/xai/masked_thumbnail_image/{self.job_info.get_hist_no()}_thumbnail"

        line_rst_dict["inference_result"] = int(inferenced_y[line_idx])
        # line_rst_dict["mask"] = mask.tolist()

        return line_rst_dict

    def make_rst_image(self, line_idx: int, line_json: Dict, masked_image, masked_thumbnail_image) -> None:
        if line_idx == 0:
            FileUtils.mkdir(f"{self.job_info.get_hist_no()}")
            FileUtils.mkdir(f"{self.job_info.get_hist_no()}_thumbnail")

        file_nm = line_json["file_ori_nm"]
        CV2Utils.imwrite(f"{self.job_info.get_hist_no()}/{file_nm}", masked_image)
        CV2Utils.imwrite(f"{self.job_info.get_hist_no()}_thumbnail/{file_nm}", masked_thumbnail_image)
