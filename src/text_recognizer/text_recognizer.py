import os
from tqdm import tqdm
import cv2
import warnings
from utils.general import visualize_img, write_ann_file
warnings.filterwarnings("ignore")
from viet_text_recognition import VietTextRecognition
from utils import get_boxes_img, get_boxes_arr_from_txt_file, write_ann_file, visualize_img
import torch
import time

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

class TextRecognizer():
    def __init__(self):
        pass

    def _init_viet_ocr_model(self, base_config_path, config_path, weight_path, gpu):
        model = VietTextRecognition(base_config_path, config_path, device=gpu, pretrained_backbone=False, ckpt_path=weight_path)
        return model

    def __call__(self, info_dict):
        print("-"*40 + "Text Recognizer: Start" + "-"*40)
        start_time = time.time()
        gpu = info_dict["config"]["general"]["gpu"]
        image_dir = info_dict["config"]["rotation_corrector"]["rot_out_img_dir"]
        txt_dir = info_dict["config"]["rotation_corrector"]["rot_out_txt_dir"]
        weight_path = info_dict["config"]["text_recognizer"]["weight_path"]
        visualize = info_dict["config"]["text_recognizer"]["visualize"]
        reg_out_visualize_dir = info_dict["config"]["text_recognizer"]["reg_out_visualize_dir"]
        reg_out_txt_dir = info_dict["config"]["text_recognizer"]["reg_out_txt_dir"]
        os.makedirs(reg_out_visualize_dir, exist_ok=True)
        os.makedirs(reg_out_txt_dir, exist_ok=True)
        reg_ocr_thresh = info_dict["config"]["text_recognizer"]["reg_ocr_thresh"]
        reg_base_config_path = info_dict["config"]["text_recognizer"]["reg_base_config_path"]
        reg_config_path = info_dict["config"]["text_recognizer"]["reg_config_path"]
        model = self._init_viet_ocr_model(reg_base_config_path, reg_config_path, weight_path, gpu)
        file_names = os.listdir(image_dir)
        file_names = [file_name for file_name in file_names if (file_name.split(".")[-1] in IMG_FORMATS)]
        stream = tqdm(file_names, total=len(file_names))
        for file_name in stream:
            ann_path = os.path.join(txt_dir, ".".join([*file_name.split(".")[:-1], "txt"]))
            img_path = os.path.join(image_dir, file_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes_arr = get_boxes_arr_from_txt_file(ann_path)
            boxes_img = get_boxes_img(img, boxes_arr, extend_box=True, extend_x_ratio=0.001, extend_y_ratio=0.001, min_extend_x=1, min_extend_y=1)
            rets = model(boxes_img)
            out_ann_path = os.path.join(reg_out_txt_dir, ".".join([*file_name.split(".")[:-1], "txt"]))
            write_ann_file(boxes_arr, rets, out_ann_path, prob_thresh=reg_ocr_thresh)
            if (visualize):
                out_vis_path = os.path.join(reg_out_visualize_dir, file_name)
                visualize_img(img_path, out_ann_path, save_path=out_vis_path)
        end_time = time.time()
        print("Time: ", end_time - start_time)
        print("-"*40 + "Text Recognizer: End" + "-"*40)
        return info_dict