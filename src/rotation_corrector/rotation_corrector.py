import os
from tqdm import tqdm
from utils import get_boxes_arr_from_txt_file, get_mean_horizontal_angle, rotate_image_bbox_angle, drop_box, write_ann_file, visualize_img, rotate_0_or_180
import cv2
import warnings
import torch
from model import MyModel
warnings.filterwarnings("ignore")
import time

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

class RotationCorrector():
    def __init__(self):
        pass
    
    def _init_rotation_angle_classification_model(self, model_name, weight_path, gpu):
        model = MyModel(model_name, num_classes=2, pretrained=False)
        is_gpu = False if (gpu is None) else True
        device = torch.device("cuda" if (is_gpu) else "cpu")
        model = model.to(device)
        if (is_gpu):
            model.load_state_dict(torch.load(weight_path))
        else:
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        model.eval()
        return model

    def __call__(self, info_dict):
        print("-"*40 + "Rotation Corrector: Start" + "-"*40)
        start_time = time.time()
        visualize = info_dict["config"]["rotation_corrector"]["visualize"]
        image_dir = info_dict["config"]["preprocessor"]["preprocess_out_img_dir"]
        rot_out_img_dir = info_dict["config"]["rotation_corrector"]["rot_out_img_dir"]
        rot_out_visualize_dir = info_dict["config"]["rotation_corrector"]["rot_out_visualize_dir"]
        rot_out_txt_dir = info_dict["config"]["rotation_corrector"]["rot_out_txt_dir"]
        model_name = info_dict["config"]["rotation_corrector"]["model_name"]
        weight_path = info_dict["config"]["rotation_corrector"]["weight_path"]
        gpu = info_dict["config"]["general"]["gpu"]
        cls_thresh = info_dict["config"]["rotation_corrector"]["cls_thresh"]
        os.makedirs(rot_out_img_dir, exist_ok=True)
        os.makedirs(rot_out_visualize_dir, exist_ok=True)
        os.makedirs(rot_out_txt_dir, exist_ok=True)
        file_names = os.listdir(image_dir)
        model = self._init_rotation_angle_classification_model(model_name, weight_path, gpu)
        stream = tqdm(file_names, total=len(file_names))
        for file_name in stream:
            if (file_name.split(".")[-1] in IMG_FORMATS):
                img_path = f"{image_dir}/{file_name}"
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ann_path = os.path.join(info_dict["config"]["text_detector"]["det_out_txt_dir"], ".".join([*file_name.split(".")[:-1], "txt"]))
                boxes_arr = get_boxes_arr_from_txt_file(ann_path)
                boxes_arr = drop_box(boxes_arr)
                try:
                    angle = get_mean_horizontal_angle(boxes_arr, cluster=True)
                    img_rotated, boxes_arr_rotated = rotate_image_bbox_angle(img, boxes_arr, angle)
                except:
                    raise ValueError(img_path)
                doc_angle = rotate_0_or_180(img_rotated, boxes_arr_rotated, model, cls_thresh, gpu)
                img_rotated, boxes_arr_rotated = rotate_image_bbox_angle(img_rotated, boxes_arr_rotated, doc_angle)
                out_img_path = os.path.join(rot_out_img_dir, file_name)
                out_ann_path = os.path.join(rot_out_txt_dir, ".".join([*file_name.split(".")[:-1], "txt"]))
                cv2.imwrite(out_img_path, img_rotated[:,:,::-1])
                write_ann_file(boxes_arr_rotated, None, out_ann_path)
                if (visualize):
                    visualize_img(out_img_path, out_ann_path, save_path=os.path.join(rot_out_visualize_dir, file_name))
        end_time = time.time()
        print("Time: ", end_time - start_time)
        print("-"*40 + "Rotation Corrector: End" + "-"*40)      
        return info_dict