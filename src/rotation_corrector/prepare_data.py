import yaml
import sys
import os
from tqdm import tqdm
sys.path = ["../", "../utils/", "../text_recognizer/vietocr_github/", "../text_recognizer/"] + sys.path
from vietocr_github import VietTextRecognition
from utils import get_boxes_arr_from_txt_file, get_mean_horizontal_angle, rotate_image_bbox_angle, drop_box, write_ann_file, visualize_img, get_boxes_img
import cv2
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from shutil import copy
from sklearn.model_selection import KFold

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
ROOT = str(ROOT)
DATASET = "/".join(ROOT.split("/")[:-2] + ["dataset"])

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

with open("../configs/text_detector.yaml", "r") as file:
    cfg_text_detector = yaml.load(file, Loader=yaml.FullLoader)
with open("../configs/preprocessor.yaml", "r") as file:
    cfg_preprocessor = yaml.load(file, Loader=yaml.FullLoader)
recognizer = VietTextRecognition("../text_recognizer/vietocr_github/config/base.yml", "../text_recognizer/vietocr_github/config/vgg-seq2seq.yml", device="0", ckpt_path="../weights/text_recognizer/transformerocr.pth")
cfg_text_detector["det_out_txt_dir"] = os.path.join(DATASET, cfg_text_detector["det_out_txt_dir"])
cfg_preprocessor["preprocess_out_img_dir"] = os.path.join(DATASET, cfg_preprocessor["preprocess_out_img_dir"])
image_dir = cfg_preprocessor["preprocess_out_img_dir"]
file_names = os.listdir(image_dir)
file_names = [file for file in file_names if (file.split(".")[-1] in IMG_FORMATS)]
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
fold = 0
for train_idx, val_idx in kfold.split(file_names):
    break
stream = tqdm(enumerate(file_names), total=len(file_names))
os.makedirs("../../dataset/data0_or_180", exist_ok=True)
data = []
for i, file_name in stream:
    is_train = i in train_idx
    img_path = f"{image_dir}/{file_name}"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    anno_path = os.path.join(cfg_text_detector["det_out_txt_dir"], file_name.replace(".jpg", ".txt"))
    boxes_arr = get_boxes_arr_from_txt_file(anno_path)
    boxes_arr = drop_box(boxes_arr)
    try:
        angle = get_mean_horizontal_angle(boxes_arr, cluster=True)
        img_rotated, boxes_arr_rotated = rotate_image_bbox_angle(img, boxes_arr, angle)
    except:
        raise ValueError(img_path)
    boxes_img = get_boxes_img(img, boxes_arr, extend_box=True, extend_x_ratio=0.001, extend_y_ratio=0.001, min_extend_x=1, min_extend_y=1)
    if (len(boxes_img) == 0):   continue
    results = recognizer(boxes_img)
    all_probs = []
    all_texts = []
    for res in results:
        all_texts.append(res[0])
        all_probs.append(res[1])
    mean_prob = np.mean(all_probs)
    if (mean_prob < 0.7):   continue
    name = file_name.split(".")[0]
    for j, box_img in enumerate(boxes_img):
        new_path = f"../../dataset/data0_or_180/{name}_{j}.jpg"
        data.append([f"{name}_{j}.jpg", "train" if (is_train) else "val"])
        cv2.imwrite(new_path, box_img[:, :, ::-1])

df = pd.DataFrame(data, columns=["file_name", "type"])
df.to_csv("../../dataset/data0_or_180.csv", index=False)