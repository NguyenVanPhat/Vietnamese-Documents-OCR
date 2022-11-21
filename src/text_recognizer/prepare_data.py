import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
sys.path = ["../utils/", "../rotation_corrector/", "../"] + sys.path
from model import MyModel
import torch
from utils import get_mean_horizontal_angle, rotate_image_bbox_angle, rotate_0_or_180, get_boxes_img
from sklearn.model_selection import KFold
import pickle

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp' 
data_dir = "../../dataset/mcocr_public_train_test_shared_data"
train_dir = f"{data_dir}/mcocr_train_data"
train_dir_imgs = f"{train_dir}/train_images"
train_anns = pd.read_csv(f"{train_dir}/mcocr_train_df.csv")
file_names = os.listdir(train_dir_imgs)
file_names = [file for file in file_names if (file.split(".")[-1] in IMG_FORMATS)]
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
fold = 0
for train_idx, val_idx in kfold.split(file_names):
    break
model = MyModel("tf_efficientnet_b0_ns", num_classes=2, pretrained=False)
model = model.cuda()
model.load_state_dict(torch.load("../weights/rotation_corrector/tf_efficientnet_b0_ns_best.pt"))
model.eval()
os.makedirs("../../dataset/text_recognition_mcocr_data", exist_ok=True)
train_texts = []
val_texts = []
stream = tqdm(enumerate(file_names), total=len(file_names))
for i, file_name in stream:
    is_train = i in train_idx
    img = cv2.imread(f"{train_dir_imgs}/{file_name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    file_info = train_anns.loc[train_anns["img_id"] == file_name].iloc[0]
    if (file_info["anno_num"] == 0):
        continue
    annotations = eval(file_info[1])
    texts = file_info[2].split("|||")
    boxes_arr = []
    for j, (ann, text) in enumerate(zip(annotations, texts)):
        polygons = []
        for item in ann["segmentation"]:
            polygons.append(np.array(item, dtype=np.int32).reshape(-1, 1, 2))
        polygons = np.concatenate(polygons, axis=0)
        rect = cv2.minAreaRect(polygons)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes_arr.append(box)
    boxes_arr = np.stack(boxes_arr, axis=0)
    angle = get_mean_horizontal_angle(boxes_arr, cluster=True)
    img, boxes_arr = rotate_image_bbox_angle(img, boxes_arr, angle)
    doc_angle = rotate_0_or_180(img, boxes_arr, model, 0.45, gpu="0")
    img, boxes_arr = rotate_image_bbox_angle(img, boxes_arr, doc_angle)
    boxes_img = get_boxes_img(img, boxes_arr, extend_box=True, extend_x_ratio=0.001, extend_y_ratio=0.001, min_extend_x=1, min_extend_y=1)
    for j, (text, box_img) in enumerate(zip(texts, boxes_img)):
        new_name = f"{file_name.split('.')[0]}_{j}.jpg"
        cv2.imwrite(os.path.join("../../dataset/text_recognition_mcocr_data/", new_name), box_img[:,:,::-1])
        if (is_train):
            train_texts.append(f"{new_name}\t{text}")
        else:
            val_texts.append(f"{new_name}\t{text}")

full_train_text = "\n".join(train_texts)
full_val_text = "\n".join(val_texts)
with open("../../dataset/text_recognition_train_data.txt", "w") as file:
    file.write(full_train_text)
with open("../../dataset/text_recognition_val_data.txt", "w") as file:
    file.write(full_val_text)

        




