import os
import pandas as pd
import re
import numpy as np
import sys
sys.path = ["./", "./utils/", "./rotation_corrector/", "./text_recognizer/vietocr_github/"] + sys.path
from model import MyModel
from text_detector import TextDetector
import torch
from utils import get_boxes_arr_from_txt_file, get_mean_horizontal_angle, rotate_image_bbox_angle, rotate_0_or_180, idx2name, name2idx, IoU, text_distance, validate_ADDRESS, validate_SELLER, score_amount, get_boxes_img
from tqdm import tqdm
import cv2
from pathlib import Path
from collections import defaultdict
import pickle
from sklearn.model_selection import train_test_split

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
ROOT = str(ROOT)
DATASET = "/".join(ROOT.split("/")[:-2] + ["dataset"])
ROOT = "/".join(ROOT.split("/")[:-1])

keyword_TIMESTAMP = ["ngày", "thời gian", "giờ"]
keyword_TOTAL_COST = ["cộng tiền hàng", "thanh toán", "tại quầy", "khách phải trả", "total", "tổng tiền", "tổng cộng",
                          "cong tien hang", "thanh toan", "tai quay", "khach phai tra", "tong tien", "tong cong"]

train_dir = os.path.join(DATASET, "mcocr_public_train_test_shared_data/mcocr_train_data")
train_csv = f"{train_dir}/mcocr_train_df.csv"
train_image_dir = f"{train_dir}/train_images"
out_dir = os.path.join(DATASET, "kie_data")
out_image_dir = f"{out_dir}/images"
out_boxes_and_transcripts_dir = f"{out_dir}/boxes_and_transcripts"
out_csv = f"{out_dir}/image_list.csv"
out_train_csv = f"{out_dir}/train_list.csv"
out_val_csv = f"{out_dir}/val_list.csv"
det_txt_dir = os.path.join(DATASET, "text_detection", "txt")
reg_txt_dir = os.path.join(DATASET, "text_recognition", "txt")

train_df = pd.read_csv(train_csv)
os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_boxes_and_transcripts_dir, exist_ok=True)

model = MyModel("tf_efficientnet_b0_ns", num_classes=2, pretrained=False)
device = torch.device("cpu")
model = model.to(device)
model.load_state_dict(torch.load("./weights/rotation_corrector/tf_efficientnet_b0_ns_best.pt", map_location="cpu"))
model.eval()

def get_store_dict(df):
    list_sellers = []
    list_addresses = []
    for i, row in df.iterrows():
        if (row["anno_num"] <= 0):
            continue
        ann_texts = np.array(row["anno_texts"].split("|||"))
        ann_labels = np.array(row["anno_labels"].split("|||"))
        seller = " ".join(ann_texts[ann_labels == "SELLER"].tolist())
        address = " ".join(ann_texts[ann_labels == "ADDRESS"].tolist())

        for item in list_sellers:
            if (item["name"] == seller):
                item["count"] += 1
                break
        else:
            list_sellers.append({"name": seller, "count": 1})

        for item in list_addresses:
            if (item["name"] == address):
                item["count"] += 1
                break
        else:
            list_addresses.append({"name": address, "count": 1})
        
    ignore_indexes = []
    for j in range(len(list_sellers)):
        for k in range(j+1, len(list_sellers)):
            if (text_distance(list_sellers[j]["name"], list_sellers[k]["name"]) < 0.4):
                if (list_sellers[j]["count"] > list_sellers[k]["count"]):
                    list_sellers[j]["count"] += list_sellers[k]["count"]
                    ignore_indexes.append(k)
                else:
                    list_sellers[k]["count"] += list_sellers[j]["count"]
                    ignore_indexes.append(j)
    final_sellers = [list_sellers[i] for i in range(len(list_sellers)) if (i not in ignore_indexes)]
    ignore_indexes = []
    for j in range(len(list_addresses)):
        for k in range(j+1, len(list_addresses)):
            if (text_distance(list_addresses[j]["name"], list_addresses[k]["name"]) < 0.4):
                if (list_addresses[j]["count"] > list_addresses[k]["count"]):
                    list_addresses[j]["count"] += list_addresses[k]["count"]
                    ignore_indexes.append(k)
                else:
                    list_addresses[k]["count"] += list_addresses[j]["count"]
                    ignore_indexes.append(j)
    final_addresses = [list_addresses[i] for i in range(len(list_addresses)) if (i not in ignore_indexes)]
    return final_sellers, final_addresses
                    
def fix_wrong_key(train_df):
    out_df = []
    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        ann_num = row["anno_num"]
        if (ann_num <= 0):
            continue
        img_id = row["img_id"]
        ann_polygons = eval(row["anno_polygons"])
        ann_texts = row["anno_texts"].split("|||")  # Raw Text 
        ann_labels = row["anno_labels"].split("|||")    # SELLER, ADDRESS, TOTAL_COST, TIMESTAMP
        if ("TOTAL_TOTAL_COST" in ann_labels):
            ann_labels[ann_labels.index("TOTAL_TOTAL_COST")] = "TOTAL_COST"
        ignore = False
        for j, text in enumerate(ann_texts):
            text_lower = text.lower()
            # Fix Address
            if ("p." in text_lower and ann_labels[j] != "ADDRESS"):
                ann_labels[j] = "ADDRESS"
                ann_polygons[j]["category_id"] = name2idx["ADDRESS"]

            # Fix timestamp keys
            for kw in keyword_TIMESTAMP:
                if ((kw in text_lower) and (ann_labels[j] != "TIMESTAMP")):
                    ann_labels[j] = "TIMESTAMP"
                    ann_polygons[j]["category_id"] = name2idx["TIMESTAMP"]
            
            # Fix timestamp numbers
            if (((re.search("(3[01]|[12][0-9]|0?[1-9])/(1[0-2]|0?[1-9])/(?:[0-9]{2})?[0-9]{2}", text_lower) is not None) or (re.search("(0?\d|1\d|2[0-3]):([1-5]\d|0?\d)(:([1-5]\d|0?\d))?", text_lower) is not None)) and (ann_labels[j] != "TIMESTAMP")):
                ann_labels[j] = "TIMESTAMP"
                ann_polygons[j]["category_id"] = name2idx["TIMESTAMP"]
            
            # Fix total_cost keys
            for kw in keyword_TOTAL_COST:
                if ((kw in text_lower) and (ann_labels[j] != "TOTAL_COST")):
                    ann_labels[j] = "TOTAL_COST"
                    ann_polygons[j]["category_id"] = name2idx["TOTAL_COST"]

            # Fix total_cost amounts
            if ((re.search("(3[01]|[12][0-9]|0?[1-9])[./-](1[0-2]|0?[1-9])[./-](?:[0-9]{2})?[0-9]{2}", text_lower) is None) and (re.search("(0?\d|1\d|2[0-3])( )?[:-]( )?([1-5]\d|0?\d)(( )?[:-]( )?([1-5]\d|0?\d))?", text_lower) is None)) and (re.search("((^|[^0-9])(\d)?(\d)?(\d)?[.,](\d){3}($|[^0-9]))|([0-9]+đ)|([0-9]+000)", text_lower) is not None) and score_amount(text_lower) and ann_labels[j] == 'TIMESTAMP':
                ann_labels[j] = "TOTAL_COST"
                ann_polygons[j]["category_id"] = name2idx["TOTAL_COST"]
            
        # Fix too many keys TOTAL_COST:
        num_TOTAL_COST = np.where(np.array(ann_labels) == "TOTAL_COST")[0].shape[0]
        if (num_TOTAL_COST > 4):
            ignore = True
        
        if not ignore:
            row["anno_polygons"] = str(ann_polygons)
            row["anno_texts"] = "|||".join(ann_texts)
            row["anno_labels"] = "|||".join(ann_labels)
            out_df.append(row)
    out_df = pd.DataFrame(out_df, columns=train_df.columns)
    return out_df

# Convert txt dir (recognize dir) + train_csv -> txt dir + kie label (kie dir)
def create_kie_data_from_ann(df, reg_txt_dir, res_dict, list_sellers, list_addresses, out_boxes_and_transcripts_dir, out_path_csv):
    img_ids = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if (row["anno_num"] <= 0):
            continue
        annotations = eval(row["anno_polygons"])
        ann_texts = row["anno_texts"].split("|||")
        ann_labels = row["anno_labels"].split("|||")
        img_id = row["img_id"]
        boxes_ann = res_dict[img_id]
        ann_path = f"{reg_txt_dir}/{img_id.replace('.jpg', '.txt')}"
        boxes_list = []
        texts = []
        with open(ann_path, "r") as file:
            for line in file.readlines():
                line = line.strip("\n")
                items = line.split(",")
                boxes_list.append([int(item) for item in items[:8]])
                texts.append(",".join(items[8:]))
        labels = [1] * len(texts)
        boxes_arr = np.array(boxes_list, dtype=np.int32).reshape(-1, 4, 2)
        for j, box_arr in enumerate(boxes_arr):
            max_iou = 0
            max_index = None
            for k, box_ann in enumerate(boxes_ann):
                iou = IoU(box_ann, box_arr)
                if (max_iou < iou):
                    max_iou = iou
                    max_index = k
            if (max_iou > 0.3):
                labels[j] = name2idx[ann_labels[max_index]]
        for j, (text, label) in enumerate(zip(texts, labels)):
            if (validate_SELLER(list_sellers, text) and label != 15):
                labels[j] = 15
            if (validate_ADDRESS(list_addresses, text) and label != 16):
                labels[j] = 16
            if (score_amount(text) and label in [15, 16]):
                labels[j] = 1
            for kw in keyword_TIMESTAMP:
                if ((kw in text.lower()) and (label != 17)):
                    labels[j] = 17
            if (((re.search("(3[01]|[12][0-9]|0?[1-9])/(1[0-2]|0?[1-9])/(?:[0-9]{2})?[0-9]{2}", text.lower()) is not None) or (re.search("(0?\d|1\d|2[0-3]):([1-5]\d|0?\d)(:([1-5]\d|0?\d))?", text.lower()) is not None)) and (label != 17)):
                labels[j] = 17
            for kw in keyword_TOTAL_COST:
                if ((kw in text.lower()) and (label != 18)):
                    labels[j] = 18
        with open(f"{out_boxes_and_transcripts_dir}/{'.'.join([*img_id.split('.')[:-1], 'tsv'])}", mode="w", encoding='utf-8') as file:
            file.write("\n".join([",".join([str(j + 1), *([str(item) for item in box_arr.reshape(-1,)]), text, idx2name[label]]) for j, (box_arr, text, label) in enumerate(zip(boxes_arr, texts, labels))]))
        img_ids.append(img_id)
    with open(out_path_csv, "w") as file:
        file.write("\n".join([",".join([str(idx + 1), "receipt", file_name]) for idx, file_name in enumerate(img_ids)]))
        
def modify_train_df(df, image_dir, det_txt_dir, out_image_dir):
    file_names = df["img_id"].unique()
    stream = tqdm(enumerate(file_names), total=len(file_names))
    res_dict = {}
    for i, file_name in stream:
        img = cv2.imread(f"{image_dir}/{file_name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        file_info = df.loc[df["img_id"] == file_name].iloc[0]
        if (file_info["anno_num"] == 0):
            continue
        annotations = eval(file_info[1])
        ann_path = f"{det_txt_dir}/{'.'.join([*file_name.split('.')[:-1], 'txt'])}"
        boxes_arr = get_boxes_arr_from_txt_file(ann_path) # from text detector
        boxes_anns = []
        for j, ann in enumerate(annotations):
            polygons = []
            for item in ann["segmentation"]:
                polygons.append(np.array(item, dtype=np.int32).reshape(-1, 1, 2))
            polygons = np.concatenate(polygons, axis=0)
            rect = cv2.minAreaRect(polygons)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes_anns.append(box)
        boxes_anns = np.stack(boxes_anns, axis=0)   # from annotation
        angle = get_mean_horizontal_angle(boxes_arr, cluster=True)
        img_rotated, boxes_arr = rotate_image_bbox_angle(img, boxes_arr, angle)
        img_rotated, boxes_anns = rotate_image_bbox_angle(img, boxes_anns, angle)
        doc_angle = rotate_0_or_180(img_rotated, boxes_arr, model, 0.45, gpu=None)
        img_rotated, boxes_anns = rotate_image_bbox_angle(img_rotated, boxes_anns, doc_angle)
        cv2.imwrite(f"{out_image_dir}/{file_info['img_id']}", img_rotated[:, :, ::-1])
        res_dict[file_name] = boxes_anns
    return res_dict

def split_train_test_data(path_csv, out_train_csv, out_val_csv):
    df = pd.read_csv(path_csv, header=None)
    train_df, val_df = train_test_split(df, shuffle=True, random_state=42, test_size=0.2)
    train_df.to_csv(out_train_csv, index=False, header=False)
    val_df.to_csv(out_val_csv, index=False, header=False)

out_df = fix_wrong_key(train_df)
final_sellers, final_addresses = get_store_dict(out_df)
res_dict = modify_train_df(out_df, train_image_dir, det_txt_dir, out_image_dir)
create_kie_data_from_ann(out_df, reg_txt_dir, res_dict, final_sellers, final_addresses, out_boxes_and_transcripts_dir, out_csv)
split_train_test_data(out_csv, out_train_csv, out_val_csv)



                 
            



