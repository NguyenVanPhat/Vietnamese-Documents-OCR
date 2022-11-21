import os
import numpy as np
import cv2
import math
from sklearn.cluster import KMeans
from albumentations import *
from albumentations.pytorch import ToTensorV2
import torch
from scipy import stats
import matplotlib.pyplot as plt
import Levenshtein

idx2name = {15: "SELLER", 16: "ADDRESS", 17: "TIMESTAMP", 18: "TOTAL_COST", 1: "OTHER"}
name2idx = {v:k for k, v in idx2name.items()}

def get_boxes_arr_from_txt_file(ann_path):
    boxes_list = []
    with open(ann_path, "r") as file:
        for line in file.readlines():
            boxes_list.append([int(item) for item in line.rstrip(",\n").split(",")])
    return np.array(boxes_list, dtype=np.int32).reshape(-1, 4, 2)

def drop_box(boxes_arr, drop_gap=[0.5, 2.]):
    res = []
    for box in boxes_arr:
        box_arr = box.reshape(-1, 1, 2)
        rect = cv2.minAreaRect(box_arr)
        w, h = rect[1]
        if (drop_gap[0] < w/h < drop_gap[1]):
            continue
        res.append(box)
    return np.array(res, dtype=np.int32)

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def check_max_wh_ratio(pts):
    if (len(pts) == 4):
        first_edge = euclidean_distance(pts[0], pts[1])
        second_edge = euclidean_distance(pts[1], pts[2])
        if first_edge > second_edge:
            long_edge_vector = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        else:
            long_edge_vector = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
        max_ratio = max(first_edge/second_edge, second_edge/first_edge)
        return max_ratio, long_edge_vector
    else:
        raise ValueError("check_max_wh_ratio: Polygon is not qualitareal")

def get_horizontal_angle(pts):
    # pts shape [4, 2]
    if (len(pts) == 4):
        max_ratio, long_edge_vector = check_max_wh_ratio(pts)
        angle_with_horizontal_line = math.atan2(long_edge_vector[1], long_edge_vector[0]) * 180/math.pi
        return angle_with_horizontal_line
    else:
        raise ValueError("get_horizontal_angle: Polygon is not qualitareal")

def filter_outlier_angle(angles, thresh=45):
    angles = np.array(angles)
    abs_angles = np.absolute(angles)
    if (np.max(abs_angles) - np.min(abs_angles) < thresh):
        return angles.tolist(), np.arange(len(angles))
    z = np.abs(stats.zscore(angles)).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(z)
    cluster_labels = kmeans.predict(z)
    u, c = np.unique(cluster_labels, return_counts=True)
    cluster_indexes = np.where(cluster_labels == u[np.argmax(c)])[0]
    return angles[cluster_indexes].tolist(), cluster_indexes

def filter_box(boxes_arr):
    if not len(boxes_arr):
        return np.array([])
    all_box_angle = []
    for box in boxes_arr:
        angle_with_horizontal_line = get_horizontal_angle(box)
        all_box_angle.append(angle_with_horizontal_line)
    angles, indexes = filter_outlier_angle(all_box_angle)
    return boxes_arr[indexes]
    
def get_mean_horizontal_angle(arr_pts, cluster=True):
    if not len(arr_pts):
        return 0
    angles = []
    for pts in arr_pts:
        angle_with_horizontal_line = get_horizontal_angle(pts)
        angles.append(angle_with_horizontal_line)
    if (cluster):
        angles, _ = filter_outlier_angle(angles)
    mean_angle = np.mean(angles)
    return mean_angle

def rotate_image_bbox_angle(img, boxes_arr, angle):
    if (angle != 0):
        h_org, w_org = img.shape[:2]
        cX, cY = (w_org/2, h_org/2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, scale=1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h_org * sin + w_org * cos)
        new_h = int(h_org * cos + w_org * sin)
        M[0, 2] += (new_w/2) - cX
        M[1, 2] += (new_h/2) - cY
        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        full_boxes = boxes_arr.reshape(-1, 2)
        transformed_boxes = M.dot(np.concatenate([full_boxes, np.ones((len(full_boxes), 1), dtype=np.int32)], axis=-1).T).T.reshape(-1, 4, 2)
        transformed_boxes = np.rint(transformed_boxes).astype(np.int32)
        return rotated_img, transformed_boxes
    else:
        return img, boxes_arr

def write_ann_file(boxes_arr, texts, save_path, prob_thresh=0.7):
    boxes_tmp = boxes_arr.reshape(-1, 8)
    full_txt = []
    for i, box in enumerate(boxes_tmp):
        new_txt = ",".join([str(item) for item in box])
        prob = 0
        text = None
        if (texts is not None):
            text, prob = texts[i]
        if (prob < prob_thresh):
            new_txt += ","
        else:
            new_txt += "," + text
        full_txt.append(new_txt)
    full_txt = "\n".join(full_txt)        
    with open(save_path, "w") as file:
        file.write(full_txt)

color_map = {15: (0, 255, 0), 16: (0, 0, 255), 17: (255, 0, 255), 18: (0, 255, 255)}

def visualize_img(img_path, ann_path, save_path=None, kie_labels=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, _ = plt.subplots(1)
    fig.set_size_inches(20, 20)
    with open(ann_path, 'r', encoding='utf-8') as f:
        ann_txt = f.readlines()
    for ann in ann_txt:
        ann = ann.rstrip('\n')
        items = ann.split(",")
        pts = items[:8]
        if (kie_labels):
            label = items[-1]
            text = ",".join(items[8:-1])
        else:
            label = None
            text = ",".join(items[8:])
        pts = np.array([int(item) for item in pts]).reshape(-1, 2)
        x_max, y_max = np.max(pts, axis=0)
        img = cv2.polylines(img, [pts], True, (255, 0, 0) if (not kie_labels) else color_map[name2idx[label]], thickness=2)
        if (len(text) > 0):
            plt.text(x_max + 5, y_max - 10, text, fontsize=20,
                    fontdict={"color": "r"})
    plt.imshow(img)
    plt.axis("off")
    if (save_path is not None):
        fig.savefig(save_path, bbox_inches='tight')

def rotation_and_crop(img, pts, rotate=True, extend=True, extend_x_ratio=0.05, extend_y_ratio=0.0001, min_extend_x=1, min_extend_y=2):
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    h_org, w_org = rect[1]
    w_org = int(round(w_org))
    h_org = int(round(h_org))
    if (extend):
        w, h = (w_org, h_org) if (w_org > h_org) else (h_org, w_org)
        ex = int(round(max([min_extend_x, extend_x_ratio * w])))
        ey = int(round(max([min_extend_y, extend_y_ratio * h])))
        if (w_org < h_org):
            ex, ey = ey, ex
    src_pts = box.astype(np.float32)
    dst_pts = np.array([
        [w_org - 1, h_org - 1],
        [0, h_org - 1],
        [0, 0],
        [w_org - 1, 0]
    ], dtype="float32")
    dst_pts = dst_pts + np.array([ex, ey], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, dsize=(w_org + 2*ex, h_org + 2*ey))
    h, w = warped.shape[:2]
    if (w < h and rotate):
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def get_boxes_img(img, boxes_arr, extend_box=True, extend_y_ratio=0.05, min_extend_y=1, extend_x_ratio=0.05, min_extend_x=2):
    boxes_img = []
    for box in boxes_arr:
        box = box.reshape(-1, 1, 2)
        box_img = rotation_and_crop(img, box, rotate=True, extend=extend_box, min_extend_y=min_extend_y, extend_y_ratio=extend_y_ratio, extend_x_ratio=extend_x_ratio, min_extend_x=min_extend_x)
        boxes_img.append(box_img)
    return boxes_img

def transform_image(img):
    tfs = Compose([
                Resize(75, 225, p=1.),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.)
    ])
    transformed_img = tfs(image=img)["image"]
    return transformed_img

@torch.no_grad()
def rotate_0_or_180(img, boxes_arr, model, threshold, gpu):
    is_gpu = False if (gpu is None) else True
    device = torch.device("cuda" if (is_gpu) else "cpu")
    boxes_img = get_boxes_img(img, boxes_arr, extend_box=True, extend_x_ratio=0.001, extend_y_ratio=0.001, min_extend_x=1, min_extend_y=1)
    result_dict = {0: 0, 1: 0}
    for box_img in boxes_img:
        box_tensor = transform_image(box_img)
        box_tensor = torch.unsqueeze(box_tensor, dim=0)
        box_tensor = box_tensor.to(device)
        output = model(box_tensor)
        pred = torch.softmax(output, dim=1).cpu().numpy()[0, 0]
        pred_label = 1 - int(pred > threshold)
        result_dict[pred_label] += 1
    if (result_dict[0] > result_dict[1]):
        return 0
    else:
        return 180

def from_txt_to_boxes_and_transcripts(txt_dir, boxes_and_transcripts_dir):
    file_names = [file for file in os.listdir(txt_dir) if (file.endswith("txt"))]
    for file_name in file_names:
        with open(f"{txt_dir}/{file_name}", mode="r", encoding="utf-8") as file:
            lines = file.readlines()
        lines = [f"{i+1},{line}" for i, line in enumerate(lines)]
        with open(f"{boxes_and_transcripts_dir}/{'.'.join(file_name.split('.')[:-1] + ['tsv'])}", mode="w", encoding="utf-8") as file:
            file.writelines(lines) 

def score_amount(s):
    if len(s) == 0:
        return False
    count = 0
    s = s.lower()
    for ch in s:
        if ch in '0123456789-,.đdvn ':
            count += 1
    score = count / len(s)
    return score > 0.8
    
def text_distance(text_1, text_2):
    return Levenshtein.distance(text_1, text_2)/max(len(text_1), len(text_2)) if (max(len(text_1), len(text_2)) > 0) else 0

def IoU(box1, box2):
    max_w1, max_h1 = np.max(box1, axis=0)
    max_w2, max_h2 = np.max(box2, axis=0)
    max_w = max(max_w1, max_w2)
    max_h = max(max_h1, max_h2)

    first_poly_mask = np.zeros((max_h, max_w)).astype(np.int32)
    cv2.fillPoly(first_poly_mask, [box1], [255, 255, 255])

    second_poly_mask = np.zeros((max_h, max_w)).astype(np.int32)
    cv2.fillPoly(second_poly_mask, [box2], [255, 255, 255])

    intersection = np.logical_and(first_poly_mask, second_poly_mask)
    union = np.logical_or(first_poly_mask, second_poly_mask)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def validate_SELLER(list_seller, input_str, cer_thres=0.2):
    if len(input_str) < 10:
        return False
    input_str = input_str.lower()
    min_cer = 1
    for seller in list_seller:
        if seller["count"] > 1:
            cer = text_distance(seller["name"].lower(), input_str)
            if cer < min_cer:
                min_cer = cer
    return True if min_cer < cer_thres else False

def validate_ADDRESS(list_address, input_str, cer_thres=0.2):
    if len(input_str) < 10:
        return False
    input_str = input_str.lower()
    min_cer = 1
    for address in list_address:
        if address["count"] > 1:
            cer = text_distance(address["name"].lower(), input_str)
            if cer < min_cer:
                min_cer = cer
    return True if min_cer < cer_thres else False





