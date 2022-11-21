from utils import DocScanner
import cv2
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import time

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

class Preprocessor:
    def __init__(self):
        self.scanner = DocScanner()
        
    def __call__(self, info_dict):
        print("-"*40 + "Preprocessor: Start" + "-"*40)
        start_time = time.time()
        image_dir = info_dict["config"]["general"]["img_dir"]
        preprocess_out_img_dir = info_dict["config"]["preprocessor"]["preprocess_out_img_dir"]
        preprocess_out_txt_dir = info_dict["config"]["preprocessor"]["preprocess_out_txt_dir"]
        os.makedirs(preprocess_out_img_dir, exist_ok=True)
        os.makedirs(preprocess_out_txt_dir, exist_ok=True)
        file_names = os.listdir(image_dir)
        stream = tqdm(file_names, total=len(file_names))
        for file_name in stream:
            if (file_name.split(".")[-1] in IMG_FORMATS):
                img_path = f"{image_dir}/{file_name}"
                img = cv2.imread(img_path)
                self.scanner.scan(img, vis_img_path=os.path.join(preprocess_out_img_dir, file_name), txt_path=os.path.join(preprocess_out_txt_dir, ".".join(file_name.split(".")[:-1]) + ".txt"))
        end_time = time.time()
        print("Time: ", end_time - start_time)
        print("-"*40 + "Preprocessor: End" + "-"*40)
        return info_dict


        