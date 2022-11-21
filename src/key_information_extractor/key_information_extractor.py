import os
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings("ignore")
from utils import from_txt_to_boxes_and_transcripts
import time
import subprocess

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

class KeyInformationExtractor():
    def __init__(self):
        pass
    
    def __call__(self, info_dict):
        print("-"*40 + "Key Information Extractor: Start" + "-"*40)
        start_time = time.time()
        # Create boxes_and_transcripts dir
        txt_dir = info_dict["config"]["text_recognizer"]["reg_out_txt_dir"]
        boxes_and_transcripts_dir = info_dict["config"]["key_information_extractor"]["kie_out_boxes_and_transcripts_dir"]
        os.makedirs(boxes_and_transcripts_dir, exist_ok=True)
        from_txt_to_boxes_and_transcripts(txt_dir, boxes_and_transcripts_dir)
        # Inference KIE
        image_dir = info_dict["config"]["rotation_corrector"]["rot_out_img_dir"]
        weight_path = info_dict["config"]["key_information_extractor"]["weight_path"]
        kie_out_txt_dir = info_dict["config"]["key_information_extractor"]["kie_out_txt_dir"]
        kie_out_visualize_dir = info_dict["config"]["key_information_extractor"]["kie_out_visualize_dir"]
        os.makedirs(kie_out_txt_dir, exist_ok=True)
        os.makedirs(kie_out_visualize_dir, exist_ok=True)
        gpu = info_dict["config"]["general"]["gpu"]
        gpu = -1 if (gpu is None) else int(gpu)
        cmd = f"cd ./key_information_extractor/PICK-pytorch && python3 test.py --checkpoint {weight_path} --boxes_transcripts {boxes_and_transcripts_dir} --images_path {image_dir} --output_folder {kie_out_txt_dir} --gpu {gpu} --bs 1 --visualize_dir {kie_out_visualize_dir}"
        subprocess.call(cmd, shell=True)
        end_time = time.time()
        print("Time: ", end_time - start_time)
        print("-"*40 + "Key Information Extractor: End" + "-"*40)
        return info_dict 