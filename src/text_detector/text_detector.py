import subprocess
import os
import warnings
warnings.filterwarnings("ignore")
import time

class TextDetector():
    def __init__(self):
        pass

    def __call__(self, info_dict):
        print("-"*40 + "Text Detector: Start" + "-"*40)
        start_time = time.time()
        use_gpu = info_dict["config"]["general"]["gpu"] != None
        image_dir = info_dict["config"]["preprocessor"]["preprocess_out_img_dir"]
        det_model_dir = info_dict["config"]["text_detector"]["weight_dir"]
        det_db_thresh = info_dict["config"]["text_detector"]["det_db_thresh"]
        det_db_box_thresh = info_dict["config"]["text_detector"]["det_db_box_thresh"]
        visualize = info_dict["config"]["text_detector"]["visualize"]
        det_out_visualize_dir = info_dict["config"]["text_detector"]["det_out_visualize_dir"]
        det_out_txt_dir = info_dict["config"]["text_detector"]["det_out_txt_dir"]
        os.makedirs(det_out_visualize_dir, exist_ok=True)
        os.makedirs(det_out_txt_dir, exist_ok=True)
        cmd = f"cd ./text_detector/PaddleOCR && python3 tools/infer/predict_det.py --use_gpu {use_gpu} --image_dir {image_dir} --det_algorithm 'DB' --det_model_dir {det_model_dir} --det_db_thresh {det_db_thresh} --det_db_box_thresh {det_db_box_thresh} --use_dilation True --det_db_score_mode 'fast' --visualize {visualize} --det_out_visualize_dir {det_out_visualize_dir} --det_out_txt_dir {det_out_txt_dir} && cd ../../"
        subprocess.call(cmd, shell=True)
        end_time = time.time()
        print("Time: ", end_time - start_time)
        print("-"*40 + "Text Detector: End" + "-"*40)
        return info_dict