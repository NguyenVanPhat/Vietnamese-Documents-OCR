import argparse
import yaml
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys
sys.path = ["./utils/", "./text_recognizer/vietocr_github/", "./rotation_corrector/"] + sys.path
from preprocessor import Preprocessor
from text_detector import TextDetector
from rotation_corrector import RotationCorrector
from text_recognizer import TextRecognizer
from key_information_extractor import KeyInformationExtractor
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
ROOT = str(ROOT)
DATASET = "/".join(ROOT.split("/")[:-1] + ["dataset"]) 

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", type=str)
parser.add_argument("--img_dir", type=str)
parser.add_argument("--use_kie", action="store_true")
opt = parser.parse_args()
print(opt)

class DocumentOCR:
    def __init__(self, cfg_general, cfg_preprocessor, cfg_text_detector, cfg_rotation_corrector, cfg_text_recognizer, cfg_key_information_extractor):
        self.__dict__.update(locals())
        self.preprocessor = Preprocessor()
        self.text_detector = TextDetector()
        self.rotation_corrector = RotationCorrector()
        self.text_recognizer = TextRecognizer()
        self.key_information_extractor = KeyInformationExtractor()

    def run(self):
        info_dict = {
            "config": {
                "general": self.cfg_general,
                "preprocessor": self.cfg_preprocessor,
                "text_detector": self.cfg_text_detector,
                "rotation_corrector": self.cfg_rotation_corrector,
                "text_recognizer": self.cfg_text_recognizer,
                "key_information_extractor": self.cfg_key_information_extractor
            }
        }
        # OCR run
        out_dict = self.text_recognizer(self.rotation_corrector(self.text_detector(self.preprocessor(info_dict))))
        # KIE run
        if (self.cfg_general["use_kie"]):
            self.key_information_extractor(out_dict)
             
if __name__ == "__main__":
    cfg_general = vars(opt)
    with open("./configs/preprocessor.yaml", "r") as file:
        cfg_preprocessor = yaml.load(file, Loader=yaml.FullLoader)
    with open("./configs/text_detector.yaml", "r") as file:
        cfg_text_detector = yaml.load(file, Loader=yaml.FullLoader)
    with open("./configs/rotation_corrector.yaml", "r") as file:
        cfg_rotation_corrector = yaml.load(file, Loader=yaml.FullLoader)
    with open("./configs/text_recognizer.yaml", "r") as file:
        cfg_text_recognizer = yaml.load(file, Loader=yaml.FullLoader)
    with open("./configs/key_information_extractor.yaml", "r") as file:
        cfg_key_information_extractor = yaml.load(file, Loader=yaml.FullLoader)
    cfg_preprocessor["preprocess_out_img_dir"] = os.path.join(DATASET, cfg_preprocessor["preprocess_out_img_dir"])
    cfg_preprocessor["preprocess_out_txt_dir"] = os.path.join(DATASET, cfg_preprocessor["preprocess_out_txt_dir"])
    cfg_text_detector["weight_dir"] = os.path.join(ROOT, cfg_text_detector["weight_dir"])
    cfg_text_detector["det_out_visualize_dir"] = os.path.join(DATASET, cfg_text_detector["det_out_visualize_dir"])
    cfg_text_detector["det_out_txt_dir"] = os.path.join(DATASET, cfg_text_detector["det_out_txt_dir"])
    cfg_rotation_corrector["rot_out_img_dir"] = os.path.join(DATASET, cfg_rotation_corrector["rot_out_img_dir"])
    cfg_rotation_corrector["rot_out_visualize_dir"] = os.path.join(DATASET, cfg_rotation_corrector["rot_out_visualize_dir"])
    cfg_rotation_corrector["rot_out_txt_dir"] = os.path.join(DATASET, cfg_rotation_corrector["rot_out_txt_dir"])
    cfg_rotation_corrector["weight_path"] = os.path.join(ROOT, cfg_rotation_corrector["weight_path"])
    cfg_text_recognizer["weight_path"] = os.path.join(ROOT, cfg_text_recognizer["weight_path"])
    cfg_text_recognizer["reg_out_visualize_dir"] = os.path.join(DATASET, cfg_text_recognizer["reg_out_visualize_dir"])
    cfg_text_recognizer["reg_out_txt_dir"] = os.path.join(DATASET, cfg_text_recognizer["reg_out_txt_dir"])
    cfg_text_recognizer["reg_base_config_path"] = os.path.join(ROOT, cfg_text_recognizer["reg_base_config_path"])
    cfg_text_recognizer["reg_config_path"] = os.path.join(ROOT, cfg_text_recognizer["reg_config_path"])
    cfg_key_information_extractor["weight_path"] = os.path.join(ROOT, cfg_key_information_extractor["weight_path"])
    cfg_key_information_extractor["kie_out_txt_dir"] = os.path.join(DATASET, cfg_key_information_extractor["kie_out_txt_dir"])
    cfg_key_information_extractor["kie_out_visualize_dir"] = os.path.join(DATASET, cfg_key_information_extractor["kie_out_visualize_dir"])
    cfg_key_information_extractor["kie_out_boxes_and_transcripts_dir"] = os.path.join(DATASET, cfg_key_information_extractor["kie_out_boxes_and_transcripts_dir"])
    document_OCR = DocumentOCR(cfg_general=cfg_general, cfg_preprocessor=cfg_preprocessor, cfg_text_detector=cfg_text_detector, cfg_rotation_corrector=cfg_rotation_corrector, cfg_text_recognizer=cfg_text_recognizer, cfg_key_information_extractor=cfg_key_information_extractor)
    document_OCR.run()
    
    