from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

class VietTextRecognition:
    def __init__(self, base_config_path, config_path, device, pretrained_backbone=False, ckpt_path=None):
        self.config = Cfg.load_config(base_config_path, config_path)
        if (ckpt_path is not None):
            self.config["weights"] = ckpt_path
        self.config["cnn"]["pretrained"] = pretrained_backbone
        if (device is not None):
            self.config["device"] = f"cuda:{device}"
        else:
            self.config["device"] = "cpu"
        self.config["predictor"]["beamsearch"] = False
        self.recognizer = Predictor(self.config)
    
    def __call__(self, imgs):
        # imgs: list of image RGB
        rets = []
        for img in imgs:
            img = Image.fromarray(img)
            rets.append(self.recognizer.predict(img, True))
        return rets 
