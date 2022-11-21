from torch.utils.data import Dataset
from albumentations import *
from albumentations.pytorch import ToTensorV2
import cv2
import os

class MyDataset(Dataset):
    def __init__(self, img_dir, df, mode="train", visualize=False):
        super().__init__()
        self.__dict__.update(locals())
        self.transforms = {
            "train": Compose([
                Resize(75, 225, p=1.),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.) 
            ], p=1.) if (not visualize) else None,
            "val": Compose([
                Resize(75, 225, p=1.),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.) 
            ]) if (not visualize) else None
        }

    def __len__(self):
        return 2 * len(self.df)

    def __getitem__(self, index):
        target = 0
        if (index >= len(self.df)):
            target = 1
            index = index - len(self.df)
        file_name = self.df.iloc[index]["file_name"]
        img = cv2.imread(os.path.join(self.img_dir, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (target):
            img = cv2.rotate(img, cv2.ROTATE_180)
        if (self.transforms[self.mode] is not None):
            img = self.transforms[self.mode](image=img)["image"]
        return (img, target)