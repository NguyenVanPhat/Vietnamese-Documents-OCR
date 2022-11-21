from dataset import MyDataset
import numpy as np
import cv2
import torch
import torch.nn as nn
import pandas as pd
from model import MyModel
from tqdm import tqdm
import random
import argparse
from torch.utils.data import DataLoader
import warnings
import os
from collections import defaultdict
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--img_dir", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--path_csv", type=str)
parser.add_argument("--resume", type=str)
parser.add_argument("--val_bs", type=int)
parser.add_argument("--num_workers", type=int)
opt = parser.parse_args()
print(opt)

@torch.no_grad()
def inference(model, val_loader, device):
    model.eval()
    stream = tqdm(enumerate(val_loader), total=len(val_loader))
    gts = []
    preds = []
    for _, data in stream:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        gts.append(targets.cpu().numpy())
        preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts

if __name__ == "__main__":
    df = pd.read_csv(opt.path_csv)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    val_df = df.loc[df["type"] == "val"].reset_index(drop=True)
    val_dataset = MyDataset(opt.img_dir, val_df, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=opt.val_bs, num_workers=opt.num_workers, shuffle=False)
    model = MyModel(opt.model_name, num_classes=2, pretrained=True)
    model = model.to(device)
    if (opt.resume is not None):
        ckpt = torch.load(opt.resume)
        model.load_state_dict(ckpt)
    preds, gts = inference(model, val_loader, device)
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_acc = 0 
    best_threshold = None
    for threshold in thresholds:
        y_hat = np.where(preds[:, 0] > threshold, 0, 1)
        acc = np.where(y_hat == gts)[0].shape[0]/len(gts)
        if (best_acc < acc):
            best_acc = acc
            best_threshold = threshold
    print(f"{best_threshold} has {best_acc} accuracy")
