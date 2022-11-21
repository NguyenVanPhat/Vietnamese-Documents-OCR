import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from model import MyModel
from tqdm import tqdm
import argparse
from dataset import MyDataset
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
parser.add_argument("--train_bs", type=int)
parser.add_argument("--val_bs", type=int)
parser.add_argument("--num_workers", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--num_epochs_per_train", type=int)
parser.add_argument("--min_lr", type=float)
parser.add_argument("--accum_grad", type=int)
parser.add_argument("--clip_grad_norm", type=float)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--save_iter", type=int)
opt = parser.parse_args()
print(opt)

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, count=1):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += count
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
    
    def __getitem__(self, metric_name):
        metric = self.metrics[metric_name]
        return metric["avg"]
    
def train_fn(epoch, model, train_loader, optimizer, scheduler, device):
    model.train()
    metric_monitor = MetricMonitor()
    stream = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in stream:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if (opt.accum_grad > 1):
            loss /= opt.accum_grad
        loss.backward()
        metric_monitor.update("Loss", loss.item())
        if (((i + 1) % opt.accum_grad == 0) or ((i + 1) == len(train_loader))):
            if (opt.clip_grad_norm is not None):
                torch.nn.utils.clip_grad_norm_(model.paramaters(), opt.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        stream.set_description(
                "Epoch: {epoch}. Train. {metric_monitor} | lr = {lr: .5f}".format(epoch=epoch, metric_monitor=metric_monitor, lr=optimizer.param_groups[0]['lr'])
        )
    if (scheduler is not None):
        scheduler.step()
    return metric_monitor["Loss"]

@torch.no_grad()
def eval_fn(epoch, model, val_loader, device):
    model.eval()
    stream = tqdm(enumerate(val_loader), total=len(val_loader))
    metric_monitor = MetricMonitor()
    preds = []
    gts = []
    for i, data in stream:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        gts.append(targets.cpu().numpy())
        preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
        metric_monitor.update("Loss", loss.item())
        stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    acc = np.where(preds == gts)[0].shape[0]/len(gts)
    return metric_monitor["Loss"], acc

if __name__ == "__main__":
    df = pd.read_csv(opt.path_csv)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    train_df = df.loc[df["type"] == "train"].reset_index(drop=True)
    val_df = df.loc[df["type"] == "val"].reset_index(drop=True)
    train_dataset = MyDataset(opt.img_dir, train_df, mode="train")
    val_dataset = MyDataset(opt.img_dir, val_df, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=opt.train_bs, num_workers=opt.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.val_bs, num_workers=opt.num_workers, shuffle=False)
    model = MyModel(opt.model_name, num_classes=2, pretrained=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs, eta_min=opt.min_lr)
    start_epoch = -1
    best_acc = 0
    if (opt.resume is not None):
        ckpt = torch.load(opt.resume)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_acc = ckpt["best_fitness"]
        start_epoch = int(opt.resume[:-3].split("_")[-1])
    start_epoch += 1
    end_epoch = min([start_epoch + opt.num_epochs_per_train, opt.num_epochs])
    for epoch in range(start_epoch, end_epoch):
        train_loss = train_fn(epoch, model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = eval_fn(epoch, model, val_loader, device)
        if (val_acc > best_acc):
            print(f"Accuracy improve from {best_acc} to {val_acc}")
            best_acc = val_acc
            torch.save(model.state_dict(), f"{opt.save_dir}/{opt.model_name}_best.pt")
        if ((epoch + 1) % opt.save_iter == 0):
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_fitness": best_acc
            }, f"{opt.save_dir}/{opt.model_name}_{epoch}.pt")
    


