import torch
import argparse
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader import FacadeDataset
from architecture import *
from tqdm import tqdm
import wandb
from losses import SADPixelwise, Loss_MRAE, Loss_RMSE
import numpy as np
import albumentations as A
from utils import HsiMaterial, make_plot_train, make_plot_val
from metrics import Metrics
from architecture.unet import UNetWithResnet50Encoder
from architecture.FPN import hrnet_fpn
from architecture.BiFPN import b4_bifpn
import segmentation_models_pytorch.losses as losses
import matplotlib.pyplot as plt
from losses import FocalLoss
from pathlib import Path
import os

os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'

parser = argparse.ArgumentParser(description="Train the model for the DMS dataset")
parser.add_argument("--model, type=str", default='FPN', help='model name')
parser.add_argument("--batch_size", type=int, default=12, help="batch size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--data_root", type=str, default='./dms_dataset/DMS_v1/')
parser.add_argument("--gpu", type=str, default='0', help='gpu')
parser.add_argument("--exp_name", type=str, default='mst_plus_plus', help='log name')
parser.add_argument("--loss" , type=str, default='ce', help='loss function')

parser.add_argument("--w1", type=float, default=3, help='w1')
parser.add_argument("--w2", type=float, default=0.5, help='w2')
args = parser.parse_args()

wandb.login(key='YOUR-WANDB-KEY')
wandb.init(project="cvpr_paper_material", name=args.exp_name)
wandb.config.update(args)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
                        scale_limit=0.1,
                        rotate_limit=(5, 30),
                        p=0.5)
    ],
    additional_targets={'image0': 'image', 'depth': 'image'}
)

dataset_train = FacadeDataset(Path(args.data_root) / 'train', aug_transform=transform)
data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=32,
 pin_memory=True)

dataset_test = FacadeDataset(Path(args.data_root) / 'test')
data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=32,
 pin_memory=True)

if args.model == 'unet':
    model = UNetWithResnet50Encoder(31).to(device)
elif args.model == 'FPN':
    model = hrnet_fpn().to(device)
elif args.model == 'BiFPN':
    model = b4_bifpn().to(device)

print('Parameters number is ', sum(param.numel() for param in model.parameters()))

optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
metric_test = Metrics('test')

criterion = None
if args.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'focal':
    criterion =  losses.FocalLoss("multiclass", gamma=2)
elif args.loss == 'tversky':
    criterion = losses.TverskyLoss("multiclass", alpha=0.6, beta=0.4)

criterionRecon = nn.MSELoss()
criterion = criterion.to(device)
criterionRecon.to(device)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=18, factor=0.5, verbose=True,  min_lr=1e-5)

print("gpu numbers:",torch.cuda.device_count())
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)


def train(model, data_loader, optimizer, lossfunc):
    model.train()
    running_loss = []

    for cubes, images, labels in tqdm(data_loader):

        images = images.to(device)
        labels = labels.to(device)
        cubes = cubes.to(device)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs , hs = model(images)
            
            loss = args.w1 * lossfunc(outputs, labels.long()) + args.w2 * criterionRecon(hs, cubes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.model == 'BiFPN' or args.model == 'FPN' or args.model == 'unet':
            with torch.no_grad():
                model.module.sam.members.clamp_(0 + 1e-5, 1 - 1e-5)

        running_loss.append(loss.item())

    epoch_loss = sum(running_loss) / len(data_loader.dataset)

    return epoch_loss


def validate(model, data_loader, lossfunc):
    model.eval()
    running_loss = []

    with torch.no_grad():
        for cubes, inputs,  labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            cubes = cubes.to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs, hs = model(inputs)
                
                loss = args.w1 * lossfunc(outputs, labels.long()) + args.w2 * criterionRecon(hs, cubes)

            running_loss.append(loss.item())

            pred = nn.functional.softmax(outputs, dim=1)
            metric_test.update(pred.argmax(1), labels.long())

    pixel_acc, macc, miou = metric_test.compute()
    metric_test.reset()
    
    val_loss = sum(running_loss) / len(running_loss)
    scheduler.step(miou)

    fig = make_plot_val(inputs, outputs, labels)


    return val_loss,fig, pixel_acc, macc, miou


best_val_miou = 0
scaler = torch.cuda.amp.GradScaler()
for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')

    epoch_loss  = train(model, data_loader_train, optimizer, criterion)

    val_loss, fig_val, pixel_acc, macc, miou = validate(model, data_loader_test, criterion)

    if best_val_miou > miou:
        best_val_miou = miou
        torch.save(model.state_dict(), 'models/best_model.pth')

    wandb.log({
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_fig': fig_val,
            'epoch': epoch,
            'pixel_acc_val': pixel_acc, 'macc_val': macc, 'miou_val': miou,
            'lr': optimizer.param_groups[0]['lr']
            })

    print(f'Epoch {epoch} train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}')

