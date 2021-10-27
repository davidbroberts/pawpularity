# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import random
import cv2


with open('config/SETTINGS.json', 'r') as f:
    config = json.load(f)
    print("CONFIG LOADED:")
    print(config)

timm_path = config['TIMM_PATH']
sys.path.append(timm_path)
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
from sklearn import metrics, model_selection, preprocessing
from sklearn.metrics import mean_squared_error

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch import optim
from torchvision import transforms

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup
import wandb
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# This stuff needs to be put into a method
#


set_seed(42)


df = pd.read_csv(f"{config['DATA_DIR']}train.csv")
df = df.sample(frac = 1).reset_index(drop = True)
y = df.Pawpularity.values
kf = model_selection.StratifiedKFold(n_splits = 5, random_state=42, shuffle=True)

for f,(t,v) in enumerate(kf.split(X=df,y=y)):
    df.loc[v,'fold'] = f


df['path'] = [f"{config['TRAIN_IMAGES_PATH']}{x}.jpg" for x in df["Id"].values]
dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]


image_size = 384
train_aug = A.Compose(
    [   A.RandomResizedCrop(image_size,image_size,p= 0.8),
        A.Resize(image_size,image_size,p=1.0),
        A.HorizontalFlip(p=0.5),   
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
          
   A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ]
)
val_aug = A.Compose(
    [ 
     A.Resize(image_size,image_size,p=1.0),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ]
)

#
# ---End of stuff that needs to be in a method


def mixup(x, y, alpha=1.0):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



class Pets(Dataset):
    def __init__(self , df,augs = None):
        self.df = df
        self.augs = augs
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        img_src = self.df.loc[idx,'path']
        image = cv2.imread(img_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.augs(image=image)
        image = transformed['image']
        meta = self.df[dense_features].iloc[idx, :].values
        target = torch.tensor(self.df['Pawpularity'].values[idx],dtype = torch.float32)
        
        return image ,torch.FloatTensor(meta), target

class Model(nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True, num_classes=0, drop_rate=0., drop_path_rate=0.,global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc3_A = nn.Linear(1024,12)
        self.fc3_B = nn.Linear(1024,1)
    
    def forward(self,image):
        image = self.backbone(image)
        image = F.dropout(image , 0.35)
        dec2 = self.fc3_B(image)
        dec1 = self.fc3_A(image)
        return F.sigmoid(dec1) , dec2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(train_loader,model,optimizer,criterion,e,epochs,scheduler):
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()
    global_step = 0
    loop = tqdm(enumerate(train_loader),total = len(train_loader))
    
    for step,(image,feat ,labels) in loop:
        image = image.to(device)
        feat = feat.to(device)
        labels= labels.to(device)/100.
        output1 , output2 = model(image)
        batch_size = labels.size(0)
        
        if torch.rand(1)[0] < 0.5:
            image, targets_a, targets_b, lam = mixup(image, labels.view(-1, 1))
            loss2 = mixup_criterion(criterion, output2, targets_a, targets_b, lam)
        else:
            loss2 = criterion(output2,labels.unsqueeze(1))
        
        loss = loss2
        
        output2 = output2.sigmoid()
        out = output2.cpu().detach().numpy()
        targets = labels.cpu().detach().numpy()
        rmse = mean_squared_error(targets*100,out*100, squared=False)
        losses.update(loss.item(), batch_size)
        scores.update(rmse.item(), batch_size)
        
        loss.backward(retain_graph = True)
        optimizer.step()
        optimizer.zero_grad()
        
        '''optimizer.first_step(zero_grad=True)
        mixup_criterion(criterion, model(image)[1], targets_a, targets_b, lam).backward()
        #criterion(model(image)[1], labels.unsqueeze(1).float()).backward()
        nn.BCELoss()(model(image)[0] , feat).backward()
        optimizer.second_step(zero_grad=True)'''

        scheduler.step()
        global_step += 1
        
        loop.set_description(f"Epoch {e+1}/{epochs}")
        loop.set_postfix(loss = loss.item(), rmse= rmse.item(), stage = 'train')
        
        
    return losses.avg,scores.avg

def val_one_epoch(loader,model,optimizer,criterion):
    losses = AverageMeter()
    scores = AverageMeter()
    model.eval()
    global_step = 0
    loop = tqdm(enumerate(loader),total = len(loader))
    
    for step,(image,feat , labels) in loop:
        image = image.to(device)
        feat = feat.to(device)
        labels = labels.to(device)/100.
        batch_size = labels.size(0)
        with torch.no_grad():
            output1 , output2 = model(image)
      
        loss2 = criterion(output2,labels.unsqueeze(1))
        #loss2 = torch.sqrt(loss2)
        loss =  loss2
        output2 = output2.sigmoid()
        out = output2.cpu().detach().numpy()
        targets = labels.cpu().detach().numpy()
        rmse = mean_squared_error(targets*100, out*100, squared=False)
        
        losses.update(loss.item(), batch_size)
        scores.update(rmse.item(), batch_size)
        loop.set_postfix(loss = loss.item(), rmse  = rmse.item(), stage = 'valid')
        
  
        
    return losses.avg,scores.avg

def fit(m, fold_n, training_batch_size = config['TRAIN_BATCH_SIZE'], validation_batch_size = config['VAL_BATCH_SIZE']):
    
    train_data= df[df.fold != fold_n]
    val_data  = df[df.fold == fold_n]
    train_data= Pets(train_data.reset_index(drop=True) , augs = train_aug)
    val_data  = Pets(val_data.reset_index(drop=True) , augs = val_aug)
    
    train_loader = DataLoader(train_data, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, batch_size=training_batch_size)
    valid_loader = DataLoader(val_data, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, batch_size=validation_batch_size)
   
    criterion= nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(m.parameters(), lr = config['LR'], weight_decay = 1e-6)
    '''base_optimizer = optim.AdamW # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(m.parameters(), base_optimizer, lr=5e-4 , weight_decay = 1e-7)'''
    
    #wandb.watch(model, criterion, log="all", log_freq=10)
    
    epochs = config['EPOCHS']
    warmup_epochs = config['WARMUP_EPOCHS']
    num_train_steps = math.ceil(len(train_loader))
    num_warmup_steps= num_train_steps * warmup_epochs
    num_training_steps=int(num_train_steps * epochs)
    sch = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps = num_warmup_steps,num_training_steps =num_training_steps) 
    
    loop = range(epochs)
    for e in loop:
        
        train_loss, train_rmse = train_one_epoch(train_loader,m,optimizer,criterion,e,epochs,sch)
    
        print(f'For epoch {e+1}/{epochs}')
        print(f'average train_loss {train_loss}')
        print(f'average train_rmse {train_rmse}' )
        
        val_loss,val_rmse= val_one_epoch(valid_loader,m,optimizer,criterion)
        
        print(f'avarage val_loss { val_loss }')
        print(f'avarage val_rmse {val_rmse}')

        torch.save(m.state_dict(), config['OUTPUT_DIR'] + f'Fold {fold_n} with val_rmse {val_rmse}.pth') 
        #wandb.log({"Train RMSE": train_rmse, "Val RMSE": val_rmse, "Train loss": train_loss, "Val Loss": val_loss, "Epoch": e})

if __name__ == '__main__':
    

    if not os.path.exists(config['OUTPUT_DIR']):
        os.makedirs(config['OUTPUT_DIR'])
 
    #with wandb.init(project="Pawpularity NN", entity='mrigavid'):
    for i in range(5):
        model = Model(True)
        model= model.to(device)
        fit(model ,i)