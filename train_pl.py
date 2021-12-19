import os
import sys
import json
import math
import random
import cv2

with open('config_pl/SETTINGS.json', 'r') as f:
    config = json.load(f)
    print("CONFIG LOADED ..")


timm_path = config['TIMM_PATH']
sys.path.append(timm_path)
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch import optim
from torchvision import transforms
from transformers import  get_cosine_schedule_with_warmup
import wandb
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

if not os.path.exists(config['OUTPUT_DIR']):
    os.makedirs(config['OUTPUT_DIR'])

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


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
        target = torch.tensor(self.df['Pawpularity'].values[idx],dtype = torch.long)
        
        return image, target


class Model(nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        self.backbone = timm.create_model(config['MODEL_NAME'], pretrained=True, num_classes=0, drop_rate=0.0, drop_path_rate=0.0,global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc3_A = nn.Linear(config['NUM_NEURONS'],12)
        self.fc3_B = nn.Linear(config['NUM_NEURONS'],1)
        self.do = nn.Dropout(p=0.3)
    
    
    def forward(self,image):
        image = self.backbone(image)
        
        if(len(image.shape) == 4):#for efficientnet models
            image = self.pool(image)
            image = image.view(image.shape[0], -1)

        image = self.do(image)
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
    
    for step,(image,labels) in loop:
        image = image.to(device)
        labels= labels.to(device)/100.
        _ , output2 = model(image)
        batch_size = labels.size(0)
        
        '''if torch.rand(1)[0] < 0.5:
            image, targets_a, targets_b, lam = mixup(image, labels.view(-1, 1))
            loss2 = mixup_criterion(criterion, output2, targets_a, targets_b, lam)
        else:
            loss2 = criterion(output2,labels.unsqueeze(1))'''
        
        loss2 = criterion(output2,labels.unsqueeze(1))
        loss = loss2
        
        output2 = output2.sigmoid()
        out = output2.cpu().detach().numpy()
        targets = labels.cpu().detach().numpy()
        rmse = mean_squared_error(targets*100,out*100, squared=False)
        losses.update(loss.item(), batch_size)
        scores.update(rmse.item(), batch_size)
        
        loss.backward(retain_graph = True)
        '''optimizer.step()
        optimizer.zero_grad()'''

        optimizer.first_step(zero_grad=True)
        criterion(model(image)[1], labels.unsqueeze(1).float()).backward()
        optimizer.second_step(zero_grad=True)
        
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
    
    for step,(image,labels) in loop:
        image = image.to(device)
        labels = labels.to(device)/100.
        batch_size = labels.size(0)
        with torch.no_grad():
            output1, output2 = model(image)
      
        loss2 = criterion(output2,labels.unsqueeze(1))
        loss = loss2
        output2 = output2.sigmoid()
        out = output2.cpu().detach().numpy()
        targets = labels.cpu().detach().numpy()
        rmse = mean_squared_error(targets*100, out*100, squared=False)
        
        losses.update(loss.item(), batch_size)
        scores.update(rmse.item(), batch_size)
        loop.set_postfix(loss = loss.item(), rmse  = rmse.item(), stage = 'valid')
        
    return losses.avg,scores.avg

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


def fit(m, fn, training_batch_size = config['TRAIN_BATCH_SIZE'], validation_batch_size = config['VAL_BATCH_SIZE']):
    
    
    train_data = df[df.fold != fn]
    train_data = train_data[["path" , "Pawpularity"]]
    plx = pl[["path" ,f'pf{fn}']]
    plx['Pawpularity'] = plx[f'pf{fn}']
    plx = plx[(plx.Pawpularity < 30) | (plx.Pawpularity> 60)]
    train_data = pd.concat([train_data , plx])

    val_data = df[df.fold == fn]

    train_data = Pets(train_data.reset_index(drop=True) , augs = train_aug)
    val_data  = Pets(val_data.reset_index(drop=True) , augs = val_aug)
    
    train_loader = DataLoader(train_data, shuffle=True, pin_memory=True, drop_last=True, batch_size=training_batch_size, num_workers=4)
    valid_loader = DataLoader(val_data, shuffle=False, pin_memory=True, drop_last=False, batch_size=validation_batch_size, num_workers=4)
   
    criterion= nn.BCEWithLogitsLoss()
    #optimizer = optim.AdamW(m.parameters(), lr = config['LR'], weight_decay = config['WEIGHT_DECAY'])
    base_optimizer = optim.AdamW # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(m.parameters(), base_optimizer, lr=config['LR'] , weight_decay =config['WEIGHT_DECAY'])
    
    wandb.watch(model, criterion, log="all", log_freq=10)
    
  
    num_train_steps = math.ceil(len(train_loader))
    num_warmup_steps = num_train_steps * (config['EPOCHS']//2 -1)
    num_training_steps = int(num_train_steps * config['EPOCHS'])
    sch = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps) 
    
    loop = range(config['EPOCHS'])
    for e in loop:
        
        train_loss, train_rmse = train_one_epoch(train_loader, m, optimizer, criterion, e, config['EPOCHS'], sch)
    
        print(f"Avg train loss {train_loss} - Avg train rmse {train_rmse}" )
        
        val_loss,val_rmse= val_one_epoch(valid_loader, m, optimizer, criterion)
        
        model_path =  config['OUTPUT_DIR'] + f"{config['MODEL_NAME']} - {config['PET_CLASS']} - Fold {fold_n} - val rmse {format(val_rmse,'.4f')}.pth"
        
        if e == 0:
            best_rmse = val_rmse
            last_model = model_path
            
        print(f"Avg val loss {val_loss} - Avg val rmse {val_rmse}")
        
        if val_rmse < best_rmse:
            if os.path.exists(last_model):
                os.remove(last_model)
                
            last_model = model_path
            best_rmse = val_rmse
            print("Saving best model!")
            torch.save(m.state_dict(), model_path) 
            
        wandb.log({"Train RMSE": train_rmse, "Val RMSE": val_rmse, "Train loss": train_loss, "Val Loss": val_loss, "Epoch": e})




if __name__ == '__main__':
    
    set_seed(42)

    df = pd.read_csv(f"{config['DATA_DIR']}train.csv")
    y = df.Pawpularity.values
    kf = model_selection.StratifiedKFold(n_splits = config['NUM_FOLDS'], random_state=42, shuffle=True)
    
    for f,(t,v) in enumerate(kf.split(X=df,y=y)):
        df.loc[v,'fold'] = f

    df['path'] = [f"{config['TRAIN_IMAGES_PATH']}{x}.jpg" for x in df["Id"].values]

    pl = pd.read_csv(f"{config['DATA_DIR']}p_labelling.csv")
    pl['path'] = [f"{config['TRAIN_IMAGES_PATH_PL']}{x}-1.jpg" for x in pl["PetID"].values]


    train_aug = A.Compose(
        [   A.RandomResizedCrop(config['IMAGE_SIZE'],config['IMAGE_SIZE'],p = config['CROP']),
            A.Resize(config['IMAGE_SIZE'],config['IMAGE_SIZE'],p = config['RESIZE']),
            A.HorizontalFlip(p = config['H_FLIP']),  
             A.VerticalFlip(p=0.5),   
            A.Transpose(p=0.1), 
            A.RandomBrightnessContrast(p = config['BRIGHT_CONTRAST']),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.Cutout(max_h_size=int(config['IMAGE_SIZE'] * 0.125), max_w_size=int(config['IMAGE_SIZE'] * 0.125), num_holes=5, p=0.5),
              
       A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ]
    )
    val_aug = A.Compose(
        [ 
         A.Resize(config['IMAGE_SIZE'],config['IMAGE_SIZE'],p = config['RESIZE']),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ]
    )

 
    with wandb.init(project="Pawpularity NN", entity = config['WANDB_ENTITY']):
        for i in range(config['START_FOLD'], config['NUM_FOLDS']):
            model = Model(True)
            model= model.to(device)
            fit(model, i)