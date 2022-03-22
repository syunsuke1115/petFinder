import streamlit as st
import os
import gc
import sys
import math
import time
import random
import shutil
import seaborn as sns
import pickle
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import joblib
import scipy as sp
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from tqdm.auto import tqdm
from functools import partial
import cv2
from PIL import Image
from IPython.core.display import display
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torchvision import models, transforms
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import timm
import lightgbm as lgb
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CnnModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.fc = nn.Linear(self.n_features, self.cfg.target_size)

    def feature(self, image):
        feature = self.model(image)
        return feature
        
    def forward(self, image):
        feature = self.feature(image)
        output = self.fc(feature)
        return output

class CFG:
    apex=False
    debug=False
    print_freq=10
    num_workers=4
    size=224 ##モデルによって変える。
    model_name='tf_efficientnet_b0_ns' ##モデルによって変える
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs=2
    factor=0.2 # ReduceLROnPlateau
    patience=4 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    T_max=3 # CosineAnnealingLR
    T_0=3 # CosineAnnealingWarmRestarts
    lr=3e-4
    min_lr=1e-6
    batch_size=16
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    target_size=1
    target_col='Pawpularity'
    n_fold=2
    trn_fold=[0, 1]
    train=True
    grad_cam=True

#seed値を固定
def set_seed(seed =42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
set_seed(seed=CFG.seed) 

def input_image(uploaded_file):
    image = Image.open(uploaded_file)
    transform = transforms.Compose([
            transforms.Resize((CFG.size,CFG.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            
        ])
    image = transform(image)
    img_batch = image[None]
    return img_batch

###main
st.header("あなたのペットの可愛さを判定します")

uploaded_file=st.file_uploader("画像アップロード", type='png')

# If button is pressed
if st.button("Submit"):
    img_batch = input_image(uploaded_file)
    model = CnnModel(CFG, pretrained=False)
    state = torch.load('../my_model/tf_efficientnet_b0_ns_fold0_best.pth', 
                       map_location=torch.device('cpu'))['model']
    model.load_state_dict(state)
    model.eval()
    relust= model(img_batch)
    # Output prediction
    st.text(f"{relust[0][0]}")
    

