import streamlit as st
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import timm
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
    batch_size=16
    seed=42
    target_size=1
    target_col='Pawpularity'

#seed値を固定
def set_seed(seed =42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
set_seed(seed=CFG.seed) 

def set_image(image):
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

def get_score(result):
    pawpularity_mean = 38.03904358353511
    pawpularity_std = 20.591990105774546
    deviation =((result-pawpularity_mean)/pawpularity_std*10 +50).round()
    result = result.round()
    return result, deviation
    
def main():
    st.header("あなたのペットの可愛さを採点します")
    st.text("※kaggleコンペティションpetfinderのデータを使用しています")
    uploaded_file=st.file_uploader("画像アップロード")
    
    if st.button("Submit"):
        if uploaded_file is None:
            st.error('画像を提出してください')
            return
        image = Image.open(uploaded_file)
        img_batch = set_image(image)
        model = CnnModel(CFG, pretrained=False)
        state = torch.load('./tf_efficientnet_b0_ns_fold0_best.pth', 
                        map_location=torch.device('cpu'))['model']
        model.load_state_dict(state)
        model.eval()
        result= model(img_batch)
        result, deviation = get_score(result[0][0])
        
        st.image(image,width= 500)
        st.success(f"得点: {int(result)}")
        st.info(f"偏差値: {int(deviation)}")
        
if __name__ == '__main__':
    main()