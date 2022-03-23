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

class CFG: #モデルによって変える。
    size=224 
    model_name='tf_efficientnet_b0_ns' 
    seed=42
    target_size=1
    model_path = './tf_efficientnet_b0_ns_fold0_best.pth'
    isTransformer =False
    
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

class TransformerModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.model   = timm.create_model(self.cfg.model_name, pretrained=pretrained, num_classes=0, in_chans=3)
        num_features = self.model.num_features
        self.linear  = nn.Linear(num_features, 1)
        
    def feature(self, image):
        feature = self.model(image)
        return feature
    
    def forward(self, x):
        x = self.model(x)
        output = self.linear(x)
        return output

def set_seed(seed =42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True

def set_image(uploaded_file):
    image = Image.open(uploaded_file)
    transform = transforms.Compose([
            transforms.Resize((CFG.size,CFG.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            
        ])
    image_tensor = transform(image)
    img_batch = image_tensor[None]
    return image,img_batch

def set_model(CFG):
    if (CFG.isTransformer):
        model = TransformerModel(CFG, pretrained=False)
    else:
        model = CnnModel(CFG, pretrained=False)
    state = torch.load(CFG.model_path, 
                        map_location=torch.device('cpu'))['model']
    model.load_state_dict(state)
    model.eval()
    return model

def get_score(result):
    #testデータより算出。
    #デプロイにデータベースを載せたくなかったので、別ファイルで計算した値をベタ打ち
    pawpularity_mean = 38.03904358353511
    pawpularity_std = 20.591990105774546
    
    deviation =((result-pawpularity_mean)/pawpularity_std*10 +50).round()
    result = result.round()
    return result, deviation

#インターフェース
def main():
    set_seed(seed=CFG.seed) 
    st.header("あなたのペットの可愛さを採点します")
    st.text("※kaggleコンペティションpetfinderのデータを使用しています")
    uploaded_file=st.file_uploader("画像アップロード")
    
    if st.button("Submit"):
        if uploaded_file is None:
            st.error('画像を提出してください')
            return
        
        image,img_batch = set_image(uploaded_file)
        model = set_model(CFG)
        
        result= model(img_batch)
        result, deviation = get_score(result[0][0])
        
        st.image(image,width= 500)
        st.success(f"得点: {int(result)}")
        st.info(f"偏差値: {int(deviation)}")
        
if __name__ == '__main__':
    main()