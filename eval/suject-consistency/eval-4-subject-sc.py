import os
import cv2
import numpy as np
import json
import torch
import supervision as sv
import sys
import pycocotools.mask as mask_util
from pathlib import Path
current_file_path=os.path.abspath(__file__)
b_folder_path=os.path.dirname(current_file_path)
sys.path.append(b_folder_path)
# import torchvision
from torchvision.ops import box_convert

from grounding_dino.groundingdino.util.inference import load_model, predict
from PIL import Image
from diffusers import FluxPipeline
import json
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import grounding_dino.groundingdino.datasets.transforms as T
from transformers import AutoProcessor, AutoModel
import random
import os
from tqdm import tqdm
import argparse
import glob
import warnings
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import sklearn.preprocessing
from packaging import version
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import pickle
GROUNDING_DINO_CONFIG="./grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT="./ckpt/groundingdino_swint_ogc.pth"
BOX_THRESHOLD=0.35
TEXT_THRESHOLD=0.25
DEVICE="cuda"
# grounding_model=load_model(
#     model_config_path=GROUNDING_DINO_CONFIG,
#     model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
#     device=DEVICE
# )
def Convert(image):
    return image.convert("RGB")
class DINOImageDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data
        self.preprocess = self._transform_test(224)
    def _transform_test(self, n_px):
        return Compose([
            Resize(256,interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    def __getitem__(self,idx):
        image=self.data[idx]
        image=self.preprocess(image)
        return {'image':image}
    def __len__(self):
        return len(self.data)
def load_image_g_pil(image_source):
    transform = T.Compose(
        [
            T.RandomResize([800],max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
    )
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed
def segment_pil_samecategory(grounding_model,img_pil,text_prompt,idx=0,obj_path=None,output_path=None):
    try:
        if output_path == None:
            output_path = "./mask_binaray.png"
        if text_prompt == None:
            text_prompt = input("text:")
        text = text_prompt
        image_source, image = load_image_g_pil(img_pil)
        boxes ,confidences ,labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        h,w,_=image_source.shape
        boxes=boxes*torch.Tensor([w,h,w,h])
        input_boxes=box_convert(boxes=boxes,in_fmt="cxcywh",out_fmt="xyxy").numpy()

        image_np=np.asarray(img_pil)
        result=image_np.copy()
        x1,y1,x2,y2=input_boxes[idx]
        x1=int(x1)
        x2=int(x2)
        y1=int(y1)
        y2=int(y2)
        cropped_image=result[y1:y2,x1:x2]
        final_image=Image.fromarray(cropped_image)
        return final_image
    except:
        return img_pil
        
def segment_pil(grounding_model,img_pil,text_prompt,obj_path=None,output_path=None):
    try:
        if output_path == None:
            output_path = "./mask_binaray.png"
        if text_prompt == None:
            text_prompt = input("text:")
        text = text_prompt
        image_source, image = load_image_g_pil(img_pil)
        boxes ,confidences ,labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        h,w,_=image_source.shape
        boxes=boxes*torch.Tensor([w,h,w,h])
        input_boxes=box_convert(boxes=boxes,in_fmt="cxcywh",out_fmt="xyxy").numpy()

        image_np=np.asarray(img_pil)
        result=image_np.copy()
        x1,y1,x2,y2=input_boxes[0]
        x1=int(x1)
        x2=int(x2)
        y1=int(y1)
        y2=int(y2)
        cropped_image=result[y1:y2,x1:x2]
        final_image=Image.fromarray(cropped_image)
        return final_image
    except:
        return img_pil

class GroundinoScorer(torch.nn.Module):
    def __init__(self, device="cuda",dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True).to(device)
        self.dino_model = self.dino_model.to(dtype=dtype)

        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        )
        self.grounding_model=self.grounding_model.to(self.device)
    def extract_all_images(self, images, model, device, dtype=torch.float32):
        all_image_features = []
        preprocess = Compose([
            Resize(256,interpolation=Image.BICUBIC),
            CenterCrop(224),
            Convert,
            ToTensor(),
            Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        with torch.no_grad():
            for b in images:
                b=preprocess(b)
                b=torch.unsqueeze(b,0)
                b=b.to(device).to(dtype=dtype)
                if hasattr(model,'encode_image'):
                    all_image_features.append(model.encode_image(b))
                else:
                    all_image_features.append(model(b))
        all_image_features=torch.cat(all_image_features,dim=0)
        return all_image_features
    @torch.no_grad()
    def __call__(self, prompt, images, ref_images_1, ref_images_2,ref_images_3,ref_images_4, category1, category2,category3,category4):
        
        images_1 = [segment_pil(self.grounding_model,image,cate1) for image,cate1 in zip(images,category1)]
        images_2 = [segment_pil(self.grounding_model,image,cate2) for image,cate2 in zip(images,category2)]
        images_3 = [segment_pil(self.grounding_model,image,cate3) for image,cate3 in zip(images,category3)]
        images_4 = [segment_pil(self.grounding_model,image,cate4) for image,cate4 in zip(images,category4)]
        
        ref_images_1_crop = [segment_pil(self.grounding_model,image,cate1) for image,cate1 in zip(ref_images_1,category1)]
        ref_images_2_crop = [segment_pil(self.grounding_model,image,cate2) for image,cate2 in zip(ref_images_2,category2)]
        ref_images_3_crop = [segment_pil(self.grounding_model,image,cate3) for image,cate3 in zip(ref_images_3,category3)]
        ref_images_4_crop = [segment_pil(self.grounding_model,image,cate4) for image,cate4 in zip(ref_images_4,category4)]
        
        try:
            images_1_feats = self.extract_all_images(images_1,self.dino_model, self.device, dtype=self.dtype)
        except:
            images_1_feats = self.extract_all_images(images,self.dino_model, self.device, dtype=self.dtype)
        try:
            ref_images_1_feats = self.extract_all_images(ref_images_1_crop,self.dino_model, self.device, dtype=self.dtype)
        except:
            ref_images_1_feats = self.extract_all_images(ref_images_1,self.dino_model, self.device, dtype=self.dtype)
        images_1_feats = images_1_feats / torch.norm(images_1_feats,p=2,dim=1,keepdim=True)
        ref_images_1_feats = ref_images_1_feats / torch.norm(ref_images_1_feats,p=2,dim=1,keepdim=True)
        res1=torch.sum(images_1_feats*ref_images_1_feats,dim=1)
        
        try:
            images_2_feats = self.extract_all_images(images_2,self.dino_model, self.device, dtype=self.dtype)
        except:
            images_2_feats = self.extract_all_images(images,self.dino_model, self.device, dtype=self.dtype)
        try:
            ref_images_2_feats = self.extract_all_images(ref_images_2_crop,self.dino_model, self.device, dtype=self.dtype)
        except:
            ref_images_2_feats = self.extract_all_images(ref_images_2,self.dino_model, self.device, dtype=self.dtype)
        images_2_feats = images_2_feats / torch.norm(images_2_feats,p=2,dim=1,keepdim=True)
        ref_images_2_feats = ref_images_2_feats / torch.norm(ref_images_2_feats,p=2,dim=1,keepdim=True)
        res2=torch.sum(images_2_feats*ref_images_2_feats,dim=1)
        
        try:
            images_3_feats = self.extract_all_images(images_3,self.dino_model, self.device, dtype=self.dtype)
        except:
            images_3_feats = self.extract_all_images(images,self.dino_model, self.device, dtype=self.dtype)
        try:
            ref_images_3_feats = self.extract_all_images(ref_images_3_crop,self.dino_model, self.device, dtype=self.dtype)
        except:
            ref_images_3_feats = self.extract_all_images(ref_images_3,self.dino_model, self.device, dtype=self.dtype)
        images_3_feats = images_3_feats / torch.norm(images_3_feats,p=2,dim=1,keepdim=True)
        ref_images_3_feats = ref_images_3_feats / torch.norm(ref_images_3_feats,p=2,dim=1,keepdim=True)
        res3=torch.sum(images_3_feats*ref_images_3_feats,dim=1)
        
        try:
            images_4_feats = self.extract_all_images(images_4,self.dino_model, self.device, dtype=self.dtype)
        except:
            images_4_feats = self.extract_all_images(images,self.dino_model, self.device, dtype=self.dtype)
        try:
            ref_images_4_feats = self.extract_all_images(ref_images_4_crop,self.dino_model, self.device, dtype=self.dtype)
        except:
            ref_images_4_feats = self.extract_all_images(ref_images_4,self.dino_model, self.device, dtype=self.dtype)
        images_4_feats = images_4_feats / torch.norm(images_4_feats,p=2,dim=1,keepdim=True)
        ref_images_4_feats = ref_images_4_feats / torch.norm(ref_images_4_feats,p=2,dim=1,keepdim=True)
        res4=torch.sum(images_4_feats*ref_images_4_feats,dim=1)
        
        scores=(res1+res2+res3+res4)/4
        return scores.cpu().tolist()
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scorer = GroundinoScorer(
        device=device,
        dtype=torch.float32
    )
    import json
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the input JSON file, e.g., attr.json")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory for data loading")
    args = parser.parse_args()


    with open(args.json_path,"r",encoding="utf-8") as f:
        data=json.load(f)
    results=[]
    data_root=args.data_root
    for idx,item in enumerate(data):
        try:
            ref1=[Image.open(os.path.join(data_root,item["input_path"][0]))]
            ref2=[Image.open(os.path.join(data_root,item["input_path"][1]))]
            ref3=[Image.open(os.path.join(data_root,item["input_path"][2]))]
            ref4=[Image.open(os.path.join(data_root,item["input_path"][3]))]
            prompts=[item["instruction"]]
            images=[item["result_path"]]
            pil_images=[Image.open(img) for img in images]
            category1=[item["category1"]]
            category2=[item["category2"]]
            category3=[item["category3"]]
            category4=[item["category4"]]
            score=scorer(prompts,pil_images,ref1,ref2,ref3,ref4,category1,category2,category3,category4)
            results.extend(score)
        except:
            print("{} don't exist".format(idx))
    print("final score: ",sum(results)/len(results))
        
       