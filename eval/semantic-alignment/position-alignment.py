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
def determine_position(locality,box1,box2,iou_threshold=0.1,distance_threshold=150):
    try:
        box1_center = ((box1["x_min"]+box1["x_max"])/2,(box1["y_min"]+box1["y_max"])/2)
        box2_center = ((box2["x_min"]+box2["x_max"])/2,(box2["y_min"]+box2["y_max"])/2)
        
        x_distance = box2_center[0] - box1_center[0]
        y_distance = box2_center[1] - box1_center[1]
        
        ##calculate iou
        x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
        y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
        intersection = x_overlap * y_overlap
        box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
        box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
        union = box1_area + box2_area - intersection
        iou = intersection / union

        # Determine position based on distances and IoU and give a soft score
        score=0
        if locality in ['next to', 'on side of', 'near']:
            if (abs(x_distance)< distance_threshold or abs(y_distance)< distance_threshold):
                score=1
            else:
                score=distance_threshold/max(abs(x_distance),abs(y_distance))
        elif locality == 'on the right of' or locality == 'to the right of':
            if x_distance < 0:
                if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                    score=1
                elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                    score=iou_threshold/iou
            else:
                score=0
        elif locality == 'on the left of' or locality == 'to the left of':
            if x_distance > 0:
                if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                    score=1
                elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                    score=iou_threshold/iou
            else:
                score=0
        elif locality =='under' or locality =='below':
            if y_distance < 0:
                if iou < iou_threshold:
                    score=1
                elif iou >= iou_threshold:
                    score=iou_threshold/iou
        elif locality =='on top of' or locality =='above':
            if y_distance > 0:
                if iou < iou_threshold:
                    score=1
                elif iou >= iou_threshold:
                    score=iou_threshold/iou
        else:
            score=0
        return score
    except:
        return 0

        
        
        
        
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
        img_pil
        
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
        box={}
        box['x_min']=x1
        box['y_min']=y1
        box['x_max']=x2
        box['y_max']=y2
        return box
    except:
        return None

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
    def __call__(self, prompt, images, ref_images_1, ref_images_2, category1, category2):
        vocab_spatial = ["to the left of","to the right of","on the left of","on the right of","above","on top of","below","under"]
        locality=[]
        for p in prompt:
            for word in vocab_spatial:
                if word in p:
                    locality.append(word)
                    break
        box_1=[segment_pil(self.grounding_model,image,cate1) for image,cate1 in zip(images,category1)]
        box_2=[segment_pil(self.grounding_model,image,cate2) for image,cate2 in zip(images,category2)]
        scores=[determine_position(l,b1,b2) for l,b1,b2 in zip(locality,box_1,box_2)]
        return scores
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scorer = GroundinoScorer(
        device=device,
        dtype=torch.float32
    )
    import json
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the input JSON file, e.g., attr.json")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory for data loading")
    args = parser.parse_args()
    with open(args.json_path,"r",encoding="utf-8") as f:
        data=json.load(f)
    results=[]

    for idx,item in enumerate(data):
        try:
            ref1=[Image.open(os.path.join(args.data_root,item["input_path"][0]))]
            ref2=[Image.open(os.path.join(args.data_root,item["input_path"][1]))]
            prompts=[item["instruction"]]
            images=[item["result_path"]]
            pil_images=[Image.open(img) for img in images]
            category1=[item["category1"]]
            category2=[item["category2"]]
            score=scorer(prompts,pil_images,ref1,ref2,category1,category2)
            results.extend(score)
        except:
            print("{} don't exist".format(idx))
    print("final score: ",sum(results)/len(results))
        
       