#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-16 15:25:34
LastEditTime: 2020-11-18 18:12:03
LastEditors: Huang Wenguan
Description: hdfs dataset
'''
import os
import torch
from torch.utils.data import IterableDataset, Dataset
import random
from typing import List, Any
import json

from dataloader import KVReader
from utils.hdfs_io import hopen, hlist_files, hexists, hcopy
from tqdm import tqdm 
import pickle 
import os
import numpy as np
import json
import io
from PIL import Image
from torchvision.transforms.transforms import *
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from toolz.sandbox import unzip
import time
from utils.logger import LOGGER
from base64 import b64decode
from torchvision import transforms
from .tokenizer import build_tokenizer
import string
import ipdb

class RetDataset(Dataset):
    def __init__(self, ann_file, image_root, video_cfg, is_train=False, tokenizer="bert_tokenizer", max_txt_len=30):
        self.ann = []

        self.ann = json.load(open(ann_file, 'r'))
        self.sample_num = video_cfg['sample_num']
        self.resolution = video_cfg['resolution']
        self.mean = video_cfg['mean']
        self.std = video_cfg['std']
        self.is_train = is_train
        if is_train:
            self.transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                Normalize(self.mean,self.std)])
        else:
            self.transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                # CenterCrop(self.resolution),
                                Normalize(self.mean,self.std)])
        
        self.image_root = image_root
        self.tokenizer_info = build_tokenizer(tokenizer)
        self.txt_encoder_tokenizer = self.tokenizer_info["txt_encoder"]["tokenizer"]
        self.txt_encoder_bos = self.tokenizer_info["txt_encoder"]["bos"]
        self.txt_encoder_eos = self.tokenizer_info["txt_encoder"]["eos"]

        self.multi_encoder_tokenizer = self.tokenizer_info["multi_encoder"]["tokenizer"]
        self.multi_encoder_bos = self.tokenizer_info["multi_encoder"]["bos"]
        self.multi_encoder_eos = self.tokenizer_info["multi_encoder"]["eos"]

        self.max_txt_len = max_txt_len
        self.punctuations = string.punctuation

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        image =image.unsqueeze(0)
        image_list=[]
        assert image.shape[1]==3, "image dim is not 3"
        for i in range(1):
            image_list.append(image)
        imgs = torch.cat(image_list,dim=0)   ### nX3xHxW
        imgs = self.transforms(imgs)

        id_ = os.path.splitext(ann["image"].split("/")[1])[0]
        ### text
        text = ann.get("caption", "" )
        if isinstance(text, list):

            if self.is_train:
                id_txt = [id_]
                rand_text = random.choice(text)
                output_txt = self.get_single_txt(rand_text, self.txt_encoder_tokenizer, self.txt_encoder_bos, self.txt_encoder_eos)
                output_multi = self.get_single_txt(rand_text, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos)
            else:
                id_txt = [id_]*len(text)
                output_txt=[]
                output_multi=[]
                for tt in text:
                    output_txt.append(self.get_single_txt(tt, self.txt_encoder_tokenizer, self.txt_encoder_bos, self.txt_encoder_eos))
                    output_multi.append(self.get_single_txt(tt, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos))
        else:
            id_txt = [id_]
            output_txt = self.get_single_txt(text, self.txt_encoder_tokenizer, self.txt_encoder_bos, self.txt_encoder_eos)
            output_multi = self.get_single_txt(text, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos)
        
        return id_, output_txt, output_multi, imgs, id_txt

    def get_single_txt(self,text, tokenizer, bos, eos):
        text = self.clean(text) 

        txt_tokens = tokenizer(text)

        txt_tokens = txt_tokens[:self.max_txt_len]

        txt_tokens = [bos] + txt_tokens + [eos] 
        
        txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)

        output = torch.zeros(self.max_txt_len + 2, dtype=torch.long)
        
        output[:len(txt_tokens)] = txt_tokens
        return output

    def clean(self, text):
        """remove duplicate spaces, lower and remove punctuations """
        text = ' '.join([i for i in text.split(' ') if i != ''])
        text = text.lower()
        for i in self.punctuations:
            text = text.replace(i,'')
        return text        


def txtvideo_collate_coco(inputs):
    
    (ids , txt_tokens, multi_tokens ,video_pixels,id_txts) = map(list, unzip(inputs))
    id_txts = [j for i in id_txts for j in i]
    if isinstance(txt_tokens[0],list):
        txt_tokens = [ j  for i in txt_tokens for j in i]
        txt_tokens = torch.stack(txt_tokens,dim = 0)
    else:
        txt_tokens = torch.stack(txt_tokens,dim = 0)

    # multi_tokens = None
    if isinstance(multi_tokens[0],list):
        multi_tokens = [ j  for i in multi_tokens for j in i]
        multi_tokens = torch.stack(multi_tokens,dim = 0)
    else:
        multi_tokens = torch.stack(multi_tokens,dim = 0)        

    video_pixels = torch.stack(video_pixels,dim=0)
    batch2m = {'ids': ids,
             'txt_tokens': txt_tokens, "multi_tokens":multi_tokens, 'video_pixels': video_pixels, 'ids_txt':id_txts}
    batch={}
    batch["batch_2m"] = batch2m
    return batch