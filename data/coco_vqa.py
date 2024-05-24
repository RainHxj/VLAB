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

class VQADataset(Dataset):
    def __init__(self, ann_file, image_root, video_cfg, is_train=False, tokenizer="bert_tokenizer", max_txt_len=30):
        self.ann = []

        self.ann = json.load(open(ann_file, 'r'))
        self.sample_num = video_cfg['sample_num']
        self.resolution = video_cfg['resolution']
        self.mean = video_cfg['mean']
        self.std = video_cfg['std']
        self.is_train = is_train
        if is_train:
            self.transforms = transforms.Compose([RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                RandomHorizontalFlip(),
                                                Normalize(self.mean,self.std)])
        else:
            self.transforms = transforms.Compose([Resize(self.resolution),
                                CenterCrop(self.resolution),
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
        question_id = ann.get("question_id","")
        id_txt = id_
        question = ann.get("question", "" )
        output_txt = self.get_single_txt(question, self.txt_encoder_tokenizer, self.txt_encoder_bos, self.txt_encoder_eos)
        output_multi = self.get_single_txt(question, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos)        

        ### question
        if self.is_train:
            ### answer
            answer = ann.get("answer", None)
            if isinstance(answer, (list, tuple)):
                 answer = random.choice(answer)
                 output_answer = self.get_single_txt(answer, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos)
                 
        else:
            answer = ann.get("answer", None)
            if answer is None:
                output_answer=None
            elif isinstance(answer, (list, tuple)):
                 answer = random.choice(answer)
                 output_answer = self.get_single_txt(answer, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos)
            
        return id_, output_multi, output_answer, question, answer, imgs, question_id


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


def txtvideo_collate_coco_vqa(inputs):
    
    (ids , question_tokens, answer_tokens, questions, answers, video_pixels, question_ids) = map(list, unzip(inputs))

    question_tokens = torch.stack(question_tokens,dim = 0)

    # multi_tokens = None
    if answer_tokens[0] is not None:
        answer_tokens = torch.stack(answer_tokens, dim = 0)        

    video_pixels = torch.stack(video_pixels,dim=0)
    batch2m = {'ids': ids,
             'questions': questions, "answers":answers, 'question_tokens': question_tokens, 'answer_tokens':answer_tokens, 'video_pixels': video_pixels,"question_ids": question_ids}
    batch={}
    batch["batch_2m"] = batch2m
    return batch
