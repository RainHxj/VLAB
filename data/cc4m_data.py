#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-16 15:25:34
LastEditTime: 2020-11-18 18:12:03
LastEditors: Huang Wenguan
Description: hdfs dataset
'''

import torch
from torch.utils.data import IterableDataset
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

import sys
from typing import List, Any
import warnings
import random
from itertools import cycle
import torch
from torch.utils.data import IterableDataset

from utils.hdfs_io import hopen, hlist_files, hopen


class DistLineReadingDatasetJson(IterableDataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """
    def __init__(self,
                 data_path: list,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = False,
                 repeat: bool = False):
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        self.files = hlist_files(data_path)
        self.files = [f for f in self.files if f.find('_SUCCESS') < 0]
        self.is_hdfs = data_path[0].startswith('hdfs')

        self.repeat = repeat
        print('[DATA]--all dataset containing {} files.'.format(len(self.files)))
        if len(self.files) % self.world_size != 0:
            print('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        sys.stdout.flush()

    def generate(self):
        if self.world_size == 1 or len(self.files) == 1:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = split_shard(
                self.files, self.rank, self.world_size)

        while True:
            if self.shuffle:
                random.shuffle(cur_dataloader_files)
            worker_info = torch.utils.data.get_worker_info()

            if worker_info is not None:
                if len(cur_dataloader_files) % worker_info.num_workers != 0:
                    print('[DATA]--current dataloader %s file num %s cannot split to worker_num %s ' %
                             (self.rank, len(cur_dataloader_files), worker_info.num_workers))
                cur_worker_files = split_shard(
                    cur_dataloader_files, worker_info.id, worker_info.num_workers)
                if worker_info.id == 0:
                    print("[DataLoader] --> Rank:{}  Workers:[{} ~ {}][{}]  Size of process file:{}  ...".format(
                        self.rank, 0, worker_info.num_workers - 1, worker_info.id, len(cur_dataloader_files)))
            else:
                cur_worker_files = cur_dataloader_files

            if self.shuffle:
                random.shuffle(cur_worker_files)
            for filepath in cur_worker_files:
                if self.is_hdfs:
                    with hopen(filepath, 'r') as reader:
                        for line in reader:
                            yield line.decode()
                    continue
                with open(filepath, 'r') as reader:
                    for line in reader:
                        yield line

            if not self.repeat:
                break

    def __iter__(self):
        return self.generate()  


def split_shard(data: List[Any], shard_idx: int, shard_size: int):
    num = len(data)
    if num < shard_size:
        raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
    start_idx = (num * shard_idx) // shard_size
    end_idx = (num * (shard_idx + 1)) // shard_size
    return data[start_idx: end_idx]


class CC4MDataset(DistLineReadingDatasetJson):

    def __init__(self,
                 data_path: list,
                 video_cfg: dict,
                 is_train: bool,
                 tokenizer = None,
                 rank: int = 0,
                 world_size: int = 1
                 ):
        super().__init__(data_path, rank, world_size)

        self.sample_num = video_cfg['sample_num']
        self.resolution = video_cfg['resolution']
        self.mean = video_cfg['mean']
        self.std = video_cfg['std']
        self.punctuations = string.punctuation
        self.max_txt_len=30

        if is_train:
            self.transforms = transforms.Compose([RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                RandomHorizontalFlip(),
                                                Normalize(self.mean,self.std)])
        else:
            self.transforms = transforms.Compose([Resize(self.resolution),
                                CenterCrop(self.resolution),
                                Normalize(self.mean,self.std)])
        
        self.tokenizer_info = build_tokenizer(tokenizer)
        
        self.txt_encoder_tokenizer = self.tokenizer_info["txt_encoder"]["tokenizer"]
        self.txt_encoder_bos = self.tokenizer_info["txt_encoder"]["bos"]
        self.txt_encoder_eos = self.tokenizer_info["txt_encoder"]["eos"]

        self.multi_encoder_tokenizer = self.tokenizer_info["multi_encoder"]["tokenizer"]
        self.multi_encoder_bos = self.tokenizer_info["multi_encoder"]["bos"]
        self.multi_encoder_eos = self.tokenizer_info["multi_encoder"]["eos"]
 
    def __iter__(self,):
        for data in self.generate():
            ### image
            data_item=json.loads(data)
            image_pil = Image.open(io.BytesIO(b64decode(data_item["binary"]))).convert("RGB")
            image_pil = transforms.ToTensor()(image_pil)
            image_pil=image_pil.unsqueeze(0)
            image_list=[]
            assert image_pil.shape[1]==3, "image dim is not 3"
            for i in range(1):
                image_list.append(image_pil)
            imgs = torch.cat(image_list,dim=0)   ### nX3xHxW
            imgs = self.transforms(imgs)

            ### text
            text = data_item.get("desc", "" )
            output_txt = self.get_single_txt(text, self.txt_encoder_tokenizer, self.txt_encoder_bos, self.txt_encoder_eos)
            output_multi = self.get_single_txt(text, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos)
            yield output_txt, output_multi, imgs

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


def dist_data_collate(inputs):
    
    (txt_tokens, multi_tokens ,video_pixels) = map(list, unzip(inputs))
    txt_tokens = torch.stack(txt_tokens,dim = 0)
    multi_tokens = torch.stack(multi_tokens,dim = 0)
    video_pixels = torch.stack(video_pixels,dim=0)
    batch2m = {
             'txt_tokens': txt_tokens, "multi_tokens":multi_tokens, 'video_pixels': video_pixels}
    batch={}
    batch["batch_2m"] = batch2m
    return batch


import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.distributed import DistributedSampler_wopadding

# from data.sampler import DistributedSampler
if __name__ == "__main__":
    

    video_cfg={"sample_num":3,"resolution":224,"mean":3,"std":4}
    num_readers=32
    data_path=["hdfs://haruna/home/byte_ailab_litg/user/zengyan/vlm/images/coco_testset_filtered",]
    train_data = CC4MDataset(data_path, video_cfg, tokenizer={"txt_encoder":"bert_tokenizer","multi_encoder":"bert_torkenizer"}, is_train=True)
    
    # train_data = KVDataset(video_path,32)#
    # ipdb.set_trace()
    # local_rank = 0
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl') 

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # print(dist.get_rank())
    batch_size = 32
    # sampler = DistributedSampler_wopadding(train_data)
    # sampler = KVSampler(train_data, batch_size=batch_size, num_replicas=2, rank=1, shuffle=True)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=1000,
                            drop_last=False,
                            num_workers=2,
                            pin_memory=False, collate_fn=dist_data_collate)
    n=0
    from tqdm import tqdm
    for batch in tqdm(dataloader):
        ipdb.set_trace()
         # iterate through your training data (batch['batch_2m']['video_pixels']).shape
        n+=1        
    print(n)