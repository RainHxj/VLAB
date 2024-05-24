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

t5_emb_path_map = {
    "hdfs://haruna/home/byte_labcv_gan/public/multimodal_all/tucong_aml_filtered_kv": "hdfs://haruna/home/byte_labcv_gan/public/multimodal_all/tucong_aml_filtered_T5_emb"
}

meta_path_map = {
    "hdfs://haruna/home/byte_uslab_cvg_lq/data/multimodal_generation/laion5b_filtered": "hdfs://haruna/home/byte_uslab_cvg_lq/data/multimodal_generation/laion5b_filtered_meta"
}

class DistLineReadingDataset(IterableDataset): 
    """
    iterate a number of data items split.
    """

    def __init__(self,
                 data_path: str,
                 ids_path: str,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = False,
                 repeat: bool = False,
                 with_t5_emb: bool = False,
                 with_meta: bool = False,
                 ):
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.with_t5_emb = with_t5_emb
        self.with_meta = with_meta
        
        if ids_path:
            self.files = json.load(open(ids_path))
        
        else:

            self.files = hlist_files( data_path.split(',') )

            if len(self.files) > 0:
                self.files = [f for f in self.files if f.find('_SUCCESS') < 0]
                self.files = [file[:-6] for file in self.files if ('.index' in file or '.snappy' in file) ]
            else:
                parent_folder = '/'.join(data_path.split('/')[:-1])
                self.files = hlist_files(parent_folder.split(','))
                self.files = [file[:-6] for file in self.files if ('.index' in file or '.snappy' in file) and file.startswith(data_path) ]
                LOGGER.info('[DATA] attempting to use hdfs files prefixed with %s' % (data_path))
        
        self.is_hdfs = True#data_path.startswith('hdfs')
        self.repeat = repeat

        if rank == 0:
            LOGGER.info(f'[DATA]--all dataset containing {len(self.files)} files.')

        if len(self.files) % self.world_size != 0:
            LOGGER.info('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))

        #if self.shuffle:
        #    random.seed(888)
        #    random.shuffle(self.files)
        #    random.seed()
            
    def generate(self, is_decode=True):
        """
        这个函数一开始做了两次split_shard，对文件进行划分。
        第一次split_shard是基于rank和world_size来分的，是gpu节点维度的；
        第二次split_shard是基于worker_info的，是一个gpu节点的dataloader内，给不同的worker划分文件。
        """
        if self.world_size == 1 or len(self.files) == 1 or len(self.files) < self.world_size:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = split_shard(
                self.files, self.rank, self.world_size)

        #print('rank and world size: ', self.rank, self.world_size , len(cur_dataloader_files))

        while True:
            #if self.shuffle:
            #    random.shuffle(cur_dataloader_files)
            worker_info = torch.utils.data.get_worker_info()

            if worker_info is not None:
                if len(cur_dataloader_files) % worker_info.num_workers != 0:
                    LOGGER.info('[DATA]--current dataloader %s file num %s cannot split to worker_num %s ' %
                             (self.rank, len(cur_dataloader_files), worker_info.num_workers))

                if worker_info.num_workers == 1 or len(cur_dataloader_files) == 1 or len(cur_dataloader_files) < worker_info.num_workers:
                    cur_worker_files =  cur_dataloader_files
                else:
                    cur_worker_files = split_shard(
                        cur_dataloader_files, worker_info.id, worker_info.num_workers)
                    #print(len(cur_worker_files), worker_info.id, worker_info.num_workers)

                if worker_info.id == 0:
                    LOGGER.info("[DataLoader] --> Rank:{}  Workers:[{} ~ {}]  Size of process file:{}  ...".format(
                                self.rank, 0, worker_info.num_workers - 1, len(cur_dataloader_files)))
            else:
                # when num_worker=0
                cur_worker_files = cur_dataloader_files

            #print('begin shuffle: ', worker_info.id, len(cur_worker_files))
            if self.shuffle:
                random.shuffle(cur_worker_files)
            #print('shuffled: ', worker_info.id, len(cur_worker_files))
            
            for filepath in cur_worker_files:
                if self.is_hdfs:
                    #print("begin loading data ...")
                    if hexists(filepath+'.index'):
                        t5_root = t5_emb_path_map.get(os.path.dirname(filepath), "")
                        t5_path = os.path.join(t5_root, os.path.basename(filepath))
                        
                        meta_root = meta_path_map.get(os.path.dirname(filepath), "")
                        meta_path = os.path.join(meta_root, os.path.basename(filepath))
                        data_reader_meta = None
                        if self.with_meta and meta_root != "" and hexists(meta_path+'.index'):
                            data_reader_meta = KVReader(meta_path, 8)

                        if self.with_t5_emb and t5_root != "" and hexists(t5_path+'.index'):
                            data_reader = KVReader(filepath, 8)
                            data_reader_t5 = KVReader(t5_path, 8)
                            keys = data_reader_t5.list_keys()

                            if self.shuffle:
                                random.shuffle(keys)
                            for key_idx in range(0, len(keys), 100):
                                cur_keys = keys[key_idx:key_idx+100]
                                values = data_reader.read_many(cur_keys)
                                emb_values = data_reader_t5.read_many(cur_keys)
                                meta_values = [None]*len(values)
                                if data_reader_meta!=None:
                                    meta_values = data_reader_meta.read_many(cur_keys)
                                for value, emb_value, meta_value in zip(values, emb_values, meta_values):
                                    if meta_value != None:
                                        meta_value = meta_value.decode()
                                    
                                    if is_decode:
                                        yield value.decode(), os.path.basename(filepath), np.frombuffer(emb_value, dtype=np.float32).reshape(-1, 1024), meta_value
                                    else:
                                        yield value, os.path.basename(filepath), np.frombuffer(emb_value, dtype=np.float32).reshape(-1, 1024), meta_value

                        else:
                            
                            data_reader = KVReader(filepath, 8)
                            keys = data_reader.list_keys() if data_reader_meta==None else data_reader_meta.list_keys()

                            if self.shuffle:
                                random.shuffle(keys)

                            for key_idx in range(0, len(keys), 100):
                                cur_keys = keys[key_idx:key_idx+100]
                                try:
                                    values = data_reader.read_many(cur_keys)
                                except:
                                    continue
                                meta_values = [None]*len(values)
                                if data_reader_meta!=None:
                                    meta_values = data_reader_meta.read_many(cur_keys)

                                for value, meta_value in zip(values, meta_values):
                                    #print("return one value")
                                    if meta_value != None:
                                        meta_value = meta_value.decode()
                                    if is_decode:
                                        yield value.decode(), os.path.basename(filepath), meta_value
                                    else:
                                        yield value, os.path.basename(filepath), meta_value

                    elif hexists(filepath+'snappy'):
                        with hopen(filepath+'snappy', 'r') as reader:
                            for line in reader:
                                if is_decode:
                                    yield line.decode(), filepath.split('/')[-2] + "_" + filepath.split('/')[-1]
                                else:
                                    yield line, filepath.split('/')[-2] + "_" + filepath.split('/')[-1]
                        continue
                    else:
                        LOGGER.info("invalid hdfs path: %s"%filepath)
                        continue 

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
    # print('spliting shards: ', num, start_idx, end_idx, shard_idx, shard_size)
    return data[start_idx: end_idx]

class LaionDataset(DistLineReadingDataset):

    def __init__(self,
                 data_path: str,
                 ids_path: str,
                 video_cfg: dict,
                 caption: str,
                 is_train: bool,
                 tokenizer = None,
                 rank: int = 0,
                 world_size: int = 1
                 ):
        super().__init__(data_path,ids_path, rank, world_size)

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

        self.caption = caption

    def __iter__(self,):
        for data in self.generate():
            ### image
            data_item=json.loads(data[0])
            image_str = b64decode(data_item.get("b64_resized_binary", data_item.get("binary", data_item.get("b64_binary", data_item.get("image","")))))
            image_pil = Image.open(io.BytesIO(image_str)).convert("RGB")
            image_pil = transforms.ToTensor()(image_pil)
            image_pil=image_pil.unsqueeze(0)
            image_list=[]
            assert image_pil.shape[1]==3, "image dim is not 3"
            for i in range(1):
                image_list.append(image_pil)
            imgs = torch.cat(image_list,dim=0)   ### nX3xHxW
            imgs = self.transforms(imgs)

            ### text
            text = data_item.get(self.caption, "")
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
    data_path="hdfs://haruna/home/byte_ailab_litg/user/zengyan/vlm/images/coco_testset_filtered, \
               hdfs://haruna/home/byte_ailab_litg/user/zengyan/vlm/images/vg_testset_filtered, \
               hdfs://haruna/home/byte_ailab_litg/user/zhangxinsong/datasets/vlm/sbu_bs64, \
               hdfs://haruna/home/byte_ailab_litg/user/zhangxinsong/datasets/vlm/cc3m_bs64"

    train_data = LaionDataset(data_path, video_cfg, is_train=True)
    
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
         # iterate through your training data (batch['batch_2m']['video_pixels']).shape
        n+=1        
    print(n)