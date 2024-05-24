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
from hdfs_io import hopen, hlist_files, hexists, hcopy

from tqdm import tqdm 
import pickle 
import os
import numpy as np


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import logging as python_logging



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

        self.files = hlist_files( data_path.split(',') )

        if len(self.files) > 0:
            self.files = [f for f in self.files if f.find('_SUCCESS') < 0]
            self.files = [file[:-6] for file in self.files if ('.index' in file or '.snappy' in file) ]
        else:
            parent_folder = '/'.join(data_path.split('/')[:-1])
            self.files = hlist_files(parent_folder.split(','))
            self.files = [file[:-6] for file in self.files if ('.index' in file or '.snappy' in file) and file.startswith(data_path) ]
            print('[DATA] attempting to use hdfs files prefixed with %s' % (data_path))
        
        self.is_hdfs = data_path.startswith('hdfs')
        self.repeat = repeat

        if rank == 0:
            print(f'[DATA]--all dataset containing {len(self.files)} files.')

        if len(self.files) % self.world_size != 0:
            print('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))

        #if self.shuffle:
        #    random.seed(888)
        #    random.shuffle(self.files)
        #    random.seed()
            
    def generate(self):
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
                    print('[DATA]--current dataloader %s file num %s cannot split to worker_num %s ' %
                             (self.rank, len(cur_dataloader_files), worker_info.num_workers))

                if worker_info.num_workers == 1 or len(cur_dataloader_files) == 1 or len(cur_dataloader_files) < worker_info.num_workers:
                    cur_worker_files =  cur_dataloader_files
                else:
                    cur_worker_files = split_shard(
                        cur_dataloader_files, worker_info.id, worker_info.num_workers)
                    #print(len(cur_worker_files), worker_info.id, worker_info.num_workers)

                if worker_info.id == 0:
                    print("[DataLoader] --> Rank:{}  Workers:[{} ~ {}]  Size of process file:{}  ...".format(
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
                                    yield value.decode(), os.path.basename(filepath), np.frombuffer(emb_value, dtype=np.float32).reshape(-1, 1024), meta_value

                        else:
                            data_reader = KVReader(filepath, 8)
                            keys = data_reader.list_keys() if data_reader_meta==None else data_reader_meta.list_keys()

                            if self.shuffle:
                                random.shuffle(keys)

                            for key_idx in range(0, len(keys), 100):
                                cur_keys = keys[key_idx:key_idx+100]
                                values = data_reader.read_many(cur_keys)
                                meta_values = [None]*len(values)
                                if data_reader_meta!=None:
                                    meta_values = data_reader_meta.read_many(cur_keys)

                                for value, meta_value in zip(values, meta_values):
                                    #print("return one value")
                                    if meta_value != None:
                                        meta_value = meta_value.decode()
                                    yield value.decode(), os.path.basename(filepath), meta_value

                    elif hexists(filepath+'snappy'):
                        with hopen(filepath+'snappy', 'r') as reader:
                            for line in reader:
                                yield line.decode(), filepath.split('/')[-2] + "_" + filepath.split('/')[-1]
                        continue
                    else:
                        print("invalid hdfs path: %s"%filepath)
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

import ipdb
if __name__ == "__main__":
    
    data_path="hdfs://haruna/home/byte_labcv_gan/public/multimodal_all/laion400m_blip_split"

    data=DistLineReadingDataset(data_path)
    xx = []
    n=0
    for i in iter(data):
        xx.append(i)
        n+=1
        if n>100:
            break
    ipdb.set_trace()