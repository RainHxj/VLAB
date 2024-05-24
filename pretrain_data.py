import argparse
from collections import defaultdict
import json
import math
import os
from os.path import exists, join
from time import time, sleep

import torch
from torch.utils.data import DataLoader, ConcatDataset, ChainDataset
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from data.webvid_data import WebvidFrameDataset
from tqdm import tqdm
import random
from utils.distributed import all_gather_list
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file

from data import  PrefetchLoader
from data.data import TxtVideoDataset, txtvideo_collate, VideoMapper, TxtMapper
import ipdb
from utils.distributed import DistributedSampler_wopadding
import torch.distributed as dist
from data.data import worker_init_fn
from data.dist_data import LaionDataset, dist_data_collate
from utils.distributed import get_rank, get_local_rank
import torch.distributed as dist
from data.cc4m_data import CC4MDataset
from data.webvid10m import Webvid10M_IterableDataset
from data.coyo_webdata import CoyoWebDataset
import ipdb


def build_dataloader(dataset, collate_fn, is_train, opts, batch_size, n_workers,sampler="dist"):
    if sampler=="dist":
        if isinstance(dataset, torch.utils.data.IterableDataset):
            sampler = None
        else:
            sampler = DistributedSampler_wopadding(dataset)


        # sampler = DistributedSampler_wopadding(dataset)
        loader = DataLoader(dataset, sampler = sampler, batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train, worker_init_fn=worker_init_fn)
    else:
        loader = DataLoader(dataset, batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train)    

    return loader

def compute_steps(epoch, batch_size, data_len):
    
    gpus = int(os.getenv('WORLD_SIZE', 1))

    total_batch_size = batch_size*gpus
    step = int((data_len // total_batch_size) * epoch)
    return step

def create_train_dataloaders(data_cfg, opts):
    data_cfg = data_cfg['train']
    dataloaders = {}
    total_step =0
    for d_cfg in data_cfg:
        concate_name = ''
        dataset_ls = []
        dataset=None
        # if 'sample_num' in d_
        videogen = 'videogen' in  d_cfg['task']

        if not isinstance(d_cfg["datasets"], list):
            dset = d_cfg["datasets"]
            if dset["name"] in ["laion2b", "distcc4m", "tuchong", 'distcc12m', "coyo700m","coyo700m_web"]:
                name=dset["name"]
                concate_name=name
                data_path = dset["data_path"]
                ids_path = dset.get("ids_path","")
                tokenizer = dset["tokenizer"]
                rank = get_rank()
                world_size = dist.get_world_size()
                if name=="laion2b":
                    dataset = LaionDataset(data_path, ids_path, opts.video_cfg, tokenizer= tokenizer, caption="caption", is_train=True, rank=rank, world_size=world_size)
                elif name=="tuchong":
                    dataset = LaionDataset(data_path, ids_path, opts.video_cfg, tokenizer= tokenizer, caption="desc", is_train=True, rank=rank, world_size=world_size)
                elif name == "distcc4m":
                    dataset = CC4MDataset(data_path, opts.video_cfg, tokenizer= tokenizer, is_train=True, rank=rank, world_size=world_size)
                elif name == "distcc12m":
                    dataset = CC4MDataset(data_path, opts.video_cfg, tokenizer= tokenizer, is_train=True, rank=rank, world_size=world_size)
                elif name == "coyo700m":
                    dataset = LaionDataset(data_path, ids_path, opts.video_cfg, tokenizer= tokenizer, caption="caption", is_train=True, rank=rank, world_size=world_size)
                elif name =="coyo700m_web":
                    dataset = CoyoWebDataset(data_path,opts.video_cfg, tokenizer= tokenizer, is_train=True)

                # elif name == "webvid10m":
                #     dataset = Webvid10MDataset(data_path, opts.video_cfg, tokenizer= tokenizer, is_train=True, rank=rank, world_size=world_size)
                collate_fn = dist_data_collate
                LOGGER.info("Create Dataset {} success".format(name))

        else:
            for dset in d_cfg['datasets']:
                name = dset['name']
                concate_name = concate_name + name if concate_name == '' else concate_name + '_' + name
                data_type = dset['datatype'] + '_' + name
                video_path = dset['video']
                txt_path = dset['txt']
                key_path = dset['ids_path']
                tokenizer = dset['tokenizer']
                if name == 'webvid_frame':
                    dataset = WebvidFrameDataset(video_path, opts.video_cfg, tokenizer, is_train=True)
                elif name == 'webvid10m':
                    dataset = Webvid10M_IterableDataset(
                        video_path,
                        opts.video_cfg,
                        rank=get_rank(),
                        world_size=dist.get_world_size(),
                        tokenizer=tokenizer,
                        is_train=True,
                        repeat=True)

                else:
                    dataset = TxtVideoDataset(video_path,
                                            txt_path,
                                            key_path,
                                            opts.video_cfg,
                                            tokenizer=tokenizer,
                                            is_train=True,
                                            data_type=data_type)
                collate_fn =  txtvideo_collate
                LOGGER.info("Create Dataset {} Success".format(name))
                dataset_ls.append(dataset)

            if isinstance(dataset_ls[0], torch.utils.data.IterableDataset):
                dataset = ChainDataset(dataset_ls)
            else:
                dataset = ConcatDataset(dataset_ls)

            # dataset = ConcatDataset(dataset_ls)
        # dataset = dataset_ls[0]
        LOGGER.info("Create Dataset {} Success".format(concate_name))
        ratio = d_cfg['mix_ratio']
        task = d_cfg['task']

        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers']

        epoch = d_cfg["epoch"]
        data_len = d_cfg["data_len"]
        steps = compute_steps(epoch, batch_size, data_len)
        if concate_name in ["laion2b", "distcc4m", "tuchong", "distcc12m", "coyo700m", "coyo700m_web"]:
            loader=build_dataloader(dataset, collate_fn, True, opts, batch_size, n_workers, sampler="None")
        else:
            loader = build_dataloader(dataset, collate_fn, True, opts, batch_size, n_workers)
        LOGGER.info("Create dataloader {} Success".format(concate_name))
        task_name = f'{task}--{concate_name}'
        dataloaders[task_name] = [loader, ratio, steps]
        total_step +=steps

    for k, v in dataloaders.items():
        prob = float(v[-1])/total_step
        v.append(prob)

    return dataloaders


def create_val_dataloaders(data_cfg, opts):
    data_cfg = data_cfg['val']
    dataloaders = {}
    for d_cfg in data_cfg:
        name = d_cfg['name']
        data_type = d_cfg['datatype']
        ids_path = d_cfg['ids_path']
        videogen = 'videogen' in  d_cfg['task']
        video_token_path = d_cfg.get('video_token','')
        audio_token_path = d_cfg.get('audio_token','')
        video_mapper = VideoMapper(d_cfg['video'], opts.video_cfg, data_type, is_training=False,videogen=videogen, token_path= video_token_path)
        txt_mapper = TxtMapper(d_cfg['txt'], opts.max_txt_len, data_type)
        audio_path = d_cfg.get('audio','')
        audio_mapper = AudioMapper(audio_path, opts.audio_cfg, data_type, videogen=videogen, token_path=audio_token_path)
        dataset = TxtVideoAudioDataset(ids_path, txt_mapper, video_mapper, audio_mapper)
        collate_fn =  txtvideo_collate
        raw_img_dir_for_FID = d_cfg.get('raw_img_dir_for_FID',None)
        dataset.raw_img_dir_for_FID = raw_img_dir_for_FID
        LOGGER.info("Create Dataset {} Success".format(name))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        loader = build_dataloader(dataset, collate_fn, False, opts, batch_size)
        task_name = f'{task}--{name}'
        dataloaders[task_name] = PrefetchLoader(loader)
    return dataloaders


