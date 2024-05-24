"""

Licensed under the MIT license.

OPT finetuning for video-Text Retrieval
"""
import argparse
import os
from os.path import join
from time import time
import torch
from torch.utils.data import DataLoader,ConcatDataset
from torch.utils.data import Dataset
from tqdm import tqdm
from data.loader import PrefetchLoader
from data.data import TxtVideoDataset, txtvideo_collate
from data.coco_data import RetDataset, txtvideo_collate_coco
from data.vqa import TxtVideoDatasetVQA, txtvideo_vqa_collate
from data.coco_vqa import VQADataset, txtvideo_collate_coco_vqa
from data.vqa import TxtVideoDatasetVQA, txtvideo_vqa_collate
import torch.nn.functional as F
from utils.distributed import DistributedSampler_wopadding
from torch.utils.data.distributed import DistributedSampler
from data.data import worker_init_fn




def build_dataloader(dataset, collate_fn, is_train, opts):
    ngpus = int(os.getenv('WORLD_SIZE', 1))
    batch_size = opts.data_cfg["train_total_batch_size"]//ngpus if is_train else opts.data_cfg["val_total_batch_size"]//ngpus
    if is_train:
        sampler = DistributedSampler(dataset)
    else:
        sampler = DistributedSampler_wopadding(dataset)

    if opts.data_cfg["name"] in ["msrvtt","msvd","didemo"]:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, drop_last=is_train,
                                num_workers=opts.data_cfg["n_workers"],
                                pin_memory=True, collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    elif opts.data_cfg["name"] in ["coco","vqav2"]:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, drop_last=is_train,
                                num_workers=opts.data_cfg["n_workers"],
                                pin_memory=True, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def build_dataset(opts):
    data_type = opts.data_cfg["datatype"]
    eval_task=True

    if opts.data_cfg["name"] in ["msrvtt","msvd"] and opts.finetune_task_name in ["vqa"]:
        train_dataset = TxtVideoDatasetVQA(opts.data_cfg["video_path"],
                                        opts.data_cfg['train_ids_path'],
                                        opts.data_cfg['train_ids_path'],
                                        opts.video_cfg,
                                        is_train=True,
                                        return_all_captions=False,
                                        tokenizer=opts.data_cfg["tokenizer"],
                                        data_type=data_type,
                                        max_txt_len=opts.data_cfg["max_txt_len"]) ### when training and onev-to-manyt, a random t is choosed.


        train_dataloader = build_dataloader(train_dataset, txtvideo_vqa_collate, True, opts)

    elif opts.data_cfg["name"] in ["msrvtt","msvd"]:

        if isinstance(opts.data_cfg["train_ids_path"], (list,tuple)):
            train_ids_paths = opts.data_cfg['train_ids_path']
            train_dataset = ConcatDataset([TxtVideoDataset(opts.data_cfg["video_path"],
                                train_ids_path,
                                train_ids_path,
                                opts.video_cfg,
                                is_train=True,
                                tokenizer=opts.data_cfg["tokenizer"],
                                data_type=data_type,
                                return_all_captions=opts.data_cfg.get('return_all_captions',False),
                                max_txt_len=opts.data_cfg["max_txt_len"]) for train_ids_path in train_ids_paths])

        else:
            min_num_caption = opts.data_cfg.get("min_num_caption", 20)
            train_dataset = TxtVideoDataset(
                opts.data_cfg["video_path"],
                opts.data_cfg['train_ids_path'],
                opts.data_cfg['train_ids_path'],
                opts.video_cfg,
                is_train=True,
                tokenizer=opts.data_cfg["tokenizer"],
                data_type=data_type,
                return_all_captions=opts.data_cfg.get('return_all_captions',
                                                  False),
                min_num_caption=min_num_caption,
                max_txt_len=opts.data_cfg["max_txt_len"],
                msvd_ret=opts.data_cfg.get('msvd_ret', False)
            )  ### when training and onev-to-manyt, a random t is choosed.
        train_dataloader = build_dataloader(train_dataset, txtvideo_collate, True, opts)

    elif opts.data_cfg["name"] in ["didemo"]:
        train_dataset = TxtVideoDataset(
            opts.data_cfg["train_video_path"],
            opts.data_cfg['train_ids_path'],
            opts.data_cfg['train_ids_path'],
            opts.video_cfg,
            is_train=True,
            tokenizer=opts.data_cfg["tokenizer"],
            data_type=data_type,
            return_all_captions=opts.data_cfg.get('return_all_captions',
                                                  False),
            max_txt_len=opts.data_cfg["max_txt_len"]
        )  ### when training and onev-to-manyt, a random t is choosed.
        train_dataloader = build_dataloader(train_dataset, txtvideo_collate, True, opts)

    elif opts.data_cfg["name"] in ["coco",]:
        train_dataset = RetDataset(opts.data_cfg["train_ann_path"],
                                   opts.data_cfg["image_root"],
                                   opts.video_cfg,
                                   is_train=True,
                                   tokenizer=opts.data_cfg["tokenizer"]
                                  )
        train_dataloader = build_dataloader(train_dataset, txtvideo_collate_coco, True, opts)

    elif opts.data_cfg["name"] in ["vqav2",]:

        if isinstance(opts.data_cfg["train_ann_path"], (list,tuple)):
            train_ann_paths = opts.data_cfg["train_ann_path"]
            train_dataset = ConcatDataset([VQADataset(train_ann_path,
                                    opts.data_cfg["image_root"],
                                    opts.video_cfg,
                                    is_train=True,
                                    tokenizer=opts.data_cfg["tokenizer"]
                                    )  for train_ann_path in train_ann_paths])
        else:
            train_dataset = VQADataset(opts.data_cfg["train_ann_path"],
                                    opts.data_cfg["image_root"],
                                    opts.video_cfg,
                                    is_train=True,
                                    tokenizer=opts.data_cfg["tokenizer"]
                                    )
        train_dataloader = build_dataloader(train_dataset, txtvideo_collate_coco_vqa, True, opts)

    # change test sample num
    if 'test_sample_num' in opts.video_cfg:
        opts.video_cfg['train_sample_num'] = opts.video_cfg['sample_num']
        opts.video_cfg['sample_num'] = opts.video_cfg['test_sample_num']

    if eval_task:

        if opts.data_cfg["name"] in ["msrvtt","msvd"] and opts.finetune_task_name in ["vqa"]:
            test_dataset = TxtVideoDatasetVQA(opts.data_cfg["video_path"],
                                        opts.data_cfg['test_ids_path'],
                                        opts.data_cfg['test_ids_path'],
                                        opts.video_cfg,
                                        is_train=False,
                                        tokenizer=opts.data_cfg["tokenizer"],
                                        data_type=data_type,
                                        return_all_captions=False,
                                        max_txt_len=opts.data_cfg["max_txt_len"])

            test_loader = build_dataloader(test_dataset, txtvideo_vqa_collate, False, opts)

        elif opts.data_cfg["name"] in ["msrvtt","msvd"]:
            test_dataset = TxtVideoDataset(opts.data_cfg["video_path"],
                                        opts.data_cfg['test_ids_path'],
                                        opts.data_cfg['test_ids_path'],
                                        opts.video_cfg,
                                        is_train=False,
                                        tokenizer=opts.data_cfg["tokenizer"],
                                        data_type=data_type,
                                        return_all_captions=True,
                                        max_txt_len=opts.data_cfg["max_txt_len"],
                                        msvd_ret=opts.data_cfg.get('msvd_ret',False))
            test_loader = build_dataloader(test_dataset, txtvideo_collate, False, opts)

        elif opts.data_cfg["name"] in ["didemo"]:
            test_dataset = TxtVideoDataset(opts.data_cfg["test_video_path"],
                                        opts.data_cfg['test_ids_path'],
                                        opts.data_cfg['test_ids_path'],
                                        opts.video_cfg,
                                        is_train=False,
                                        tokenizer=opts.data_cfg["tokenizer"],
                                        data_type=data_type,
                                        return_all_captions=True,
                                        max_txt_len=opts.data_cfg["max_txt_len"])
            test_loader = build_dataloader(test_dataset, txtvideo_collate, False, opts)

        elif opts.data_cfg["name"] in ["coco",]:
            test_dataset = RetDataset(opts.data_cfg["test_ann_path"],
                                    opts.data_cfg["image_root"],
                                    opts.video_cfg,
                                    is_train=False,
                                    tokenizer=opts.data_cfg["tokenizer"]
                                    )
            test_loader = build_dataloader(test_dataset, txtvideo_collate_coco, False, opts)

        elif opts.data_cfg["name"] in ["vqav2",]:
            test_dataset = VQADataset(opts.data_cfg["test_ann_path"],
                                    opts.data_cfg["image_root"],
                                    opts.video_cfg,
                                    is_train=False,
                                    tokenizer=opts.data_cfg["tokenizer"]
                                    )
            test_loader = build_dataloader(test_dataset, txtvideo_collate_coco_vqa, False, opts)

    if 'train_sample_num' in opts.video_cfg:
        opts.video_cfg['sample_num'] = opts.video_cfg['train_sample_num']
    return train_dataloader, test_loader, train_dataset
