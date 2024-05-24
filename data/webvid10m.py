"""

Licensed under the MIT license.

Dataset interfaces
"""

import json
import logging as log
import os
import pickle
import random
import string
from base64 import b64decode
from os.path import join
from time import time
from typing import Any, Dict, List, Optional, Tuple

import ipdb
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_pretrained_bert import BertTokenizer
from toolz.sandbox import unzip
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from torchvision.transforms.transforms import *

from utils.hdfs_io import hlist_files, hload_pkl, hopen
from utils.logger import LOGGER

from .tokenizer import build_tokenizer

punctuation = string.punctuation
import gc
import io
import math
import multiprocessing as mp

import decord
import ipdb
import numpy as np
from dataloader import KVReader



def split_shard(data: List[Any], shard_idx: int, shard_size: int):
    num = len(data)
    if num < shard_size:
        raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
    start_idx = (num * shard_idx) // shard_size
    end_idx = (num * (shard_idx + 1)) // shard_size
    return data[start_idx: end_idx]



class TxtMapper(object):

    def __init__(self,
                 return_all_captions,
                tokenizer: dict = None,
                 max_txt_len=30,
                 data_type=''):

        self.data_type = data_type
        self.max_txt_len = max_txt_len

        meta = json.load(open(join('./data/meta.json')))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']
        self.eos = 10

        self.tokenizer_info = build_tokenizer(tokenizer)

        self.txt_encoder_tokenizer = self.tokenizer_info["txt_encoder"][
            "tokenizer"]
        self.txt_encoder_bos = self.tokenizer_info["txt_encoder"]["bos"]
        self.txt_encoder_eos = self.tokenizer_info["txt_encoder"]["eos"]

        self.multi_encoder_tokenizer = self.tokenizer_info["multi_encoder"][
            "tokenizer"]
        self.multi_encoder_bos = self.tokenizer_info["multi_encoder"]["bos"]
        self.multi_encoder_eos = self.tokenizer_info["multi_encoder"]["eos"]

        self.punctuations = string.punctuation
        self.return_all_captions = return_all_captions

    def process(self, text):

        output_txt = self.get_single_txt(
            text,
            self.txt_encoder_tokenizer,
            self.txt_encoder_bos,
            self.txt_encoder_eos,
        )
        output_multi = self.get_single_txt(
            text,
            self.multi_encoder_tokenizer,
            self.multi_encoder_bos,
            self.multi_encoder_eos,
        )

        return {
            "txt_tokens": output_txt,
            "multi_tokens": output_multi,
            "video_clip": None,
        }

    def get_single_txt(self, text, tokenizer, bos, eos):
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
            text = text.replace(i, '')
        return text

    def tokenize(self, tokenizer, text):
        ids = []
        for word in text.strip().split():
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        return ids

    def detokenize(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

class VideoMapper(object):

    def __init__(
        self,
        video_cfg,
        is_train=True,
        videogen=False,
        token_path="",
    ):
        self.videogen = videogen
        self.frame_syncaug = True
        self.is_train = is_train
        if not self.videogen:
            self.sample_num = video_cfg["sample_num"]
            self.resolution = video_cfg["resolution"]
            self.mean = video_cfg["mean"]
            self.std = video_cfg["std"]
            # self.patch_num = self.resolution // video_cfg['patch_size']
            # self.file_dict = hload_pkl(txt_dir)
            LOGGER.info(f"resolution : {self.resolution}")
            LOGGER.info(f"sample_num : {self.sample_num}")
            if is_train:

                self.transforms = transforms.Compose([
                    RandomResizedCrop(self.resolution, [0.8, 1.0], [1.0, 1.0]),
                    RandomHorizontalFlip(),
                    Normalize(self.mean, self.std),
                ])
            else:
                self.transforms = transforms.Compose([
                    Resize(self.resolution),
                    CenterCrop(self.resolution),
                    Normalize(self.mean, self.std),
                ])

        else:
            self.token_path = token_path

    def process(self, frames):
        sample_num = self.sample_num
        video_list = []
        # time_begin = time.time()
        # sample_idx = random.sample(list(range(len(frames))), sample_num)
        sample_idx = sorted(np.random.choice(range(len(frames)), sample_num))

        for ind in sample_idx:
            frame = tf.image.decode_jpeg(frames[ind],
                                         channels=3,
                                         dct_method='INTEGER_ACCURATE').numpy()
            frame = transforms.ToTensor()(frame)  ## frame: 3XhXw

            video_list.append(frame.unsqueeze(0))
        video_pixels = torch.cat(video_list, dim=0)  ### nX3xHxW

        video_pixels = self.transforms(video_pixels)

        return video_pixels

class WrapDataset(IterableDataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """
    def __init__(self,
                 data_path: str,
                 rank: int = 0,
                 world_size: int = 1,
                 data_format: str = 'KVReader',
                 shuffle: bool = True,
                 repeat: bool = False,
                 verbose: bool = True,
                 pipeline=None,
                 text_field = [],
                 image_bin_field = [],
                 image_name_field=[],
                 batch_sizes=1,
                 **kwargs):
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.data_format = data_format
        self.pipeline = Compose(pipeline)
        self.text_field = text_field
        self.image_bin_field = image_bin_field
        self.image_name_field = image_name_field
        self.batch_sizes = 1#batch_sizes[0]
        self.data_num = 0
        self.iter_num = 0
        self.dataset_len = 0
        self.files = hlist_files(data_path.split(','))
        if self.data_format == 'json':
            self.files = [f for f in self.files if f.find('_SUCCESS') < 0]
            self.files.sort()
        elif self.data_format == 'KVReader':
            self.files = [f for f in self.files if f.find('_SUCCESS') < 0]
            self.files = [file[:-6] for file in self.files if '.index' in file]
            self.files.sort()
        self.is_hdfs = data_path.startswith('hdfs')
        self.repeat = repeat
        log.info(
            '[DATA]--all dataset containing {} files.'.format(len(self.files)))
        if len(self.files) % self.world_size != 0:
            log.info('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        self.verbose = verbose
        #兼容codebase的enumerator
        self.data_iter = self.generate()

    def transform(self, data):
        video_bin, results = data
        results['img_bin'] = video_bin
        return self.pipeline(results)

    def generate(self, seed=42):
        """
        # TODO: 加cache，加prefetch
        在dataloader里调用，dataloader会启num_worker个进程来遍历dataset。
        这个函数一开始做了两次split_shard，对文件进行划分。
        self.files是总的数据集的文件数，
        第一次split_shard是基于rank和world_size来分的，是gpu节点维度的；
        第二次split_shard是基于worker_info的，是一个gpu节点的dataloader内，给不同的worker划分文件。
        """
        # 第一次 split: 按 rank 划分。
        # 先对files做一次sort和（seed）shuffle，每个rank拿到的seed都是一样的。这样得到的list是一样的，保证split时不重复。
        # TODO: 这里的seed实际一直一样，后面可以想办法，让seed 从trainer 传过来，让每个epoch里，每个rank拿到的file不一样，更好的做shuffle。
        #if self.shuffle:
        #    self.files = self.sort_and_shuffle(self.files, seed)
        #else:
        #    self.files.sort()
        if self.world_size == 1 or len(self.files) == 1:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = split_shard(
                self.files, self.rank, self.world_size)
        # 第二次 split：各个rank内部，将 cur_dataloader_files 按 num_workers 分。注意每个worker都会执行。
        # 每个rank的每个 worker 拿到的都是这个：cur_dataloader_files，是一样的
        while True:
            if self.shuffle:
                self.files = self.sort_and_shuffle(self.files, seed)
            else:
                self.files.sort()
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                if len(cur_dataloader_files) % worker_info.num_workers != 0 and self.verbose:
                    log.info('[DATA]--current dataloader [%s] file num %s cannot split to worker_num %s ' %
                             (self.rank, len(cur_dataloader_files), worker_info.num_workers))
                # 这里是真正做第二次split的地方， cur_worker_files 是每个worker 拿到的
                cur_worker_files = split_shard(
                    cur_dataloader_files, worker_info.id, worker_info.num_workers)
            else:
                # num_worker=0，只有主进程的情况
                cur_worker_files = cur_dataloader_files
            if self.shuffle:  # 每个epoch下，虽然每个rank-每个worker对应的files是一样的，但会这里shuffle一下，读的顺序按file有打乱。
                random.shuffle(cur_worker_files)
            # cur_worker_files 是每个worker拿到的结果
            if self.verbose:
                log.info(
                    f"[DataLoader] --> Rank:[{self.rank}]  Workers:[{worker_info.id if worker_info else 0}] process file: {len(cur_worker_files)} :{self.get_surfix(cur_worker_files[:3])}  ..."
                )
            if self.data_format == "json":
                for filepath in cur_worker_files:
                    if self.is_hdfs:
                        with hopen(filepath, 'r') as reader:
                            for line in reader:
                                yield line.decode()
                        continue
                    with open(filepath, 'r') as reader:
                        for line in reader:
                            yield line
            elif self.data_format == "KVReader":
                for filepath in cur_worker_files:
                    # if self.is_hdfs:
                        #print("begin loading data ...")
                    data_reader = KVReader(filepath, 8)
                    keys = data_reader.list_keys()
                    self.data_num += len(keys)
                    self.iter_num += 1
                    random.shuffle(keys)
                    for key_idx in range(0, len(keys), 100):
                        cur_keys = keys[key_idx:key_idx+100]
                        try:
                            values = data_reader.read_many(cur_keys)
                        except:
                            print("error_keys", cur_keys)
                            continue
                        for value in values:
                            #print("return one value")
                            yield value
            if not self.repeat:
                break
    def __iter__(self):
        for data in self.generate():
            try:
                data_item = json.loads(data)
                results = {}
                # 1. 文本处理：根据text_field 来取文本的字段
                text = ''
                if True:
                    for field in self.text_field:
                        if field in data_item:
                            cur_field = data_item[field]
                            if cur_field:
                                text += cur_field + ' '
                    text = text.strip()
                results['text'] = text

                # 2. image 处理 兼容抖音封面数据和图搜数据
                if True:
                    for field in self.image_bin_field:
                        if field in data_item:
                            image_binary = data_item[field]
                            image_str = b64decode(image_binary)
                            break
                    for field in self.image_name_field:
                        if field in data_item:
                            image_name = data_item[field]
                            break
                    results['img_info'] = {'filename':image_name}
                    results['img_prefix'] = None
                    results['modality'] = 'RGB'
                    results['label'] = -1  # 为了避免改代码，加上这一行，这样可以在recognizer3d上运行，因为它必须要求有label

                res = self.transform((image_str, results))
                #print(res)
                yield res
            except Exception as e:
                log.error('encounter broken data: %s' % e)
                log.error(data_item.keys())
                continue
    def reset(self, seed):
        return self.generate(seed)
    def sort_and_shuffle(self, data, seed):
        data.sort()
        random.Random(seed).shuffle(data)
        return data
    def get_surfix(self, name_list):
        return [n.split('/')[-1] for n in name_list]



class Webvid10M_IterableDataset(WrapDataset):

    def __init__(self,
                 data_path: str,
                 video_cfg: dict,
                 tokenizer: dict = None,
                 is_train=True,
                 return_all_captions=False,
                 pipeline=None,
                 rank: int = 0,
                 world_size: int = 1,
                 batch_sizes=1,
                 repeat=False,
                 **kwargs):
        super().__init__(data_path,
                         rank,
                         world_size,
                         batch_sizes=batch_sizes,
                         pipeline=pipeline,
                         repeat=repeat)
        self.txt_mapper = TxtMapper(return_all_captions,
                                    tokenizer=tokenizer,
                                    **kwargs)
        self.video_mapper = VideoMapper(video_cfg, is_train=is_train)
        self.punctuations = string.punctuation

    def decode_tfrecord(self, example_proto):

        ## example_proto: kvreader 里面的value，是bytes

        ## 文本部分
        context_features = {}
        context_features['title'] = tf.io.FixedLenFeature([], dtype=tf.string)

        ## frames 部分
        sequence_features = {
            'data': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        }

        contexts, sequences = tf.io.parse_single_sequence_example(
            example_proto,
            context_features=context_features,
            sequence_features=sequence_features)

        # rawframes and title
        rawframes = sequences['data']
        title_text = contexts['title'].numpy().decode('utf-8')
        return rawframes, title_text

    def __iter__(self, ):
        for data in self.generate():
            ### image
            try:
                data_item = data
                results = {}
                filename = 'test'
                rawframes, title_text = self.decode_tfrecord(data_item)
                video_pixels = self.video_mapper.process(rawframes)
                ### text
                text = title_text
                output = self.txt_mapper.process(text)
                txt_tokens = output["txt_tokens"]
                multi_tokens = output["multi_tokens"]
                id_txt = filename
                # text = self.clean(text)
                # yield results
                yield filename, txt_tokens, multi_tokens, video_pixels, id_txt
            except Exception as e:
                log.error('encounter broken data: %s' % e)
                log.error(data_item.keys())
                continue

    def __len__(self):
        return 238720 * 32 // self.world_size

    def clean(self, text):
        """remove duplicate spaces, lower and remove punctuations """
        text = ' '.join([i for i in text.split(' ') if i != ''])
        text = text.lower()
        for i in self.punctuations:
            text = text.replace(i, '')
        return text

    def split(self, frame_name_lists, sample_num):
        if len(frame_name_lists) < sample_num:  ###padding with the last frame
            frame_name_lists += [frame_name_lists[-1]
                                 ] * (sample_num - len(frame_name_lists))
        k, m = divmod(len(frame_name_lists), sample_num)
        return [
            frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in list(range(sample_num))
        ]


def txtvideo_collate(inputs):

    (ids, txt_tokens, multi_tokens, video_pixels,
     id_txts) = map(list, unzip(inputs))
    txt_tokens = torch.stack(txt_tokens, dim=0)
    multi_tokens = torch.stack(multi_tokens, dim=0)
    video_pixels = torch.stack(video_pixels, dim=0)
    batch2m = {
        'ids': ids,
        'txt_tokens': txt_tokens,
        "multi_tokens": multi_tokens,
        'video_pixels': video_pixels,
        'ids_txt': id_txts
    }
    batch = {}
    batch["batch_2m"] = batch2m
    return batch


def get_keys(args):
    return KVReader(*args).list_keys()


from torch.utils.data import ChainDataset, ConcatDataset


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Avoid "cannot pickle KVReader object" error
    if isinstance(dataset, ConcatDataset):
        for i in range(len(dataset.datasets)):
            dataset.datasets[i].video_reader = KVReader(
                dataset.datasets[i].video_path,
                dataset.datasets[i].num_readers)
            dataset.datasets[i].txt_reader = None
            if not dataset.datasets[i].txt_path.endswith("pkl"):
                dataset.datasets[i].txt_reader = KVReader(
                    dataset.datasets[i].txt_path,
                    dataset.datasets[i].num_readers)
    elif isinstance(dataset, ChainDataset):
        pass
    else:
        dataset.video_reader = KVReader(dataset.video_path,
                                        dataset.num_readers)
        dataset.txt_reader = None
        if not dataset.txt_path.endswith("pkl"):
            dataset.txt_reader = KVReader(dataset.txt_path,
                                          dataset.num_readers)


import os

import torch.distributed as dist
from torch.utils.data import DataLoader

from utils.distributed import DistributedSampler_wopadding

# from data.sampler import DistributedSampler
if __name__ == "__main__":

    # key_path = "hdfs://haruna/home/byte_labcv_default/liyinan/msrvtt/annotation/test_Retri_1k_anno.pkl"
    # txt_path = "hdfs://haruna/home/byte_labcv_default/liyinan/msrvtt/annotation/test_Retri_1k_anno.pkl"
    # video_path = "hdfs://haruna/home/byte_labcv_default/liyinan/msrvtt/all"
    key_path = "/PATH/webvid2.5M_package/webvid_pretrain_anno_new.pkl"
    txt_path = "/PATH/webvid2.5M_package/webvid_pretrain_anno_new.pkl"
    video_path = "/PATH/webvid2.5M_package/webvid2.5M_pretrain"
    # key_path = "hdfs://haruna/home/byte_labcv_default/liyinan/howto100m_annotations/howto100m_csv_realfps_onlykeys.pkl"
    # txt_path = "hdfs://haruna/home/byte_labcv_default/liyinan/howto100m_annotations/howto100m_realfps_csv_caption_package/howto100m_realfps_csv"
    # video_path = "hdfs://haruna/dp/mloops/datasets/howto100m/howto100m/arnold/v4/howto100m_video_package_40/howto100m_pretrain"
    # key_path = "hdfs://haruna/dp/mloops/datasets/webvid/webvid2_5m/arnold/v1/webvid2.5M_package/webvid_pretrain_anno_new.pkl"
    # txt_path = "hdfs://haruna/dp/mloops/datasets/webvid/webvid2_5m/arnold/v1/webvid2.5M_package/webvid_pretrain_anno_new.pkl"
    # video_path = "hdfs://haruna/dp/mloops/datasets/webvid/webvid2_5m/arnold/v1/webvid2.5M_package/webvid2.5M_pretrain"
    video_cfg = {"sample_num": 3, "resolution": 224, "mean": 3, "std": 4}
    num_readers = 32
    data_path = 'hdfs://haruna/home/byte_labcv_gan/public/multimodal_all/video/webvid/train10MAll_new/'

    tokenizer={"txt_encoder":"bert_tokenizer","multi_encoder":"bert_torkenizer"}
    train_data = Webvid10M_IterableDataset(data_path, video_cfg, tokenizer=tokenizer, return_all_captions=False)

    # train_data = KVDataset(video_path,32)#
    # ipdb.set_trace()
    local_rank = -1
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print(dist.get_rank())
    batch_size = 32
    # sampler = DistributedSampler_wopadding(train_data)
    # sampler = KVSampler(train_data, batch_size=batch_size, num_replicas=2, rank=1, shuffle=True)
    dataloader = torch.utils.data.DataLoader(train_data,
                                             batch_size=32,
                                             drop_last=False,
                                             num_workers=8,
                                             pin_memory=False,
                                             collate_fn=txtvideo_collate,
                                             worker_init_fn=worker_init_fn)

    for _ in range(5):  # 5 epoches
        for batch in dataloader:
            ipdb.set_trace()
            print(
                "========"
            )  # iterate through your training data (batch['batch_2m']['video_pixels']).shape
        print("Finish one epoch!")
