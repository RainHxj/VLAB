"""

Licensed under the MIT license.

Dataset interfaces
"""

from .tokenizer import build_tokenizer
import json
from toolz.sandbox import unzip
import torch
from torch.utils.data import Dataset

from torchvision.transforms.transforms import *
from torchvision import transforms
import random
from os.path import join
from PIL import Image
from utils.logger import LOGGER

import string

punctuation = string.punctuation
import numpy as np
import pickle
import io
import multiprocessing as mp
import decord
import ipdb
import gc
import time


class TxtMapper(object):

    def __init__(self,
                 text_path,
                 return_all_captions,
                 tokenizer="bert_tokenizer",
                 max_txt_len=30,
                 data_type='',
                 min_num_cap=1):

        self.data_type = data_type
        self.max_txt_len = max_txt_len

        meta = json.load(open(join('./data/meta.json')))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']
        self.eos = 10
        self.text_path = text_path
        if self.text_path.endswith("pkl"):
            self.file_dict = hload_pkl(self.text_path)

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
        self.min_num_cap = min_num_cap

    def __getitem__(self, file_name, id_, txt_reader):

        info = self.get_info(file_name, id_, txt_reader)
        text = info["text"]
        rand_id = None

        if isinstance(text, list):
            if not self.return_all_captions:
                rand_text = random.choice(text)
                rand_id = text.index(rand_text)
                output_txt = self.get_single_txt(rand_text,
                                                 self.txt_encoder_tokenizer,
                                                 self.txt_encoder_bos,
                                                 self.txt_encoder_eos)
                output_multi = self.get_single_txt(
                    rand_text, self.multi_encoder_tokenizer,
                    self.multi_encoder_bos, self.multi_encoder_eos)
            else:
                output_txt = []
                output_multi = []
                for i, t in enumerate(text):
                    txt = self.get_single_txt(t, self.txt_encoder_tokenizer,
                                              self.txt_encoder_bos,
                                              self.txt_encoder_eos)
                    multi = self.get_single_txt(t,
                                                self.multi_encoder_tokenizer,
                                                self.multi_encoder_bos,
                                                self.multi_encoder_eos)
                    output_txt.append(txt)
                    if i  < self.min_num_cap:
                        output_multi.append(multi)
                if i < self.min_num_cap - 1:
                    for _ in range(self.min_num_cap - i - 1):
                        t = random.choice(text)
                        txt = self.get_single_txt(t,
                                                  self.txt_encoder_tokenizer,
                                                  self.txt_encoder_bos,
                                                  self.txt_encoder_eos)
                        multi = self.get_single_txt(
                            t, self.multi_encoder_tokenizer,
                            self.multi_encoder_bos, self.multi_encoder_eos)
                        output_multi.append(multi)
                output_multi = torch.stack(output_multi)
        else:
            output_txt = self.get_single_txt(text, self.txt_encoder_tokenizer,
                                             self.txt_encoder_bos,
                                             self.txt_encoder_eos)
            output_multi = self.get_single_txt(text,
                                               self.multi_encoder_tokenizer,
                                               self.multi_encoder_bos,
                                               self.multi_encoder_eos)

        if rand_id and info.get("start", None) and info.get("end", None):
            return {
                "txt_tokens": output_txt,
                "multi_tokens": output_multi,
                "video_clip": [info["start"][rand_id], info["end"][rand_id]]
            }

        return {
            "txt_tokens": output_txt,
            "multi_tokens": output_multi,
            "video_clip": None
        }

    def get_info(self, file_name, id, txt_reader):

        if self.text_path.endswith("pkl"):
            return self.file_dict[id]
        else:
            assert txt_reader is not None, "text reader is None"
            info = txt_reader.read_many([file_name])
            info = eval(info[0])
            return info

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

    def __init__(self,
                 video_dir,
                 video_cfg,
                 data_type='video',
                 is_train=True,
                 videogen=False,
                 token_path=''):
        self.videogen = videogen
        self.video_dir = video_dir
        self.datatype = data_type
        self.frame_syncaug = True
        self.is_train = is_train
        if not self.videogen:
            self.sample_num = video_cfg['sample_num']
            self.resolution = video_cfg['resolution']
            self.mean = video_cfg['mean']
            self.std = video_cfg['std']
            # self.patch_num = self.resolution // video_cfg['patch_size']
            # self.file_dict = hload_pkl(txt_dir)
            LOGGER.info(f'{data_type} resolution : {self.resolution}')
            LOGGER.info(f'{data_type} sample_num : {self.sample_num}')
            if is_train:
                self.transforms = transforms.Compose([
                    RandomResizedCrop(self.resolution, [0.8, 1.0], [1.0, 1.0]),
                    RandomHorizontalFlip(),
                    Normalize(self.mean, self.std)
                ])
            else:
                self.transforms = transforms.Compose([
                    Resize(self.resolution),
                    CenterCrop(self.resolution),
                    Normalize(self.mean, self.std)
                ])
        else:
            self.token_path = token_path

    @staticmethod
    def frame_generator(container, stream):
        """Frame generator for PyAV."""
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_rgb().to_ndarray()

    def __getitem__(self, key, video_clip, video_reader):

        if self.datatype.startswith('video'):
            if self.videogen:
                video_tokens = np.load(
                    os.path.join(self.token_path, id_ + '.npy'))  ### 10,8,8
                video_tokens = torch.from_numpy(video_tokens).reshape(-1)
                return video_tokens
            else:
                try:
                    sample_num = self.sample_num
                    video_pixels = []

                    videos = video_reader.read_many([
                        key,
                    ])

                    # e_t = time.time()
                    # print("data time: {}".format(s_t-e_t))

                    # s_t = time.time()

                    # with io.BytesIO(videos[0]) as file_obj:
                    file_obj = io.BytesIO(videos[0])
                    # container = av.open(file_obj, metadata_errors="ignore")
                    container = decord.VideoReader(file_obj)
                    # ipdb.set_trace()
                    # frames=[i for i in container.decode(video=0)]
                    if video_clip is not None:
                        frames_ids = list(range(video_clip[0], video_clip[1]))
                        # frames=frames[video_clip[0]:video_clip[1]]
                    else:
                        frames_ids = list(range(len(container)))

                    # container.streams.video[0].thread_type = 'AUTO'
                    # frames_ids = range(0,container.streams.video[0].frames)
                    frames_splited = self.split(frames_ids, sample_num)
                    if self.is_train:
                        sample_idx = [random.choice(i) for i in frames_splited]
                    else:
                        sample_idx = [
                            i[(len(i) + 1) // 2 - 1] for i in frames_splited
                        ]

                    frames = container.get_batch(sample_idx).asnumpy()

                    video_list = []

                    for i in frames:
                        frame = Image.fromarray(i)
                        frame = transforms.ToTensor()(frame)  ## frame: 3XhXw
                        video_list.append(frame.unsqueeze(0))
                    video_pixels = torch.cat(video_list, dim=0)  ### nX3xHxW
                    video_pixels = self.transforms(video_pixels)

                    for v in video_list:
                        del v
                    del container
                    gc.collect()

                    return video_pixels

                except:
                    return None
        elif self.datatype.startswith("frame"):
            try:

                sample_num = self.sample_num
                # print(key)
                videos = video_reader.read_many([
                    key,
                ])
                video_list = []

                frames = pickle.loads(videos[0])
                sample_idx = random.sample(list(range(frames.shape[0])),
                                           sample_num)

                for ind in sample_idx:
                    frame = Image.fromarray(frames[ind])
                    frame = transforms.ToTensor()(frame)  ## frame: 3XhXw

                    video_list.append(frame.unsqueeze(0))
                video_pixels = torch.cat(video_list, dim=0)  ### nX3xHxW
                video_pixels = self.transforms(video_pixels)

                return video_pixels
            except:
                return None

        elif self.datatype.startswith('image'):
            try:
                sample_num = self.sample_num
                video_list = []
                videos = video_reader.read_many([
                    key,
                ])
                with io.BytesIO(videos[0]) as f:
                    img = Image.open(f)
                    img = img.convert("RGB")
                    img = np.array(img)
                    img = transforms.ToTensor()(img)
                img = img.unsqueeze(0)

                assert img.shape[1] == 3, "image dim is not 3"
                for i in range(sample_num):
                    video_list.append(img)
                video_pixels = torch.cat(video_list, dim=0)  ### nX3xHxW
                video_pixels = self.transforms(video_pixels)
                for v in video_list:
                    del v
                return video_pixels
            except:
                return None

    def split(self, frame_name_lists, sample_num):
        if len(frame_name_lists) < sample_num:  ###padding with the last frame
            frame_name_lists += [frame_name_lists[-1]
                                 ] * (sample_num - len(frame_name_lists))
        k, m = divmod(len(frame_name_lists), sample_num)
        return [
            frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in list(range(sample_num))
        ]


from dataloader import KVReader
from utils.hdfs_io import hload_pkl


class TxtVideoDataset(torch.utils.data.Dataset):

    def __init__(self,
                 video_path,
                 txt_path,
                 key_path,
                 video_cfg,
                 num_readers=12,
                 tokenizer="bert_tokenizer",
                 is_train=True,
                 data_type="video",
                 return_all_captions=False,
                 msvd_ret=False,
                 min_num_caption=1,
                 **kwargs):

        self.txt_mapper = TxtMapper(txt_path,
                                    return_all_captions,
                                    min_num_cap=min_num_caption,
                                    tokenizer=tokenizer,
                                    **kwargs)
        self.video_mapper = VideoMapper(video_path,
                                        video_cfg,
                                        is_train=is_train,
                                        data_type=data_type)

        self.dataset_name = self.video_mapper.datatype.split('_')[-1]
        self.return_all_captions = return_all_captions

        self.video_path = video_path
        self.num_readers = num_readers
        self.txt_path = txt_path
        self.key_path = key_path
        self.ids = self.get_ids()
        self.idx = list(range(len(self.ids)))
        self.msvd_ret = msvd_ret

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):

        id_ = self.ids[i]
        output = self.txt_mapper.__getitem__(id_["filename"], i,
                                             self.txt_reader)
        txt_tokens = output["txt_tokens"]
        multi_tokens = output["multi_tokens"]
        if not self.return_all_captions or not isinstance(txt_tokens, list):
            id_txt = [id_['filename']]
        else:
            id_txt = [id_['filename']] * len(txt_tokens)

        video_clip = output["video_clip"]

        video_pixels = self.video_mapper.__getitem__(id_['filename'],
                                                     video_clip,
                                                     self.video_reader)

        if video_pixels is None:  ###wrong img/video and needs to resample
            resample_idx = random.choice(self.idx)

            LOGGER.info(
                f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.'
            )
            return self.__getitem__(resample_idx)
        if self.msvd_ret:
            return id_[
                "filename"], txt_tokens, txt_tokens, video_pixels, id_txt
        else:
            return id_[
                "filename"], txt_tokens, multi_tokens, video_pixels, id_txt

    def get_ids(self):

        assert self.key_path.endswith("pkl")
        file_list = hload_pkl(self.key_path)

        return file_list


def txtvideo_collate(inputs):

    (ids, txt_tokens, multi_tokens, video_pixels,
     id_txts) = map(list, unzip(inputs))
    id_txts = [j for i in id_txts for j in i]

    if isinstance(txt_tokens[0], list):
        txt_tokens = [j for i in txt_tokens for j in i]
        txt_tokens = torch.stack(txt_tokens, dim=0)
    else:
        txt_tokens = torch.stack(txt_tokens, dim=0)

    # multi_tokens = None
    # if isinstance(multi_tokens[0],list):
    #     multi_tokens = [ j  for i in multi_tokens for j in i]
    #     multi_tokens = torch.stack(multi_tokens,dim = 0)
    # else:
    if isinstance(multi_tokens[0], list):
        multi_tokens = [j for i in multi_tokens for j in i]
        multi_tokens = torch.stack(multi_tokens, dim=0)
    else:
        multi_tokens = torch.stack(multi_tokens, dim=0)

    # multi_tokens = torch.stack(multi_tokens,dim = 0)
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


# from torch.utils.data import ConcatDataset
# def worker_init_fn(_):
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset
#     # Avoid "cannot pickle KVReader object" error
#     if  isinstance(dataset, ConcatDataset):
#         for i in range(len(dataset.datasets)):
#             dataset.datasets[i].video_reader = KVReader(dataset.datasets[i].video_path, dataset.datasets[i].num_readers)
#             dataset.datasets[i].txt_reader = None
#             if not dataset.datasets[i].txt_path.endswith("pkl"):
#                 dataset.datasets[i].txt_reader = KVReader(dataset.datasets[i].txt_path, dataset.datasets[i].num_readers)
#     else:
#         dataset.video_reader = KVReader(dataset.video_path, dataset.num_readers)
#         dataset.txt_reader = None
#         if not dataset.txt_path.endswith("pkl"):
#             dataset.txt_reader = KVReader(dataset.txt_path, dataset.num_readers)

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


import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.distributed import DistributedSampler_wopadding
import os
# from data.sampler import DistributedSampler
if __name__ == "__main__":

    # key_path = "hdfs://haruna/home/byte_labcv_default/liyinan/msrvtt/annotation/test_Retri_1k_anno.pkl"
    # txt_path = "hdfs://haruna/home/byte_labcv_default/liyinan/msrvtt/annotation/test_Retri_1k_anno.pkl"
    # video_path = "hdfs://haruna/home/byte_labcv_default/liyinan/msrvtt/all"
    key_path = "/PATH/datasets/webvid2.5M_package/webvid_pretrain_anno_new.pkl"
    txt_path = "/PATH/datasets/webvid2.5M_package/webvid_pretrain_anno_new.pkl"
    video_path = "/PATH/datasets/webvid2.5M_package/webvid2.5M_pretrain"
    # key_path = "hdfs://haruna/home/byte_labcv_default/liyinan/howto100m_annotations/howto100m_csv_realfps_onlykeys.pkl"
    # txt_path = "hdfs://haruna/home/byte_labcv_default/liyinan/howto100m_annotations/howto100m_realfps_csv_caption_package/howto100m_realfps_csv"
    # video_path = "hdfs://haruna/dp/mloops/datasets/howto100m/howto100m/arnold/v4/howto100m_video_package_40/howto100m_pretrain"
    # key_path = "hdfs://haruna/dp/mloops/datasets/webvid/webvid2_5m/arnold/v1/webvid2.5M_package/webvid_pretrain_anno_new.pkl"
    # txt_path = "hdfs://haruna/dp/mloops/datasets/webvid/webvid2_5m/arnold/v1/webvid2.5M_package/webvid_pretrain_anno_new.pkl"
    # video_path = "hdfs://haruna/dp/mloops/datasets/webvid/webvid2_5m/arnold/v1/webvid2.5M_package/webvid2.5M_pretrain"
    video_cfg = {"sample_num": 3, "resolution": 224, "mean": 3, "std": 4}
    num_readers = 32

    train_data = TxtVideoDataset(video_path,
                                 txt_path,
                                 key_path,
                                 video_cfg,
                                 tokenizer={
                                     "txt_encoder": "bert_tokenizer",
                                     "multi_encoder": "bert_torkenizer"
                                 },
                                 is_train=True,
                                 data_type="video",
                                 return_all_captions=False)

    # train_data = KVDataset(video_path,32)#
    # ipdb.set_trace()
    local_rank = -1
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print(dist.get_rank())
    batch_size = 32
    sampler = DistributedSampler_wopadding(train_data)
    # sampler = KVSampler(train_data, batch_size=batch_size, num_replicas=2, rank=1, shuffle=True)
    dataloader = torch.utils.data.DataLoader(train_data,
                                             batch_size=32,
                                             sampler=sampler,
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
