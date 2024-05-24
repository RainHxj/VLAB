from utils.logger import LOGGER
import pickle
import numpy as np
import webdataset as wds
from torchvision import transforms
from torchvision.transforms.transforms import (
    CenterCrop,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
)
from PIL import Image
import random
import torch
from os.path import join
import json
from data.tokenizer import build_tokenizer
import string
import os


class TxtMapper(object):

    def __init__(
        self,
        return_all_captions,
        tokenizer: dict = None,
        max_txt_len=30,
        data_type="",
    ):

        self.data_type = data_type
        self.max_txt_len = max_txt_len

        meta = json.load(open(join('./data/meta.json')))
        self.cls_ = meta["CLS"]
        self.sep = meta["SEP"]
        self.mask = meta["MASK"]
        self.v_range = meta["v_range"]
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
        # text = self.clean(text)
        txt_tokens = tokenizer(text)

        txt_tokens = txt_tokens[:self.max_txt_len]

        txt_tokens = [bos] + txt_tokens + [eos]

        txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)

        output = torch.zeros(self.max_txt_len + 2, dtype=torch.long)
        output[:len(txt_tokens)] = txt_tokens
        return output

    def clean(self, text):
        """remove duplicate spaces, lower and remove punctuations """
        text = " ".join([i for i in text.split(" ") if i != ""])
        text = text.lower()
        for i in self.punctuations:
            text = text.replace(i, "")
        return text


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

    def process(self, video):
        sample_num = self.sample_num
        video_list = []
        # time_begin = time.time()
        frames = pickle.loads(video)
        # print("pickle load time is ", (time.time()-time_begin)*1000)
        # sample_idx = random.sample(list(range(frames.shape[0])), sample_num)
        replace = len(frames) < sample_num
        sample_idx = sorted(np.random.choice(range(len(frames)), sample_num, replace=replace))

        for ind in sample_idx:
            frame = Image.fromarray(frames[ind])
            frame = transforms.ToTensor()(frame)  ## frame: 3XhXw

            video_list.append(frame.unsqueeze(0))
        video_pixels = torch.cat(video_list, dim=0)  ### nX3xHxW

        video_pixels = self.transforms(video_pixels)

        return video_pixels


class TxtVideoDataset:

    def __init__(
        self,
        video_cfg,
        tokenizer: dict = None,
        is_train=True,
        return_all_captions=False,
        **kwargs,
    ):

        if tokenizer is None:
            tokenizer = {
                "txt_encoder": "clip_tokenizer",
                "multi_encoder": "bert_tokenizer"
            }
        self.txt_mapper = TxtMapper(return_all_captions,
                                    tokenizer=tokenizer,
                                    **kwargs)
        self.video_mapper = VideoMapper(video_cfg, is_train=is_train)

        self.return_all_captions = return_all_captions

    def process(self, item):
        output = self.txt_mapper.process(
            # item[1]
            str(item[1], 'UTF-8'))
        txt_tokens = output["txt_tokens"]
        multi_tokens = output["multi_tokens"]
        filename = item[2]
        if not self.return_all_captions or not isinstance(txt_tokens, list):
            id_txt = filename
        else:
            id_txt = [filename] * len(txt_tokens)

        video_pixels = self.video_mapper.process(item[0])
        return filename, txt_tokens, multi_tokens, video_pixels, id_txt


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    print(exn)
    return True


def WebvidFrameDataset(data_root,
                       video_cfg,
                       tokenizer,
                       is_train,
                       total_num=300,
                       test=False):
    # video_cfg = {"sample_num": 1,
    #              "resolution": 224,
    #              "mean": [0.48145466, 0.4578275, 0.40821073],
    #              "std": [0.26862954, 0.26130258, 0.27577711]}
    # pipeline_func=partial(pipeline_func,data_type=data_type)

    # tokenizer = {
    #                     "txt_encoder": "clip_tokenizer",
    #                     "multi_encoder": "bert_tokenizer"
    #                 }
    data = TxtVideoDataset(video_cfg, tokenizer=tokenizer, is_train=True)
    # data = LaionDataset(video_cfg=video_cfg, caption="caption", is_train=True, tokenizer=tokenizee)
    # data_root = data_root+"/{000001..000251}.tar"
    data_root = [
        os.path.join(data_root, item) for item in os.listdir(data_root)
        if item.endswith('tar')
    ]

    dataset = wds.WebDataset(data_root, shardshuffle=True, resampled=True).shuffle(100).\
        to_tuple("mp4", "txt", "__key__", handler=warn_and_continue).map(data.process,handler=warn_and_continue)
    # with_epoch(1000).with_length(1000)
    # loader = DataLoader(dataset, batch_size=batch_size, prefetch_factor=2,
    #                     num_workers=n_workers, pin_memory=True,
    #                     drop_last=True)

    return dataset
