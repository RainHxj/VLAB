"""

Licensed under the MIT license.

VQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import TxtVideoDataset, TxtMapper
import json
import os
import string 
punctuation = string.punctuation
from pytorch_pretrained_bert import BertTokenizer
import random
from utils.logger import LOGGER
from .data import TxtVideoDataset
import copy


class TxtMapperVQA(TxtMapper):
    def __init__(self, text_path, return_all_captions, tokenizer="bert_tokenizer", max_txt_len=30, data_type= '', is_train=False):
        super().__init__(text_path, return_all_captions, tokenizer, max_txt_len, data_type)

        self.is_train = is_train

    def __getitem__(self, file_name, id_, txt_reader):
        
        info = self.get_info(file_name, id_, txt_reader)
        text = info["text"]
        answer = info["answer"]

        output_multi = self.get_single_txt(text, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos)
        
        output_answer = self.get_single_txt(answer, self.multi_encoder_tokenizer, self.multi_encoder_bos, self.multi_encoder_eos)

        return {"answer_tokens": output_answer,"questions": text, "question_tokens":output_multi,"answers":answer, "video_clip":None}


class TxtVideoDatasetVQA(TxtVideoDataset):
    def __init__(self, video_path, txt_path, key_path, video_cfg, num_readers=12, tokenizer="bert_tokenizer", is_train=True, data_type="video", return_all_captions=False, **kwargs):

        super().__init__(video_path, txt_path, key_path, video_cfg, num_readers, tokenizer, is_train, data_type, return_all_captions, **kwargs)

        self.txt_mapper = TxtMapperVQA(txt_path, return_all_captions, tokenizer=tokenizer, is_train=is_train, **kwargs)

    def __getitem__(self, i):
        
        id_ = self.ids[i]
        output = self.txt_mapper.__getitem__(id_["filename"], i, self.txt_reader)
        question_tokens = output["question_tokens"]
        answer_tokens = output["answer_tokens"]
        answers=output["answers"]
        questions=output["questions"]
        if not self.return_all_captions or not isinstance(txt_tokens, list):
            id_txt = id_['filename']
        else:
            id_txt = [id_['filename']]*len(txt_tokens)
        
        video_clip=output["video_clip"]
        
        video_pixels = self.video_mapper.__getitem__(id_['filename'], video_clip, self.video_reader)

        if video_pixels is None: ###wrong img/video and needs to resample 
            resample_idx = random.choice(self.idx)
            LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
            return self.__getitem__(resample_idx)
        
        return id_["filename"], questions, answers, question_tokens, answer_tokens, video_pixels, id_txt



def txtvideo_vqa_collate(inputs):
    
    (ids, questions, answers, question_tokens, answer_tokens, video_pixels,id_txts) = map(list, unzip(inputs))
    question_tokens = torch.stack(question_tokens, dim = 0)
    answer_tokens = torch.stack(answer_tokens, dim = 0)
    video_pixels = torch.stack(video_pixels, dim=0)
    batch2m = {'ids': ids, "questions": questions, 'answers':answers,
             'question_tokens': question_tokens, "answer_tokens":answer_tokens, 'video_pixels': video_pixels, 'ids_txt':id_txts}
    batch={}
    batch["batch_2m"] = batch2m
    return batch
