"""

Licensed under the MIT license.

OPT finetuning for video-Text Retrieval
"""
import argparse
import os
from os.path import join
from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import all_gather_list
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed

import torch.distributed as dist
import json
import ipdb
import math
import torch.nn.functional as F
from utils.distributed import get_rank, get_local_rank
from finetune.finetune_eval import evaluate_vqa, evaluate_caption, evaluate_retrieval



def build_evn(opts):

    local_rank = opts.local_rank
    torch.cuda.set_device(get_local_rank())
    dist.init_process_group(backend='nccl') 
    set_random_seed(opts.optimizer["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if opts.zero_shot:
        if get_rank() == 0:
            pass
        else:
            LOGGER.disabled = True
        return None, LOGGER

    if get_rank() == 0:
        save_training_meta(opts)  ###saving later
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        os.makedirs(join(opts.output_dir, 'results_test'),exist_ok=True)  
    else:
        LOGGER.disabled = True
        model_saver = NoOp()
    return model_saver,LOGGER


def load_checkpoint(opts):

    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint, map_location = 'cpu')
    elif opts.pretrain_dir:
        checkpoint_dir = os.path.join(opts.pretrain_dir,'ckpt')
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
        checkpoint_ls.sort()
        checkpoint_name = checkpoint_ls[-1]
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), map_location = 'cpu')

    else:
        checkpoint = {}

    if checkpoint != {}:
        if checkpoint.get("video_frame_embedding", None) is not None:

            w = checkpoint["video_frame_embedding"]
            w = F.interpolate(w.permute(0, 2, 1), opts.video_cfg["sample_num"], mode='linear').permute(0, 2, 1)
            checkpoint["video_frame_embedding"] = w

        if opts.video_cfg["resolution"]!=224:
            
            backbone_id = opts.multi_encoder.get("multi_cross", 1)
            for i in range(backbone_id):
                print("========{}=====".format(i))
                if i == 0:
                    w = checkpoint['opt.vision_text_model.models.{}.visual.pos_embed'.format(i)].squeeze(0)
                else:
                    w = checkpoint['opt.vision_text_model.models.{}.visual.positional_embedding'.format(i)]
                src_cls = w[0:1]
                src_oth = w[1:]
                grid = opts.video_cfg["resolution"] // 14
                dim = w.size(-1)
                import math
                raw_grid = int(math.sqrt(w.size(-2)))
                src_oth = F.interpolate(src_oth.reshape(raw_grid,raw_grid,dim).permute(2,0,1).unsqueeze(0),(grid, grid),mode='bilinear') # any
                # src_oth = F.interpolate(src_oth.reshape(16,16,dim).permute(2,0,1).unsqueeze(0),(24, 24),mode='bilinear') # 336
                #src_oth = F.interpolate(src_oth.reshape(16,16,dim).permute(2,0,1).unsqueeze(0),(28,28),mode='bilinear') # 392
                # src_oth = F.interpolate(src_oth.reshape(16,16,dim).permute(2,0,1).unsqueeze(0),(34,34),mode='bilinear') # 476
                src_oth = src_oth[0].permute(1,2,0).reshape(-1,dim)
                tgt = torch.cat((src_cls,src_oth),dim=0)
                if i == 0:
                    checkpoint['opt.vision_text_model.models.{}.visual.pos_embed'.format(i)] = tgt.unsqueeze(0)
                else:
                    checkpoint['opt.vision_text_model.models.{}.visual.positional_embedding'.format(i)] = tgt
                print("xxx{}xxx{}".format(i,tgt.shape))

            for key, value in checkpoint.items():
                if 'positional_embedding' in key and 'video_adapter' in key:
                    w = checkpoint[key]
                    src_cls = w[0:1]
                    src_oth = w[1:]
                    resolution = opts.video_cfg["resolution"]
                    grid = resolution // 14
                    dim = w.size(-1)
                    import math
                    raw_grid = int(math.sqrt(w.size(-2)))
                    src_oth = F.interpolate(src_oth.reshape(raw_grid,raw_grid,dim).permute(2,0,1).unsqueeze(0), (grid, grid), mode='bilinear') # any
                    src_oth = src_oth[0].permute(1,2,0).reshape(-1,dim)
                    tgt = torch.cat((src_cls,src_oth),dim=0)
                    checkpoint[key] = tgt
    return checkpoint




def evaluate_and_logger(model, test_loader, opts, LOGGER, global_step=0, metric_logger_dict=None):

    if hasattr(model.module, 'video_frame_embedding'):
        video_frame_embedding = model.module.video_frame_embedding
        cp_video_frame_embedding = model.module.video_frame_embedding.clone().detach()
        if 'test_sample_num' in opts.video_cfg:
            cp_video_frame_embedding = F.interpolate(cp_video_frame_embedding.permute(0, 2, 1), opts.video_cfg["test_sample_num"], mode='linear').permute(0, 2, 1)
        model.module.video_frame_embedding = torch.nn.Parameter(cp_video_frame_embedding)

    if opts.finetune_task_name=="caption":
        task_metric_name="CIDEr"
        eval_log, results_2m = evaluate_caption(model, test_loader, opts, LOGGER)
    elif opts.finetune_task_name=="vqa":
        task_metric_name="accuracy"
        eval_log, results_2m = evaluate_vqa(model, test_loader,opts,global_step,LOGGER)
    elif opts.finetune_task_name=="retrieval":
        task_metric_name="video_rsum"
        eval_log= evaluate_retrieval(model, test_loader,LOGGER)
    else:
        raise NotImplementedError
    
    if hasattr(model.module, 'video_frame_embedding'):
        model.module.video_frame_embedding = video_frame_embedding

    best_step=-1

    if dist.get_rank() == 0:
        if metric_logger_dict is not None:
            for eval_name, metric in eval_log.items():
                metric_logger_dict[eval_name][str(global_step)] = metric
                if ('best_step' not in metric_logger_dict[eval_name]) or \
                        (metric[task_metric_name] >= metric_logger_dict[eval_name]['best_value']):
                    metric_logger_dict[eval_name]['best_step'] = global_step
                    metric_logger_dict[eval_name]['best_value'] = metric[task_metric_name]

                best_step = metric_logger_dict[eval_name]['best_step']

                LOGGER.info(f"======evaluation--{eval_name}====history best step: {best_step}==\n")
                LOGGER.info(metric_logger_dict[eval_name][str(best_step)])

                LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--==========\n")
                LOGGER.info(metric)
        else:
            for eval_name, metric in eval_log.items():
                LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--==========\n")
                LOGGER.info(metric)

        result_folder = join(opts.output_dir, 'results_test')
        if opts.finetune_task_name=="caption" and not opts.zero_shot:
            json.dump(results_2m, open(join(result_folder, 'step_{}_2m.json'.format(global_step)), 'w'))
        
        return best_step
    else:
        return -1 

