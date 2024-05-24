"""
VLAB fine-tuning.
"""
import argparse
import os
from os.path import join
from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from tqdm import tqdm

# from data import (PrefetchLoader, TxtMapper,VideoMapper)

from data.data import TxtVideoDataset, txtvideo_collate
from data.coco_data import RetDataset, txtvideo_collate_coco
from model.down_streams.video_caption import VLABForVideoCaption
from model.down_streams.video_qa import VLABForVideoQA
from model.down_streams.video_retrieval import VLABForVideoRetrieval
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import all_gather_list, barrier
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import ipdb
import math
import torch.nn.functional as F
from collections import defaultdict
from utils.distributed import DistributedSampler_wopadding
from data.data import worker_init_fn
from utils.misc import AttrDict

from finetune.util import build_evn, load_checkpoint, evaluate_and_logger
from finetune.finetune_data import build_dataset
from utils.distributed import get_rank, get_local_rank

def main(opts):

    model_saver, LOGGER = build_evn(opts)
    checkpoint = load_checkpoint(opts)
    train_dataloader, test_loader, train_dataset = build_dataset(opts)
    
    opts.optimizer["num_train_steps"]=int((len(train_dataset)/opts.data_cfg["train_total_batch_size"])*opts.data_cfg["epoch"])

    if opts.finetune_task_name=="retrieval":
        model = VLABForVideoRetrieval.from_pretrained(opts, checkpoint)
    elif opts.finetune_task_name=="vqa":
        model = VLABForVideoQA.from_pretrained(opts, checkpoint)
    elif opts.finetune_task_name=="caption":
        model = VLABForVideoCaption.from_pretrained(opts, checkpoint)
    else:
        raise NotImplementedError

    device = torch.device("cuda", get_local_rank())
    model.to(device)

    # Prepare optimizer
    optimizer = build_optimizer(model, AttrDict(opts.optimizer))
    model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank(), find_unused_parameters=False)
    model.train()
    # model._set_static_graph()
    ## train parameters gradient_accumulation_steps
    gradient_accumulation_steps = opts.optimizer["gradient_accumulation_steps"]
    grad_norm_init = opts.optimizer["grad_norm"]


    n_epoch = 0
    evaluate_epoch = opts.schedule["evaluate_epoch"]
    EPOCHS = opts.data_cfg["epoch"]
    metric_logger_dict = defaultdict(dict)
    global_step = 0
    n_gpu = dist.get_world_size()
    LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset) )
    LOGGER.info("  Train step = %d", opts.optimizer["num_train_steps"])
    LOGGER.info("  Accumulate steps = %d", gradient_accumulation_steps)
    LOGGER.info("  Num epochs = %d", EPOCHS)

    loss_moving_averagetors ={}

    if opts.finetune_task_name=="caption":
        task_metric_name="CIDEr"
    elif opts.finetune_task_name=="vqa":
        task_metric_name="accuracy"
    elif opts.finetune_task_name=="retrieval":
        task_metric_name="video_r1"
    else:
        raise NotImplementedError
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    loss_moving_averagetors ={}

    if opts.zero_shot or opts.first_eval:
        evaluate_and_logger(model, test_loader, opts, LOGGER)
        if opts.zero_shot:
            return 

    for epoch in range(EPOCHS):
        
        if get_rank()==0:
            loop = tqdm(total=len(train_dataloader))
        else:
            loop = NoOp()

        for step, batch in enumerate(train_dataloader):

            loss_dict = model(batch, compute_loss=True)
            loss = sum(list(loss_dict.values()))
            loss_dict['total_loss'] = loss
            
            delay_unscale = (step+1) % gradient_accumulation_steps != 0
            with torch.autocast(device_type='cuda'):
                loss.backward()
                
            for k in loss_dict.keys():
                if not k in loss_moving_averagetors:
                ### first time initialize 
                    loss_moving_averagetors[f'loss/{k}'] = RunningMeter()
            
            for k,v in loss_dict.items():
                loss_moving_averagetors[f'loss/{k}'](v)

            if (step + 1) % gradient_accumulation_steps == 0:

                global_step += 1

                # learning rate scheduling
                lr_ratio = get_lr_sched(global_step, AttrDict(opts.optimizer))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['init_lr'] * lr_ratio

                TB_LOGGER.add_scalar('lr_ratio', lr_ratio, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.log_scaler_dict({name: averagetor.val
                                        for name, averagetor in loss_moving_averagetors.items()
                                        if averagetor.val is not None})

                if global_step % 100 == 0:    
                    LOGGER.info({name : averagetor.val for name, averagetor in loss_moving_averagetors.items()})                                   
   
                # update model params
                if grad_norm_init != -1:
                    grad_norm = clip_grad_norm_(model.parameters(),
                                                grad_norm_init)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()

                TB_LOGGER.step()
                if get_rank()==0:
                    loop.update(1)
                    loop.set_description("Epoch [{}/{} ]".format(n_epoch,EPOCHS))
        barrier()
        loop.close()
        n_epoch += 1
        if n_epoch % evaluate_epoch==0:
            best_step = evaluate_and_logger(model, test_loader, opts, LOGGER, global_step, metric_logger_dict)
            model_saver.save(model, global_step, best_step)
        LOGGER.info(f"finished {n_epoch} epochs")
        train_dataloader.sampler.set_epoch(n_epoch)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained MLM")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model "
                             "checkpoints will be written.")

    parser.add_argument('--config', help='JSON config files')

    parser.add_argument('--zero_shot', action='store_true',
                        help="Always run full evaluation during training")

    parser.add_argument('--pretrain_dir', type=str, default='',
                        help="random seed for initialization")

    parser.add_argument('--return_all_text', type=bool, default=False,
                        help="random seed for initialization")

    parser.add_argument('--first_eval', action='store_true',
                        help="Always run full evaluation during training")   
    
    parser.add_argument('--finetune_task', type=str, default='',
                        help="finetune task name")
    ## vqa
    parser.add_argument('--submit_file', action='store_true', default=False,
                        help="if submit to server for evaluation")
    ## config vision_encoder
    parser.add_argument("--vision_encoder.frozen_vision", action = "store_true",help="frozen vision")
    parser.add_argument("--vision_encoder.diff_view", action = "store_true",help="different view for different clip")


    args = parse_with_config(parser)


    main(args)
