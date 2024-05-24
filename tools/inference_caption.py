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

# from data import (PrefetchLoader, TxtMapper,VideoMapper)

from data.data import TxtVideoDataset, txtvideo_collate
from data.coco_data import RetDataset, txtvideo_collate_coco
from model.down_streams.video_caption_mul import OPTForVideoCaptionEnsemble
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
    

    checkpoints=[
        "./output/experiments/cbconfig/pretrain-cc4m-cc12m-clip_large-20-20_64gpus/msrvtt_caption_1node/ckpt/model_step_5000.pt",
        "./output/experiments/cbconfig/pretrain-cc4m-cc12m-clip_large-20-20_64gpus/msrvtt_caption_1node_allcaptions_scst/ckpt/model_step_19190.pt"]

    model_saver, LOGGER = build_evn(opts)
    
    checkpoints = []
    for chs in checkpoints:
        checkpoint = torch.load(chs, map_location = 'cpu')
        checkpoints.append(checkpoint)


    train_dataloader, test_loader, train_dataset = build_dataset(opts)
    
    opts.optimizer["num_train_steps"]=int((len(train_dataset)/opts.data_cfg["train_total_batch_size"])*opts.data_cfg["epoch"])


    if opts.finetune_task_name=="retrieval":
        model = VLABForVideoRetrieval.from_pretrained(opts, checkpoint)
    elif opts.finetune_task_name=="vqa":
        model = VLABForVideoQA.from_pretrained(opts, checkpoint)
    elif opts.finetune_task_name=="caption":
        models=[]
        for checkpoint in checkpoints:
            model = OPTForVideoCaptionEnsemble.from_pretrained(opts, checkpoint)
            models.append(model)
    else:
        raise NotImplementedError
    
    ddp_models=[]
    for model in models:
        device = torch.device("cuda", get_local_rank())
        model.to(device)

        # Prepare optimizer
        optimizer = build_optimizer(model, AttrDict(opts.optimizer))
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank(), find_unused_parameters=False)
        ddp_models.append(model)

    evaluate_caption(ddp_models, test_loader, opts, LOGGER)





"""

Licensed under the MIT license.

OPT finetuning for video-Text Retrieval
"""
import argparse
import os
from os.path import  join
from time import time
import torch
from tqdm import tqdm
from utils.distributed import all_gather_list, ddp_allgather
from cococaption.pycocoevalcap.eval import COCOEvalCap
from cococaption.pycocotools.coco import COCO
from utils.distributed import all_gather_list
import torch.distributed as dist
from utils.misc import NoOp
import math
import torch.nn.functional as F
import ipdb
import json


def decode_sequence(opts, seq):
    vocab = [line.strip() for line in open(opts.caption_cfg["toker"])]
    i2v = {i:vocab[i]  for i in range(len(vocab))}
    eos_token = opts.caption_cfg['eos_token']
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t].item()
            if ix == eos_token:
                break
            words.append(i2v[ix])
        sent = ' '.join(words)
        sents.append(sent.replace(' ##', ''))
    return sents

def compute_caption_metric(results, opts):

    coco = COCO(opts.data_cfg["ann_path"])
    cocoRes = coco.loadRes(results)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    metric = cocoEval.eval
    metric = {k: round(v*100,2)  for k,v in metric.items()}
    print(metric)
    return metric


@torch.no_grad()
def evaluate_caption(models, eval_loader, opts, LOGGER):
    st = time()
    LOGGER.info("start running Video Caption evaluation ...")

    for model in models:
        model.eval()

    results_2m = []

    pbar = tqdm(total=len(eval_loader))

    for batch in eval_loader:
        
        ids_2m = batch['batch_2m']['ids']
        sample_num_2m = len(ids_2m)
        sents,_ = generate_each_model(batch, models)

        if sample_num_2m > 0 :

            if opts.multi_encoder["multi_type"] == "bert":
                sents = decode_sequence(opts, sents.data)
            elif opts.multi_encoder["multi_type"] == "deberta_v2":
                from transformers import DebertaV2Tokenizer

                tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
                res=[]
                for i in range(sents.data.shape[0]):
                    res.append(tokenizer.decode(sents.data[i]))
                sents=res
            
            for  i in range(sample_num_2m):
                result = {'video_id':ids_2m[i], 'caption': sents[i]}
                results_2m.append(result)

        pbar.update(1)
    pbar.close()
    all_results_2m = [i for results_2m in all_gather_list(results_2m)  for i in results_2m]


    if dist.get_rank() != 0:
        return {},{}

    metric_2m = compute_caption_metric(all_results_2m, opts)   
    eval_log = {'metric_2m': metric_2m }
    tot_time = time()-st

    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds")
    return eval_log, all_results_2m


def generate_each_model(batch, models):

    max_generation_len=30
    eos_token=102
    batch_size = batch['batch_2m']['video_pixels'].size(0)
    sents = torch.zeros((batch_size, max_generation_len), dtype=torch.long).fill_(eos_token).cuda()
    logprobs = torch.zeros(batch_size, max_generation_len).cuda()
    unfinished = torch.ones(batch_size, dtype=torch.bool).cuda()

    state = None
    video_feat=False
    batch1= batch
    batch2= batch
    for t in range(max_generation_len):

        logprobs_t1, batch1 = models[0](batch1, unfinished, state, video_feat=video_feat)

        logprobs_t2, batch2 = models[1](batch2, unfinished, state, video_feat=video_feat)
        
        logprobs_t = logprobs_t1 + logprobs_t2
       
        video_feat=True

        logP_t, wt = torch.max(logprobs_t, 1)

        wt = wt.view(-1).long()
        unfinished = unfinished * (wt != eos_token)
        wt = wt * unfinished.type_as(wt) + (1 - unfinished.type_as(wt)) * eos_token

        sents[:,t] = wt
        logprobs[:,t] = logP_t.view(-1)
        state = wt.unsqueeze(1) if state is None else torch.cat((state,wt.unsqueeze(1)),dim=1)

        if unfinished.sum() == 0:
            break
    
    return sents, logprobs


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
