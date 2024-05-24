"""

Licensed under the MIT license.

OPT pre-training
"""
import argparse
import os
from os.path import join
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist

from model.pretrain import OPTForPretraining
from model.cb_pretrain import VLABForPretraining
from optim import get_lr_sched
from optim.misc import build_optimizer
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import get_rank, get_local_rank
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_random_seed
from pretrain_data import create_train_dataloaders, create_val_dataloaders
from pretrain_validate import validate
from data import  MetaLoader, PrefetchLoader 
from utils.misc import AttrDict
from tqdm import tqdm
import wandb
import time
import ipdb

def main(opts):

    torch.cuda.set_device(get_local_rank())
    dist.init_process_group(backend='nccl') 
    set_random_seed(opts.optimizer["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    """ resume """
    if opts.resume:
        ckpt_dir = os.path.join(opts.output_dir,'ckpt')
        previous_optimizer_state = [i  for i in os.listdir(ckpt_dir) if i.startswith('optimizer')]
        assert len(previous_optimizer_state)==1
        previous_optimizer_state = previous_optimizer_state[0]
        previous_model_state = previous_optimizer_state.replace('optimizer','model')
        previous_step = int(previous_model_state.split('.')[0].split('_')[-1])
        previous_optimizer_state = os.path.join(ckpt_dir, previous_optimizer_state)
        previous_model_state = os.path.join(ckpt_dir, previous_model_state)
        
        assert os.path.exists(previous_optimizer_state) and os.path.exists(previous_model_state)
        LOGGER.info("choose previous model: {}".format(previous_model_state))
        LOGGER.info("choose previous optimizer: {}".format(previous_optimizer_state))
    else:
        previous_model_state = ''
        previous_optimizer_state = ''
        previous_step = 0

    # dataloader
    train_dataloaders = create_train_dataloaders(opts.data_cfg, opts)
    tmp=0
    for k,v in train_dataloaders.items():
        step = v[-2]
        tmp+=step
    num_train_steps = tmp
    opts.optimizer["num_train_steps"]=num_train_steps

    if opts.schedule["use_validate"]:
        val_dataloaders= create_val_dataloaders(opts.data_cfg, opts)

    n_gpu = dist.get_world_size()
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.optimizer["gradient_accumulation_steps"],
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    if get_rank() == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=num_train_steps,initial=previous_step)
        TB_LOGGER.set_step(previous_step)
        model_saver = ModelSaver(join(args.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        if opts.wandb:
            wandb.init(job_type="training",
                mode="online",
                project="VLP", 
                entity="rainhxj",
                name=opts.output_dir.split("/")[-1]
            )
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
   
    # Prepare model
    if previous_model_state:
        checkpoint = torch.load(previous_model_state, map_location = 'cpu')
        LOGGER.info("Load Checkpoint {}".format(previous_model_state))
    elif opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint, map_location = 'cpu')
                
        LOGGER.info("Load Checkpoint {}".format(opts.checkpoint))
    else:
        checkpoint = {}
    ### add video_dim to model_cfg
    # model = OPTForPretraining.from_pretrained(
    #     opts, checkpoint)
    model = VLABForPretraining.from_pretrained(
        opts, checkpoint)



    del(checkpoint)
    #ipdb.set_trace()
    device = torch.device("cuda", get_local_rank())
    model.to(device)

    # Prepare optimizer
    LOGGER.info("build_optimizer")
    optimizer = build_optimizer(model, AttrDict(opts.optimizer))
    if previous_optimizer_state:
        checkpoint_optimizer = torch.load(previous_optimizer_state, map_location = 'cpu')
        optimizer.load_state_dict(checkpoint_optimizer)
        del(checkpoint_optimizer)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank(), find_unused_parameters=False)
    model.train()

    global_step = previous_step
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Accumulate steps = %d", opts.optimizer["gradient_accumulation_steps"])
    for k,v in train_dataloaders.items():
        LOGGER.info("step of {} = {}".format(k.split("--")[-1], v[-2]))
    LOGGER.info("  vision_lr = %f", optimizer.vision_lr)
    LOGGER.info("  basic_lr = %f", optimizer.basic_lr)
    LOGGER.info("  Num steps = %d", num_train_steps)
    LOGGER.info("  Start step = %d", previous_step)


    # to compute training statistics
    loss_moving_averagetors ={}
    grad_norm = 0
    optimizer.zero_grad()
    optimizer.step()
    
    ## train parameters 
    gradient_accumulation_steps = opts.optimizer["gradient_accumulation_steps"]
    grad_norm_init = opts.optimizer["grad_norm"]
    # num_train_steps = opts.optimizer["num_train_steps"]
    use_validate = opts.schedule["use_validate"]
    valid_steps = opts.schedule["valid_steps"]

    for step, (name, batch) in enumerate(meta_loader):

        task = name.split('--')[0]
       
        loss_dict = model(batch, task=task, compute_loss=True)

        loss = sum(list(loss_dict.values()))
        loss_dict['total_loss'] = loss
        loss_dict = {k:v.item() for k,v in loss_dict.items()}

        delay_unscale = (step+1) % gradient_accumulation_steps != 0

        with torch.autocast(device_type='cuda'):
            loss.backward()
        
        if not name in loss_moving_averagetors:
            ### first time initialize 
            for k in loss_dict.keys():
                loss_moving_averagetors[f'loss_{name}/{k}'] = RunningMeter()
        
        for k,v in loss_dict.items():
            loss_moving_averagetors[f'loss_{name}/{k}'](v)


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

            if global_step % 200 == 0:    
                LOGGER.info({name : averagetor.val for name, averagetor in loss_moving_averagetors.items()})   
            if global_step % 100 == 0 and get_rank()==0 and opts.wandb:
                wandb.log({name : averagetor.val for name, averagetor in loss_moving_averagetors.items()}, step=global_step)                                 

            # update model params
            if grad_norm_init != -1:
                grad_norm = clip_grad_norm_(model.parameters(),
                                            grad_norm_init)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

        if global_step % valid_steps == 0:
            LOGGER.info(f'Step {global_step}: start validation')

            if use_validate:
                validate(model, val_dataloaders, opts, global_step = global_step)
                
            model_saver.save(model, global_step, optimizer=optimizer)

        TB_LOGGER.step()

        if global_step >= num_train_steps:
            model_saver.save(model, global_step, optimizer=optimizer, last=True)
            break
        
    # if opts.use_validate:
    #     validate(model, val_dataloaders, opts, global_step = global_step)
    #     model_saver.save(model, global_step, optimizer)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")
    parser.add_argument('--resume', action = 'store_true', help='use txt out')
    parser.add_argument("--output_dir", default=None, type=str, help="save model path")
    # can use config files
    parser.add_argument('--config', required=True, help='JSON config files')
    parser.add_argument('--wandb', action = 'store_true', help='use wandb')

    ## config vision_encoder
    parser.add_argument("--vision_encoder.frozen_vision", action = "store_true",help="frozen vision")

    args = parse_with_config(parser)
    main(args)
