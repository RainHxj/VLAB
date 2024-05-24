"""

Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW
import ipdb

# def build_optimizer(model, opts):
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer
#                     if not any(nd in n for nd in no_decay)],
#          'weight_decay': opts.weight_decay},
#         {'params': [p for n, p in param_optimizer
#                     if any(nd in n for nd in no_decay)],
#          'weight_decay': 0.0}
#     ]

#     # currently Adam only
#     if opts.optim == 'adam':
#         OptimCls = Adam
#     elif opts.optim == 'adamax':
#         OptimCls = Adamax
#     elif opts.optim == 'adamw':
#         OptimCls = AdamW
#     else:
#         raise ValueError('invalid optimizer')
#     optimizer = OptimCls(optimizer_grouped_parameters,
#                          lr=opts.learning_rate, betas=opts.betas)
#     return optimizer

def build_optimizer(model, opts):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    basic_params = []
    basic_params_name = []
    basic_params_no_decay = []
    vision_params = []
    vision_params_name = []
    vision_params_no_decay = []


    for k, v in model.named_parameters():
        if  'vision' in k and  not any(nd in k for nd in no_decay):
            vision_params.append(v)
            vision_params_name.append(k)
        elif 'vision' in k and  any(nd in k for nd in no_decay):
            vision_params_no_decay.append(v)
            vision_params_name.append(k)
        elif not any(nd in k for nd in no_decay):
            basic_params.append(v)
            basic_params_name.append(k)
        elif any(nd in k for nd in no_decay):
            basic_params_no_decay.append(v)
            basic_params_name.append(k)
    
    opts.vision_lr = getattr(opts,'vision_lr',5e-7)
    optimizer_grouped_parameters = [
        {'params': basic_params, 'weight_decay': opts.weight_decay, 'lr': opts.learning_rate},
        {'params': basic_params_no_decay, 'weight_decay': 0.0, 'lr': opts.learning_rate},
        {'params': vision_params, 'weight_decay': opts.weight_decay, 'lr': opts.vision_lr},
        {'params': vision_params_no_decay, 'weight_decay': 0.0, 'lr': opts.vision_lr}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')

    for i in optimizer_grouped_parameters:
        i['init_lr'] = i['lr']
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)

    optimizer.basic_lr = opts.learning_rate
    optimizer.vision_lr = opts.vision_lr

    return optimizer







def build_optimizer_for_VQA(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = 'vqa_head'
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and not new_param in n],
         'weight_decay': opts.weight_decay,
         'lr': opts.learning_rate},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and not new_param in n],
         'weight_decay': 0.0,
         'lr': opts.learning_rate},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and new_param in n],
         'weight_decay': opts.weight_decay,
         'lr': opts.learning_rate*5},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and new_param in n],
         'weight_decay': 0.0,
         'lr': opts.learning_rate*5}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                          betas=opts.betas)
    return optimizer
