"""

Licensed under the MIT license.

saving utilities
"""
import json
import os
from os.path import abspath, dirname, exists, join
import subprocess

import torch

from utils.logger import LOGGER


def save_training_meta(args):

    if not exists(args.output_dir):
        os.makedirs(join(args.output_dir, 'log'))
        os.makedirs(join(args.output_dir, 'ckpt'))

    with open(join(args.output_dir, 'log', 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    # model_cfg = json.load(open(args.model_cfg))
    # with open(join(args.output_dir, 'log', 'model.json'), 'w') as writer:
    #     json.dump(model_cfg, writer, indent=4)



class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, best_step=-1, optimizer=None, last=False):
        ###remove previous model
        previous_state = [i  for i in os.listdir(self.output_dir) if i.startswith('model')]
        for p in previous_state:
            if str(best_step) in p:
                pass
            else:
                os.remove(os.path.join(self.output_dir,p))
        if last:
            output_model_file = join(self.output_dir,
                        f"{self.prefix}_{step}_last.{self.suffix}")
        else:
            output_model_file = join(self.output_dir,
                                    f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.module.state_dict().items()}
        torch.save(state_dict, output_model_file)

        if optimizer is not None:
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
            previous_state = [i  for i in os.listdir(self.output_dir) if i.startswith('optimizer')]
            for p in previous_state:
                os.remove(os.path.join(self.output_dir,p))
            torch.save(optimizer.state_dict(), f'{self.output_dir}/optimizer_step_{step}.pt')
