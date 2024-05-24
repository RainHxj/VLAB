"""

Licensed under the MIT license.

Misc utilities
"""
import json
import random
import sys
import torch
import numpy as np
import copy
from utils.logger import LOGGER
import ipdb

class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def parse_with_config(parser):
    args = parser.parse_known_args()[0]
    if args.config is not None:

        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        
        dict_keys = [arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--') and "." in arg ]
        for dict_key in dict_keys:
            k1,k2 = dict_key.split(".")
            config_args[k1][k2] = getattr(args, dict_key)

        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    #del args.config
    return args


VE_ENT2IDX = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}

VE_IDX2ENT = {
    0: 'contradiction',
    1: 'entailment',
    2: 'neutral'
}


class Struct(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AttrDict(object):
    def __init__(self,
                config):
        
        if isinstance(config, dict):
            for key, value in config.items():
                self.__dict__[key] = value

        else:
            raise ValueError("First argument must be either a vocabulary size "
                            "(int) or the path to a pretrained model config "
                            "file (str)")

class OPTConfig(object):
    def __init__(self,
                 config):
        
        if isinstance(config, dict):
            for key, value in config.items():
                self.__dict__[key] = value

        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `OPTConfig` from a
           Python dictionary of parameters."""
        config = OPTConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `OPTConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

