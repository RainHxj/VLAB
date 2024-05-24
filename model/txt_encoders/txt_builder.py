import torch
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from utils.logger import LOGGER

def builder(cfg):

    text_type=cfg["text_type"]

    if text_type == "bert":
    
        bert_config_path=cfg["bert_config"]
        pretrained_path=cfg["pretrained"]
        from model.txt_encoders.bert import BertModel, BertConfig
        bert_config = BertConfig.from_json_file(bert_config_path)
        bert_model = BertModel(bert_config)
        bert_weight = torch.load(pretrained_path, map_location='cpu')
        bert_weight = {k.replace('bert.','').replace('gamma','weight').replace('beta','bias') : v for k,v in bert_weight.items()}
        missing_keys, unexpected_keys = bert_model.load_state_dict(bert_weight, strict=False)
        LOGGER.info(f'unexpected_keys in bert_multiencoder: {unexpected_keys}')
        LOGGER.info(f'missing_keys in bert_multiencoder : {missing_keys}')

        return bert_model