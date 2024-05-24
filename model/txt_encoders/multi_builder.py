import torch
from torch import nn
from utils.logger import LOGGER
from model.head import MetaOPTHead, BertHead, DebertaV2Head
from utils.misc import AttrDict
import ipdb

def builder(cfg):
    
    multi_type=cfg["multi_type"]
    if multi_type == "bert":
        
        bert_config_path=cfg["bert_config"]
        pretrained_path=cfg["pretrained"]
        from model.txt_encoders.bert import BertModel, BertConfig
        bert_config = BertConfig.from_json_file(bert_config_path)
        
        ##
        multi_cross = cfg.get("multi_cross",1)
        ensemble_type = cfg.get("ensemble_type", None)
        mm_attn_type=cfg.get("mm_attn_type","cross_attn")
        ##
        bert_model = BertModel(bert_config, multi_cross=multi_cross, mm_attn_type=mm_attn_type, ensemble_type=ensemble_type)
        bert_weight = torch.load(pretrained_path, map_location='cpu')
        bert_weight = {k.replace('bert.','').replace('gamma','weight').replace('beta','bias') : v for k,v in bert_weight.items()}
        missing_keys, unexpected_keys = bert_model.load_state_dict(bert_weight, strict=False)
        LOGGER.info(f'unexpected_keys in bert_multiencoder: {unexpected_keys}')
        LOGGER.info(f'missing_keys in bert_multiencoder : {missing_keys}')

        head=BertHead(AttrDict(cfg),bert_model.embeddings.word_embeddings.weight)

        cls_head_weight = {}
        cls_head_weight['dense.weight']  = bert_weight['cls.predictions.transform.dense.weight']
        cls_head_weight['dense.bias']  = bert_weight['cls.predictions.transform.dense.bias']
        cls_head_weight['layernorm.weight'] = bert_weight['cls.predictions.transform.LayerNorm.weight' ]
        cls_head_weight['layernorm.bias'] =bert_weight['cls.predictions.transform.LayerNorm.bias']
        cls_head_weight['decoder.weight'] = bert_weight['cls.predictions.decoder.weight']
        cls_head_weight['decoder.bias'] = bert_weight['cls.predictions.bias']
        missing_keys, unexpected_keys = head.load_state_dict(cls_head_weight)
        LOGGER.info(f'unexpected_keys in bert_head : {unexpected_keys}')
        LOGGER.info(f'missing_keys in bert_head : {missing_keys}')
        del(bert_weight)
        del(cls_head_weight)

        return (bert_model, head)

    elif multi_type == "meta_opt":
        meta_config=cfg["meta_config"]
        pretrained_path=cfg["pretrained"]
        from .meta_opt import MetaOPTConfig, MetaOPTModel               
        metaoptconfig = MetaOPTConfig.from_pretrained(meta_config)
        metaoptweight = torch.load(pretrained_path)
        multimodal_dim = metaoptconfig.hidden_size
        metaoptconfig.use_cache = False
        multimodal_encoder = MetaOPTModel(metaoptconfig)
        metaoptweight =  {k.replace('model.','') : v for k,v in metaoptweight.items() }
        missing_keys, unexpected_keys = multimodal_encoder.load_state_dict(metaoptweight, strict=False)
        LOGGER.info(f'unexpected_keys in opt_multiencoder: {unexpected_keys}')
        LOGGER.info(f'missing_keys in opt_multiencoder: {missing_keys}')
        del(metaoptweight)

        head = MetaOPTHead(metaoptconfig, multimodal_encoder)
        
        return (multimodal_encoder, head)

    elif multi_type == "deberta_v2":
        config = cfg["deberta_config"]
        pretrained_path=cfg["pretrained"]
        from .deberta_v2 import DebertaV2Model, DebertaV2Config
        deberta_v2_config = DebertaV2Config.from_pretrained(config)
        deberta_v2_weight = torch.load(pretrained_path)
        multimodal_encoder = DebertaV2Model(deberta_v2_config)
        deberta_v2_weight =  {k.replace('deberta.','') : v for k,v in deberta_v2_weight.items() }

        head=DebertaV2Head(AttrDict(cfg), multimodal_encoder.embeddings.word_embeddings.weight)

        missing_keys, unexpected_keys = multimodal_encoder.load_state_dict(deberta_v2_weight, strict=False)
        LOGGER.info(f'unexpected_keys in opt_multiencoder: {unexpected_keys}')
        LOGGER.info(f'missing_keys in opt_multiencoder: {missing_keys}')

        cls_head_weight = {}
        cls_head_weight['dense.weight']  = deberta_v2_weight['lm_predictions.lm_head.dense.weight']
        cls_head_weight['dense.bias']  = deberta_v2_weight['lm_predictions.lm_head.dense.bias']
        cls_head_weight['layernorm.weight'] = deberta_v2_weight['lm_predictions.lm_head.LayerNorm.weight' ]
        cls_head_weight['layernorm.bias'] =deberta_v2_weight['lm_predictions.lm_head.LayerNorm.bias']
        cls_head_weight['decoder.bias'] = deberta_v2_weight['lm_predictions.lm_head.bias']
        missing_keys, unexpected_keys = head.load_state_dict(cls_head_weight, strict=False)
        LOGGER.info(f'unexpected_keys in deberta_v2_head : {unexpected_keys}')
        LOGGER.info(f'missing_keys in deberta_v2_head : {missing_keys}')

        del(deberta_v2_weight)

        return (multimodal_encoder, head)

    


