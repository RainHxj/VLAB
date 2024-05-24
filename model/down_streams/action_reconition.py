"""

Licensed under the MIT license.

OPT for VQA model
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm as FusedLayerNorm
import torch

from .pretrain import OPTForPretraining, OPTModel, pool_for_contra



class CLS_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_size, config.class_num)
    def forward(self, cls_token):
        return self.linear2(self.activation(self.linear1(cls_token)))

class OPTForActionRecognition(OPTForPretraining):
    def __init__(self,config, video_cfg, audio_cfg):
        super().__init__(config,video_cfg, audio_cfg)
        self.cls_head = CLS_head(config)
        
    def forward(self,batch,compute_loss=True):
        if batch['batch_3m']['ids'] != [] :
            output_3m =  self.forward_batch(batch['batch_3m'], compute_loss)
        else:
            output_3m = {}
        if batch['batch_2m']['ids'] != []:
            output_2m = self.forward_batch(batch['batch_2m'], compute_loss)
        else:
            output_2m = {}

        return {**output_3m, **output_2m }


        
    def forward_batch(self,batch,compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        video_pixels = batch['video_pixels']
        labels = batch['labels']

        video_output = self.opt.forward_video_encoder(video_pixels)

        is_3m = 'audio_spectrograms' in batch
        if is_3m:
            audio_spectrograms = batch['audio_spectrograms']        
            audio_output = self.opt.forward_audio_encoder(audio_spectrograms)
            video_input = self.opt.get_multimodal_forward_input_video(video_output)
            audio_input = self.opt.get_multimodal_forward_input_audio(audio_output)
            multimodal_output = self.opt.forward_multimodal_encoder(None, None, video_input, audio_input)
            multimodal_output = pool_for_contra(multimodal_output, 'avg')
            logits = self.cls_head(multimodal_output)
        else:
            cls_token = pool_for_contra(video_output, 'avg')
            logits = self.cls_head(cls_token)

        if compute_loss:
            loss = F.cross_entropy(logits, labels, reduction='mean')
            if is_3m:
                return {'loss_3m': loss}
            else:
                return {'loss_2m': loss}

        else:
            if is_3m:
                return {'logits_3m': logits,
                        'labels_3m': labels}
            else:
                return {'logits_2m': logits,
                        'labels_2m': labels}



