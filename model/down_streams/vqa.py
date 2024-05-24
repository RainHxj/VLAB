"""

Licensed under the MIT license.

OPT for VQA model
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm as FusedLayerNorm
import torch

from .pretrain import OPTForPretraining, OPTModel

def pool_for_vqa(multimodal_output, attn_mask_txt):
    attn_masks_all = torch.ones(multimodal_output.shape[:2]).to(attn_mask_txt)
    attn_masks_all[:,:attn_mask_txt.shape[1]] = attn_mask_txt
    multimodal_output = (multimodal_output * attn_masks_all.unsqueeze(-1)).sum(dim=1) / attn_masks_all.sum(dim=1,keepdim=True)
    return multimodal_output

class VQA_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_size, config.answer_num)
    def forward(self, cls_token):
        return self.linear2(self.activation(self.linear1(cls_token)))

class OPTForOpenEndedVQA(OPTForPretraining):
    def __init__(self,config,video_cfg):
        super().__init__(config,video_cfg)
        self.vqa_head = VQA_head(config)
        self.strengthen_two = getattr(config,'strengthen_two',True)
    def forward(self,batch,compute_loss=True):

        if batch['batch_2m']['ids'] != []:
            output_2m = self.forward_batch(batch['batch_2m'], compute_loss)
        else:
            output_2m = {}

        return {**output_2m}

    def forward_batch(self,batch,compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']  
        video_pixels = batch['video_pixels']
        answers = batch['answers']
        sample_num = batch['sample_num']

        txt_input, attn_mask_txt, _ = self.opt.get_multimodal_forward_input_txt(txt_tokens, perform_mask=False)
        video_output = self.opt.forward_video_encoder(video_pixels)
        video_input = self.opt.get_multimodal_forward_input_video(video_output)
        # import ipdb
        # ipdb.set_trace()
        if not compute_loss:
            video_input_expand = []
            for i in range(video_input.shape[0]):
                video_input_expand.append( video_input[i:i+1].expand(sample_num[i],-1,-1))
            video_input = torch.cat(video_input_expand,dim=0)



        

        is_3m = 'audio_spectrograms' in batch
        if is_3m:
            audio_spectrograms = batch['audio_spectrograms']        
            audio_output = self.opt.forward_audio_encoder(audio_spectrograms)
            audio_input = self.opt.get_multimodal_forward_input_audio(audio_output)
            if not compute_loss:
                audio_input_expand = []
                for i in range(audio_input.shape[0]):
                    audio_input_expand.append(audio_input[i:i+1].expand(sample_num[i],-1,-1))
                audio_input = torch.cat(audio_input_expand,dim=0)
        else:
            audio_input=None

        multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, video_input, audio_input)
        multimodal_output = pool_for_vqa(multimodal_output, attn_mask_txt)
        logits = self.vqa_head(multimodal_output)

        if compute_loss:
            
            loss = F.cross_entropy(logits, answers, reduction='mean')
            if is_3m and self.strengthen_two:
                multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, video_input,None)
                multimodal_output = pool_for_vqa(multimodal_output, attn_mask_txt)
                logits = self.vqa_head(multimodal_output)
                loss_2m = F.cross_entropy(logits, answers, reduction='mean')
                return {'vqa_loss_3m':0.5*loss+0.5*loss_2m}

            elif is_3m:
                return  {'vqa_loss_3m':loss}
            else:
                return  {'vqa_loss_2m':loss}
            
        else:
            if is_3m:
                multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, video_input,None)
                multimodal_output = pool_for_vqa(multimodal_output, attn_mask_txt)
                logits_woaudio = self.vqa_head(multimodal_output)

                return {'logits_3m': logits,
                        'logits_3m_woaudio': logits_woaudio}
            else:
                return {'logits_2m': logits}


