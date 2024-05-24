"""

Licensed under the MIT license.

OPT for pretraining
"""
from collections import defaultdict
from logging import logMultiprocessing

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm as FusedLayerNorm

from model.pretrain import OPTForPretraining
from model.utils import txt_token_mask
import ipdb


class OPTForVideoQA(OPTForPretraining):
    
    def __init__(self, config):
        super().__init__(config)

        self.max_generation_len = config.caption_cfg["max_generation_len"]
        self.beam_size  = config.caption_cfg["beam_size"]
        self.decode_mode = config.caption_cfg["decode_mode"]
        self.mask_token = config.caption_cfg["mask_token"]
        self.eos_token = config.caption_cfg["eos_token"]
        self.cls_token = config.caption_cfg["cls_token"]

        self.strengthen_two = getattr(config,'strengthen_two',True)
        self.label_smoothing = getattr(config,'label_smoothing',0.1)

        self.max_generation_len = 1
        

    def forward_caption(self, batch, compute_loss=True):
        
        answer_tokens = batch['batch_2m']['answer_tokens']
        question_tokens = batch['batch_2m']['question_tokens']
        video_pixels = batch['batch_2m']['video_pixels']
        answer_prompt = torch.tensor([1996, 3437, 2003]).long().cuda().repeat(video_pixels.shape[0],1) # the answer is
        question_tokens = torch.cat((question_tokens[:,:], answer_prompt), dim=-1)
        question_tokens[question_tokens==102]=0

        if compute_loss:

            b,n,_,h,w = video_pixels.shape
            video_output = self.opt.forward_video_encoder(video_pixels)
            video_output = video_output.reshape(b,-1,video_output.shape[-1])

            answer_input, txt_labels = self.text_masker(answer_tokens)
            answer_input = answer_input[:,1:]
            txt_labels = txt_labels[:,1:]
            txt_inputs = torch.cat([question_tokens, answer_input], dim=1)
            txt_inputs[:,-1]=102
            txt_labels[:,-1]=-1


            video_input = self.get_multimodal_forward_input_video(video_output)
            caption_output = self.opt.forward_multimodal_encoder(txt_inputs, video_feat=video_input, casual=False)
            caption_output = caption_output[:, question_tokens.shape[1]:txt_inputs.shape[1], :]
            caption_output = caption_output[txt_labels != -1]
            prediction_scores_caption = self.opt.multimodal_head(caption_output)

            masked_lm_loss = F.cross_entropy(prediction_scores_caption,
                                        txt_labels[txt_labels != -1],
                                        reduction='mean')    
            if self.label_smoothing > 0:
                smooth_loss = -F.log_softmax(prediction_scores_caption, dim=-1).mean(dim=-1)
                masked_lm_loss  = (1- self.label_smoothing) * masked_lm_loss + self.label_smoothing * smooth_loss
            
            masked_lm_loss = masked_lm_loss.mean()
            return {"vqa_loss_2m": masked_lm_loss}

        else:
            video_input = batch["batch_2m"]['video_input']
            predict_tokens = batch["batch_2m"]["predict_tokens"]
            txt_inputs=torch.cat([question_tokens, predict_tokens],dim=1)
            # ipdb.set_trace()
            caption_output = self.opt.forward_multimodal_encoder(txt_inputs, video_feat=video_input, casual = False)
            caption_output_txt = caption_output[:, :txt_inputs.shape[1], :]
            caption_output_txt = caption_output_txt[:, -2]
            prediction_scores_caption = self.opt.multimodal_head(caption_output_txt)  
            return prediction_scores_caption


    
    def forward(self, batch, compute_loss=True):
        if compute_loss:
            return self.forward_caption(batch ,compute_loss=True)
        else:
            return self.generate_caption(batch)
    
    def generate_caption(self, batch):

        video_pixels = batch["batch_2m"]['video_pixels']

        b,n,_,h,w = video_pixels.shape
        video_output = self.opt.forward_video_encoder(video_pixels)
        video_output = video_output.reshape(b,-1,video_output.shape[-1])
        video_input = self.get_multimodal_forward_input_video(video_output)
   
        batch["batch_2m"]['video_input'] = video_input
    
        if self.beam_size >1:
            generated_sequences = self.decode_beam(batch)
        
        else:
            generated_sequences = self.decode_greedy(batch)
        
        return {'generated_sequence_2m': generated_sequences}



    def decode_greedy(self, batch):

        video_pixels = batch["batch_2m"]['video_pixels']
        batch_size = video_pixels.size(0)
        state = None        
        sents = torch.zeros((batch_size, self.max_generation_len), dtype=torch.long).fill_(self.eos_token).cuda()
        logprobs = torch.zeros(batch_size, self.max_generation_len).cuda()
        unfinished = torch.ones(batch_size, dtype=torch.bool).cuda()

        
        for t in range(self.max_generation_len):
            logprobs_t = self.get_logprobs(batch, state)
        
            if self.decode_mode == 'greedy': 
                logP_t, wt = torch.max(logprobs_t, 1)
            elif self.decode_mode =='sample':
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            else:
                raise NotImplementedError
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt != self.eos_token)
            wt = wt * unfinished.type_as(wt) + (1 - unfinished.type_as(wt)) * self.eos_token
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)
            state = wt.unsqueeze(1) if state is None else torch.cat((state,wt.unsqueeze(1)),dim=1)

            if unfinished.sum() == 0:
                break
        
        return sents
    def get_logprobs(self, batch, state):

        video_pixels = batch["batch_2m"]['video_pixels']
        batch_size = video_pixels.size(0)
        masked_tokens = torch.zeros(batch_size,1, dtype = torch.long, device = video_pixels.device).fill_(self.mask_token)
        cls_token = torch.zeros(batch_size,1, dtype = torch.long, device = video_pixels.device).fill_(self.cls_token)
        eos_token = torch.zeros(batch_size,1, dtype = torch.long, device = video_pixels.device).fill_(self.eos_token)
        # ipdb.set_trace()
        txt_tokens = torch.cat((masked_tokens,eos_token), dim=1)

        
        batch["batch_2m"]['predict_tokens'] = txt_tokens
        logits = self.forward_caption(batch, compute_loss = False)
        return F.log_softmax(logits, dim =1 )

