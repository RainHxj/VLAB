"""

Licensed under the MIT license.

OPT for ITM model
"""
from collections import defaultdict

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm as FusedLayerNorm

from data import retrieval

from .transformer import GELU
from .model import OPTModel, OPTPreTrainedModel
import ipdb
import numpy as np
import random
import torch.distributed as dist
from utils.logger import LOGGER
import yaml
from torchvision.transforms import *
import math
from time import time
from tqdm import tqdm
from utils.misc import NoOp
import os
from utils.distributed import any_broadcast
from utils.distributed import all_gather_list, ddp_allgather_with_grads
from model.pretrain import OPTForPretraining
class OPTForVideoAudioGen(OPTForPretraining):

    def __init__(self, config, video_cfg, audio_cfg):
        super().__init__(config,video_cfg, audio_cfg)
        self.beam_size_video = 1
        self.max_generation_len_video = 650
        self.max_generation_len_audio = 260 
        self.max_generation_len = self.max_generation_len_video + self.max_generation_len_audio
        self.decode_mode = 'sample'
        self.rank_size = getattr(config,'rank_size', 1)
        self.sample_topk = getattr(config,'sample_topk', 200)
        self.videogen_batches = getattr(config,'videogen_batches', 1)
    def forward(self, batch, compute_loss=True, gen_tokens = False):
        if compute_loss:
            loss_dict = {}
        else:
            evaluation_dict = {}
        batch = batch['batch_3m']
        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']
        video_pixels = batch['video_pixels']
        audio_spectrograms = batch['audio_spectrograms']
        video_tokens = video_pixels #### b,640
        b = video_tokens.shape[0]
        video_tokens = video_tokens.reshape(b*10,-1)
        #### create input  video tokens template
        ### input tokens [SOV] [0],[1],...[255],[SOV],[0],[1],...,[255]
        ### target tokens [0],[1],...[255],[SOV],[0],[1],...,[255][SOV]
        SOV_token = torch.tensor(self.video_codebook_size).to(video_tokens).unsqueeze(0).expand(b*10,-1)
        video_tokens = torch.cat((SOV_token, video_tokens), dim=1)
        video_tokens = video_tokens.reshape(b,-1) 
        target_video_tokens = torch.cat((video_tokens[:,1:],video_tokens[:,0:1]),dim=1)

        audio_tokens = audio_spectrograms #### b,250
        b = audio_tokens.shape[0]
        audio_tokens = audio_tokens.reshape(b*10,-1)
        #### create input  video tokens template
        ### input tokens [SOV] [0],[1],...[255],[SOV],[0],[1],...,[255]
        ### target tokens [0],[1],...[255],[SOV],[0],[1],...,[255][SOV]
        SOA_token = torch.tensor(self.audio_codebook_size).to(audio_tokens).unsqueeze(0).expand(b*10,-1)
        audio_tokens = torch.cat((SOA_token, audio_tokens), dim=1)
        audio_tokens = audio_tokens.reshape(b,-1) 
        target_audio_tokens = torch.cat((audio_tokens[:,1:],audio_tokens[:,0:1]),dim=1)

        txt_input, attn_mask_txt, _ = self.opt.get_multimodal_forward_input_txt(txt_tokens, perform_mask=False)
        output_embeddings = self.opt.forward_video_decoder(txt_input, attn_mask_txt, video_tokens, audio_tokens)
        txt_len = txt_tokens.shape[1]
        video_len = video_tokens.shape[1]
        output_embeddings_video = output_embeddings[:, txt_len:txt_len+video_len]
        output_embeddings_audio = output_embeddings[:, txt_len+video_len :]
        logits_video = self.video_decoder_head(output_embeddings_video)
        logits_audio = self.audio_decoder_head(output_embeddings_audio)
        video_generation_loss = F.cross_entropy(logits_video.reshape(-1,logits_video.shape[-1]), target_video_tokens.reshape(-1))
        audio_generation_loss = F.cross_entropy(logits_audio.reshape(-1,logits_audio.shape[-1]), target_audio_tokens.reshape(-1))
        if compute_loss:
            loss_dict['video_generation_loss'] = video_generation_loss
            loss_dict['audio_generation_loss'] = audio_generation_loss
            
        else:
            evaluation_dict['video_generation_loss'] = video_generation_loss
            evaluation_dict['audio_generation_loss'] = audio_generation_loss

            if gen_tokens:

                batch_size, n,c = txt_input.shape
                #### expand  rank size
                txt_input = txt_input.unsqueeze(1).expand(-1,self.rank_size, -1, -1).reshape(batch_size*self.rank_size,n,c)
                attn_mask_txt = attn_mask_txt.unsqueeze(1).expand(-1,self.rank_size, -1).reshape(batch_size*self.rank_size,-1)

                generated_tokens_ls = []
                for i in range(self.rank_size):

                    batch['txt_input'] = txt_input[i*batch_size:(i+1)*batch_size]
                    batch['attn_mask_txt'] = attn_mask_txt[i*batch_size:(i+1)*batch_size]


                    if self.beam_size_video >1:
                        generated_sequences = self.decode_beam(batch)
                    else:
                        generated_sequences = self.decode_greedy(batch)  
                    
                    generated_tokens_ls.append(generated_sequences)
                generated_tokens = torch.cat(generated_tokens_ls,dim=0)
                batch_size = generated_tokens.shape[0]
                generated_tokens_video = generated_tokens[:,:self.max_generation_len_video] ##b,650
                generated_tokens_video = generated_tokens_video.reshape(batch_size,10,65)[:,:,:-1].reshape(batch_size, 10, 8, 8)
                generated_tokens_audio = generated_tokens[:,self.max_generation_len_video:] ###b,260
                generated_tokens_audio = generated_tokens_audio.reshape(batch_size,10,26)[:,:,:-1].reshape(batch_size, 50,5).permute(0,2,1)
                evaluation_dict['generated_tokens_video'] = generated_tokens_video
                evaluation_dict['generated_tokens_audio'] = generated_tokens_audio

        if compute_loss:
            return loss_dict
        else:
            return evaluation_dict





    def decode_greedy(self, batch):

        batch_size = batch['txt_input'].size(0)
        sents = torch.zeros((batch_size, self.max_generation_len), dtype=torch.long).cuda()
        logprobs = torch.zeros(batch_size, self.max_generation_len).cuda()
        state = torch.ones(batch_size, dtype=torch.bool).cuda().unsqueeze(1) * self.video_codebook_size
        if dist.get_rank() == 0:
            pbar = tqdm(total=self.max_generation_len)
        else:
            pbar = NoOp()
        for t in range(self.max_generation_len):

            if t<self.max_generation_len_video -1 :
                gen_type = 'video'
            elif t == self.max_generation_len_video -1: ###649  
                wt = torch.ones(batch_size,1).long().cuda() * self.audio_codebook_size
                state = torch.cat((state,wt),dim=1)
                continue
            else:
                gen_type = 'audio'
            logits = self.get_logits(batch, state)
            logits[:,-1] = -10000
            if self.decode_mode == 'greedy': 
                _, wt = torch.max(logits, 1)
            elif self.decode_mode =='sample':
                ### sample from the top-100
                if gen_type == 'video':
                    logits = logits.scatter(1, (-logits).topk(self.video_codebook_size + 1 - self.sample_topk, dim=1)[1],-10000) 
                else:
                    logits = logits.scatter(1, (-logits).topk(self.audio_codebook_size + 1 - self.sample_topk, dim=1)[1],-10000)
                probs_t = F.softmax(logits,dim=1)
                wt = torch.multinomial(probs_t, 1)
            else:
                raise NotImplementedError
            wt = wt.view(-1).long()
            sents[:,t] = wt
            #logprobs[:,t] = logP_t.view(-1)
            state = wt.unsqueeze(1) if state is None else torch.cat((state,wt.unsqueeze(1)),dim=1)
            pbar.update(1)
            # ipdb.set_trace()
        pbar.close()
        return sents

    def get_logits(self, batch, state):

        txt_input = batch['txt_input']
        attn_mask_txt =  batch['attn_mask_txt']
        video_tokens = state[:,:self.max_generation_len_video]
        if state.shape[1]>self.max_generation_len_video:
            audio_tokens = state[:,self.max_generation_len_video:]
        else:
            audio_tokens = None
        output_embeddings = self.opt.forward_video_decoder(txt_input,attn_mask_txt,video_tokens, audio_tokens)

        if state.shape[1]>self.max_generation_len_video:
            logits = self.audio_decoder_head(output_embeddings[:,-1])
        else:
            logits = self.video_decoder_head(output_embeddings[:,-1])
        #return F.log_softmax(logits, dim =1 )
        return logits

    def decode_beam(self, batch):
        if dist.get_rank() == 0:
            pbar = tqdm(total=self.max_generation_len)
        else:
            pbar = NoOp()
        beam_size = self.beam_size_video
        batch_size = batch['txt_output_unmasked'].size(0)

        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = torch.ones(batch_size, dtype=torch.bool).cuda().unsqueeze(1) * self.vqvae_codebook_num
        #wt = torch.zeros(batch_size, dtype=torch.long).fill_(self.BOS_token).cuda()
        outputs = []
        for t in range(self.max_generation_len):
            cur_beam_size = 1 if t == 0 else beam_size
            word_logprob = self.get_logprobs(batch, state)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 999999).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                #old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # suppress UNK tokens in the decoding
            #candidate_logprob[:, :, candidate_logprob.size(-1) - 1] = -99999

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # for s in range(len(state)):
            #     state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)




            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            

            if state is not None:
                state = self._expand_state(batch_size, beam_size, state, selected_beam)
                state = torch.cat((state,selected_words),dim = 1)
            else:
                state = selected_words

            if t == 0:
                batch['txt_output_unmasked'] = self.expand_tensor(batch['txt_output_unmasked'], beam_size)
                batch['attn_mask_txt'] = self.expand_tensor(batch['attn_mask_txt'], beam_size)

            pbar.update(1)
        pbar.close()
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.max_generation_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.max_generation_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs

         

    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob


    def _expand_state(self, batch_size, beam_size, state, selected_beam):
        if state.shape[0] != batch_size * beam_size:
            state = self.expand_tensor(state, beam_size)

        seq_len = state.size(-1)
        beam = selected_beam               #beam:  Bxbeam_size     state:(B*cur_beamm_size)xLXL
        beam = beam.unsqueeze(-1)       
        
        state = torch.gather(
            state.view(batch_size, beam_size, seq_len), 1,
            beam.expand(batch_size, beam_size,seq_len)
        )
        state = state.view(-1, seq_len)
        return state


    def expand_tensor(self, tensor, size, dim=1):
        if size == 1 or tensor is None:
            return tensor
        tensor = tensor.unsqueeze(dim)
        tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim+1:])).contiguous()
        tensor = tensor.view(list(tensor.shape[:dim-1]) + [-1] + list(tensor.shape[dim+1:]))
        return tensor

