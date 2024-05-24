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
import numpy as np
from torch.nn import LayerNorm as FusedLayerNorm

from model.pretrain import OPTForPretraining
from model.cb_pretrain import VLABForPretraining
from model.utils import txt_token_mask
from model.down_streams.scorer.scorer import Scorer


import ipdb


class VLABForVideoCaption(VLABForPretraining):
    
    def __init__(self, config):
        super().__init__(config)

        self.max_generation_len = config.caption_cfg["max_generation_len"]
        self.beam_size  = config.caption_cfg["beam_size"]
        self.decode_mode = config.caption_cfg["decode_mode"]
        self.mask_token = config.caption_cfg["mask_token"]
        self.eos_token = config.caption_cfg["eos_token"]
        self.cls_token = config.caption_cfg["cls_token"]
        
        self.return_all_captions = config.caption_cfg.get("return_all_captions",False)
        self.strengthen_two = getattr(config,'strengthen_two',True)
        self.label_smoothing = getattr(config,'label_smoothing',0.0)
        self.scst_finetuning = config.caption_cfg.get("scst_finetune",False)



        self.vision_type = [k for k in config.vision_encoder['pretrained'].keys()]
        self.frozen()

        if self.scst_finetuning:
            ann_path = config.data_cfg["ann_path"]
            ids_path = config.data_cfg["train_ids_path"]
            tokenizer_type = config.data_cfg["tokenizer"]["multi_encoder"]
            from transformers import BertTokenizer
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.scorer = Scorer(ids_path, bert_tokenizer)
            self.init_alpah()

    def frozen(self,):

        for i, k in enumerate(self.vision_type):

            if k.startswith("eva_clip") or k.startswith("adapter_eva"):
                for p in self.opt.vision_text_model.models[i].text.parameters():
                    p.requires_grad=False

                del self.opt.vision_text_model.models[i].visual.head
            else:
                for p in self.opt.vision_text_model.models[i].transformer.parameters():
                    p.requires_grad=False
                for p in [self.opt.vision_text_model.models[i].logit_scale, self.opt.vision_text_model.models[i].text_projection, self.opt.vision_text_model.models[i].visual.proj]:
                    p.requires_grad=False
                for p in self.opt.vision_text_model.models[i].ln_final.parameters():
                    p.requires_grad=False
                del self.opt.vision_text_model.models[i].token_embedding
                del self.opt.vision_text_model.models[i].positional_embedding


    def forward_caption(self, batch, compute_loss=True):
        
        multi_tokens = batch['batch_2m']['multi_tokens']
        video_pixels = batch['batch_2m']['video_pixels']

        if compute_loss:
            b,n,_,h,w = video_pixels.shape
            # ipdb.set_trace()
            video_output = self.opt.forward_video_encoder(video_pixels)


            video_input, video_global, video_patch = self.get_multimodal_forward_input_video(video_output,b)          

            if self.return_all_captions:
                assert multi_tokens.dim() == 3, "please check the multi token dimensions"
                _, cap_num, _ = multi_tokens.shape
                multi_tokens = multi_tokens.flatten(0, 1)
                tmp=[]
                for v in video_input:
                    v = v.repeat_interleave(cap_num, dim=0)
                    tmp.append(v)
                video_input=tmp

                tmp=[]
                for v in video_global:
                    v = v.repeat_interleave(cap_num, dim=0)
                    tmp.append(v)
                video_global=tmp

                tmp=[]
                for v in video_patch:
                    v = v.repeat_interleave(cap_num, dim=0)
                    tmp.append(v)
                video_patch=tmp

            if self.mask_type=="pseudo_mask":
                txt_input, txt_labels, txt_posids = self.text_masker(multi_tokens)
            else:
                txt_input, txt_labels = self.text_masker(multi_tokens)
                txt_posids=None
            caption_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, video_global=video_global, video_patch=video_patch, casual = True)
            # caption_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, casual = True, txt_posids=txt_posids)
            caption_output = caption_output[:, :txt_input.shape[1], :]
            caption_output = caption_output[txt_labels != -1]
            prediction_scores_caption = self.opt.multimodal_head(caption_output)

            
            masked_lm_loss = F.cross_entropy(prediction_scores_caption,
                                        txt_labels[txt_labels != -1],
                                        reduction='mean')    
            if self.label_smoothing > 0:
                    
                smooth_loss = -F.log_softmax(prediction_scores_caption, dim=-1).mean(dim=-1)
                masked_lm_loss  = (1- self.label_smoothing) * masked_lm_loss + self.label_smoothing * smooth_loss
            
            masked_lm_loss = masked_lm_loss.mean()
            return {"caption_loss_2m": masked_lm_loss}

        else:
            video_input = batch["batch_2m"]['video_input']
            video_global = batch["batch_2m"]['video_global']
            video_patch = batch['batch_2m']['video_patch']
            txt_input = multi_tokens

            caption_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, video_global=video_global, video_patch=video_patch, casual = True)
            caption_output_txt = caption_output[:, :multi_tokens.shape[1], :]
            caption_output_txt = caption_output_txt[:, -1]
            prediction_scores_caption = self.opt.multimodal_head(caption_output_txt)  
            return prediction_scores_caption


    
    def forward(self, batch, compute_loss=True):
        if compute_loss:
            if self.scst_finetuning:
                return self.forward_cap_scst(batch)
            else:
                return self.forward_caption(batch,compute_loss=True)
        else:
            return self.generate_caption(batch)
    
    def generate_caption(self, batch, mode=None):

        video_pixels = batch["batch_2m"]['video_pixels']

        b,n,_,h,w = video_pixels.shape
        video_output = self.opt.forward_video_encoder(video_pixels)

        video_input, video_global, video_patch = self.get_multimodal_forward_input_video(video_output,b)     

        # video_input = self.get_multimodal_forward_input_video(video_output,b)  
        batch["batch_2m"]['video_global'] = video_global
        batch['batch_2m']['video_patch'] = video_patch
        batch["batch_2m"]['video_input'] = video_input
    
        if self.beam_size > 1 and mode is None:
            generated_sequences, logits = self.decode_beam(batch)
        else:
            generated_sequences, logits = self.decode_greedy(batch, mode)
        
        return {'generated_sequence_2m': generated_sequences, "generate_sequence_logits": logits}



    def decode_greedy(self, batch, mode=None):
        if mode is not None:
            self.decode_mode=mode

        batch_size = batch['batch_2m']['video_pixels'].size(0)
        sents = torch.zeros((batch_size, self.max_generation_len), dtype=torch.long).fill_(self.eos_token).cuda()
        logprobs = torch.zeros(batch_size, self.max_generation_len).cuda()
        unfinished = torch.ones(batch_size, dtype=torch.bool).cuda()

        state = None
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
        
        return sents, logprobs
    def get_logprobs(self, batch, state):

        video_pixels = batch["batch_2m"]['video_pixels']
        batch_size = video_pixels.size(0)
        masked_tokens = torch.zeros(batch_size,1, dtype = torch.long, device = video_pixels.device).fill_(self.mask_token)
        cls_token = torch.zeros(batch_size,1, dtype = torch.long, device = video_pixels.device).fill_(self.cls_token)
        txt_tokens = torch.cat((state,masked_tokens), dim=1 ) if state is not None else masked_tokens
        txt_tokens = torch.cat((cls_token,txt_tokens), dim=1 )
        
        batch["batch_2m"]['multi_tokens'] = txt_tokens
        logits = self.forward_caption(batch, compute_loss = False)
        return F.log_softmax(logits, dim =1 )

    def decode_beam(self, batch):
        
        beam_size = self.beam_size
        batch_size = batch["batch_2m"]['video_pixels'].size(0)

        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = None
        #wt = torch.zeros(batch_size, dtype=torch.long).fill_(self.BOS_token).cuda()
        outputs = []
        for t in range(self.max_generation_len):
            cur_beam_size = 1 if t == 0 else beam_size
            word_logprob = self.get_logprobs(batch, state)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob
            # ipdb.set_trace()
            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != self.eos_token).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                #old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)


            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]


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
                batch["batch_2m"]['video_pixels'] = self.expand_tensor(batch["batch_2m"]['video_pixels'], beam_size)
                if isinstance(batch["batch_2m"]['video_input'], (list,tuple)):
                    for i in range(len(batch["batch_2m"]['video_input'])):
                        batch["batch_2m"]['video_input'][i] = self.expand_tensor(batch["batch_2m"]['video_input'][i], beam_size)
                        batch["batch_2m"]['video_global'][i] = self.expand_tensor(batch["batch_2m"]['video_global'][i], beam_size)
                        batch["batch_2m"]['video_patch'][i] = self.expand_tensor(batch["batch_2m"]['video_patch'][i], beam_size)
                else:
                    batch["batch_2m"]['video_input'] = self.expand_tensor(batch["batch_2m"]['video_input'], beam_size)

                
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.max_generation_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.max_generation_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, None

         

    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob


    def _expand_state(self, batch_size, beam_size, state, selected_beam):
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




    ##### scst related work
 
    def init_alpah(self):
    
        self.alpha_type = 0

        self.total_alpha = 0.7
        self.beta = 1.0
        self.recent_alpha = 0.7
        self.recent_num = 5000
        self.recent_alpha_list = np.linspace(0, 0, self.recent_num)
        self.recent_index = 0

        self.reward_sample_total = 0
        self.reward_greedy_total = 0
        self.reward_num = 0

    def update_alpha(self, rewards_sample, rewards_max):

        sample_mean = rewards_sample.mean()
        greedy_mean = rewards_max.mean()

        # total
        self.reward_sample_total += sample_mean
        self.reward_greedy_total += greedy_mean
        self.reward_num += 1
        self.total_alpha = self.reward_sample_total / self.reward_greedy_total

        # recent num
        self.recent_alpha_list[self.recent_index % self.recent_num] = sample_mean / greedy_mean
        self.recent_index += 1
        self.recent_alpha = np.mean(self.recent_alpha_list[:min(self.recent_index, self.recent_num)])

        reward_sample_avg = self.reward_sample_total / self.reward_num
        reward_greedy_avg = self.reward_greedy_total / self.reward_num
        #print("[avg_sample_reward: %.3f] [avg_greedy_reward: %.3f]" % (reward_sample_avg, reward_greedy_avg))

    def get_alpha(self):

        if self.alpha_type == 0:
            temp_alpha = 1.0
        elif self.alpha_type == 1:
            temp_alpha = self.recent_alpha * self.beta
        elif self.alpha_type == 2:
            temp_alpha = self.total_alpha * self.beta
        else:
            raise Exception("Error alpha_type")
        #print("[alpha_type: %d] [total_alpha: %.3f] [recent_alpha: %.3f]" % (self.alpha_type, self.total_alpha, self.recent_alpha))
        return temp_alpha

    def process_scst(self, seq):
        N, T = seq.size()
        sents = []
        for n in range(N):
            tokens = []
            for t in range(T):
                ix = seq[n, t].item()
                if ix == self.eos_token:
                    break
                tokens.append(ix)
            sents.append(tokens)
        return sents


    def forward_cap_scst(self, batch):

        loss_dict = {}
        batch_ids = batch['batch_2m']['ids']
        # batch_ids = batch['batch_2m']['txt_tokens']
        self.eval()
        with torch.no_grad():
            evaluation_dict_greedy = self.generate_caption(batch, mode='greedy')  ### compute  reward baseline
        self.train()
        evaluation_dict_sample = self.generate_caption(batch, mode='sample')  ### compute  reward baseline

        generated_sequences_t_v_greedy = self.process_scst(evaluation_dict_greedy['generated_sequence_2m'])
        generated_sequences_t_v_sample = self.process_scst(evaluation_dict_sample['generated_sequence_2m'])
        logprobs_t_v_sample = evaluation_dict_sample['generate_sequence_logits'] 

        reward_greedy = self.scorer(batch_ids, generated_sequences_t_v_greedy)
        reward_sample = self.scorer(batch_ids, generated_sequences_t_v_sample)

        self.update_alpha(reward_sample, reward_greedy)
        rewards = reward_sample - reward_greedy * self.get_alpha()
        rewards = torch.from_numpy(rewards).float().cuda()
        caption_loss_tv = self.reward_loss(evaluation_dict_sample['generated_sequence_2m'], logprobs_t_v_sample, rewards)    
        loss_dict['caption_loss_2m'] = caption_loss_tv
        
        return loss_dict


    def reward_loss(self, seq, logP, rewards):
        mask = seq !=self.eos_token 
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)
        return loss





