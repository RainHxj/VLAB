from torch import nn
import numpy as np
import torch
import random

def txt_token_mask(config, txt_tokens):

        
        mask_prob = config.losses["mask_prob"]
        mask_token = config.losses["mask_token"]
        vocab_range = [106, config.multi_encoder["vocab_size"] ]
        txt_labels = None
        txt_tokens = txt_tokens.clone() ### important, must have
        txt_mask_indicator = None
        
        txt_tokens = np.array(txt_tokens.cpu().numpy())

        if txt_mask_indicator is None:
            ### generate indicator first:
            txt_mask_indicator = np.zeros(txt_tokens.shape, dtype=np.int64)
            for i in range(len(txt_mask_indicator)):
                while all(txt_mask_indicator[i] == 0):
                    for j in range(1, len(txt_mask_indicator[0])):
                        if txt_tokens[i][j]!=0 and random.random() < mask_prob:
                            txt_mask_indicator[i][j] = 1
        
        labels = -np.ones(txt_tokens.shape, dtype=np.int64)
        for i in range(txt_tokens.shape[0]):
            for j in range(txt_tokens.shape[1]):
                
                if txt_mask_indicator[i][j] == 1 :
                    src_token = txt_tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.8:
                        txt_tokens[i][j] = mask_token  ### e-6 have no idea why too much 
                    elif prob < 0.9: 
                        txt_tokens[i][j] = random.choice(list(range(*vocab_range)))   
                        
                    labels[i][j] = src_token

        txt_tokens =torch.from_numpy(txt_tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return txt_tokens, labels


class TextMaskerWithoutReplace(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_token = config.losses["mask_token"]
        self.range = [106, config.multi_encoder["vocab_size"]]
        self.mask_prob = config.losses["mask_prob"]

    def forward(self, txt_tokens,  txt_mask_indicator = None):
        # txt_tokens = txt_tokens['bert_tokens']
        txt_tokens = txt_tokens.clone() ### important, must have

        txt_tokens, txt_labels = self.perform_mask(txt_tokens, txt_mask_indicator=txt_mask_indicator)
        return txt_tokens, txt_labels

    
    def perform_mask(self, txt_tokens, txt_mask_indicator=None):
        
        txt_tokens = np.array(txt_tokens.cpu().numpy())

        if txt_mask_indicator is None:
            ### generate indicator first:
            txt_mask_indicator = np.zeros(txt_tokens.shape, dtype=np.int64)
            for i in range(len(txt_mask_indicator)):
                while all(txt_mask_indicator[i] == 0):
                    for j in range(1, len(txt_mask_indicator[0])):
                        if txt_tokens[i][j]!=0 and random.random() < self.mask_prob:
                            txt_mask_indicator[i][j] = 1
        
        labels = -np.ones(txt_tokens.shape, dtype=np.int64)
        for i in range(txt_tokens.shape[0]):
            for j in range(txt_tokens.shape[1]):
                
                if txt_mask_indicator[i][j] == 1 :
                    src_token = txt_tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    txt_tokens[i][j] = self.mask_token  ### e-6 have no idea why too much  
                    #txt_tokens[i][j] = self.mask_token
                    labels[i][j] = src_token

                
        txt_tokens =torch.from_numpy(txt_tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return txt_tokens, labels


class TextMasker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_token = config.losses["mask_token"]
        self.range = [106, config.multi_encoder["vocab_size"]]
        self.mask_prob = config.losses["mask_prob"]

    def forward(self, txt_tokens,  txt_mask_indicator = None):
        # txt_tokens = txt_tokens['bert_tokens']
        txt_tokens = txt_tokens.clone() ### important, must have

        txt_tokens, txt_labels = self.perform_mask(txt_tokens, txt_mask_indicator=txt_mask_indicator)
        return txt_tokens, txt_labels

    
    def perform_mask(self, txt_tokens, txt_mask_indicator=None):
        
        txt_tokens = np.array(txt_tokens.cpu().numpy())

        if txt_mask_indicator is None:
            ### generate indicator first:
            txt_mask_indicator = np.zeros(txt_tokens.shape, dtype=np.int64)
            for i in range(len(txt_mask_indicator)):
                while all(txt_mask_indicator[i] == 0):
                    for j in range(1, len(txt_mask_indicator[0])):
                        if txt_tokens[i][j]!=0 and random.random() < self.mask_prob:
                            txt_mask_indicator[i][j] = 1
        
        labels = -np.ones(txt_tokens.shape, dtype=np.int64)
        for i in range(txt_tokens.shape[0]):
            for j in range(txt_tokens.shape[1]):
                
                if txt_mask_indicator[i][j] == 1 :
                    src_token = txt_tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.8:
                        txt_tokens[i][j] = self.mask_token  ### e-6 have no idea why too much 
                    elif prob < 0.9: 
                        txt_tokens[i][j] = random.choice(list(range(*self.range)))   
                    #txt_tokens[i][j] = self.mask_token
                    labels[i][j] = src_token

                
        txt_tokens =torch.from_numpy(txt_tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return txt_tokens, labels



class PseduoTextMasker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_token = config.losses["mask_token"]
        self.range = [106, config.multi_encoder["vocab_size"]]
        self.mask_prob = config.losses["mask_prob"]
        self.len_txt=60

    def forward(self, txt_tokens,  txt_mask_indicator = None):
        # txt_tokens = txt_tokens['bert_tokens']
        txt_tokens = txt_tokens.clone() ### important, must have

        txt_tokens, txt_labels, txt_posid = self.perform_mask(txt_tokens, txt_mask_indicator=txt_mask_indicator)
        return txt_tokens, txt_labels, txt_posid

    
    def perform_mask(self, txt_tokens, txt_mask_indicator=None):

        txt_tokens = np.array(txt_tokens.cpu().numpy())

        new_txt_tokens_list=[]
        label_list = []
        pos_list = []
        for i in range(txt_tokens.shape[0]):
            tmp = []
            label=[]
            pos=[]
            tmp.append(txt_tokens[i][0])
            label.append(-1)
            pos.append(0)
            for j in range(1,txt_tokens.shape[1]):
                
                if txt_tokens[i][j]!=0 and random.random() < self.mask_prob:
                    tmp.append(self.mask_token)
                    tmp.append(txt_tokens[i][j])
                    label.append(txt_tokens[i][j])
                    label.append(-1)
                    pos.append(j)
                    pos.append(j)
                        
                else: 
                    tmp.append(txt_tokens[i][j])
                    label.append(-1)
                    pos.append(j)

            if len(tmp)<self.len_txt:
                
                tmp.extend([0]*(self.len_txt-len(tmp)))
                label.extend([0]*(self.len_txt-len(label)))
                pos.extend(range(len(pos),(self.len_txt)))
            else:
                tmp = tmp[:self.len_txt]
                label = label[:self.len_txt]
                pos = pos[:self.len_txt]
            new_txt_tokens_list.append(tmp)
            label_list.append(label)
            pos_list.append(pos)
            

        new_txt_tokens = np.array(new_txt_tokens_list, dtype=np.int64)
        labels = np.array(label_list,dtype=np.int64)
        pos_ids = np.array(pos_list,dtype=np.int64)


        new_txt_tokens =torch.from_numpy(new_txt_tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        pos_ids = torch.from_numpy(pos_ids).long().cuda()

        return new_txt_tokens, labels, pos_ids
