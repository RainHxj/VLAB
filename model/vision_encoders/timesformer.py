
import logging
import math
import copy
import torch
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
import torch.nn.functional as F

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output



class TimesFormerLayer(nn.Module):
    def __init__(self, config, video_cfg):
        super().__init__()
        self.video_encoder_type = config.video_encoder_type
        assert self.video_encoder_type in ['timesformer_time_space','timesformer_joint','timesformer_space'], "not implemented"
        if  self.video_encoder_type == 'timesformer_time_space':
            self.attention_time = MultiHeadAttention(config,video_cfg)
        self.attention_space = MultiHeadAttention(config,video_cfg)
        self.ff_layer = FeedForward(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        if  self.video_encoder_type == 'timesformer_time_space':
            self.layernorm1 = LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm2 = LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm3 = LayerNorm(config.hidden_size, eps=1e-12)


    def forward(self, hidden_states):

        if  self.video_encoder_type == 'timesformer_time_space':
            residual = hidden_states
            hidden_states = self.layernorm1(hidden_states)
            attention_output = self.attention_time(hidden_states, mode='time')
            hidden_states = residual + self.dropout(attention_output)

        residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        attn_mode = 'joint' if  self.video_encoder_type == 'timesformer_joint' else 'space'
        attention_output = self.attention_space(hidden_states, mode= attn_mode)
        hidden_states = residual + self.dropout(attention_output)


        residual = hidden_states
        hidden_states = self.layernorm3(hidden_states)
        ff_output = self.ff_layer(hidden_states)
        hidden_states = residual + self.dropout(ff_output)

        return hidden_states



def clones(x,times):
    return nn.ModuleList([copy.deepcopy(x) for i in range(times)])



class MultiHeadAttention(nn.Module):
    def __init__(self, config, video_cfg):
        super().__init__()
        self.linears = clones(nn.Linear(config.hidden_size, config.hidden_size), 4)
        self.head_num = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.hidden_size_head = self.hidden_size // self.head_num
        self.dropout=nn.Dropout(config.attention_dropout)
        self.frame_num = video_cfg['sample_num']

    def forward(self,hidden_states, mode):
        batch_size=hidden_states.shape[0]
        q,k,v=[layer(x).reshape(batch_size,-1,self.head_num, self.hidden_size_head).transpose(1,2).reshape(batch_size*self.head_num, -1, self.hidden_size_head) \
                 for layer,x in zip(self.linears,(hidden_states,hidden_states,hidden_states))]   ### shape b*h,(1+f*p),c
        
        
        if mode == 'joint':
            output = self.compute_attention(q,k,v)
            return output 

        q_cls, k_cls, v_cls = q[:,0], k[:,0], v[:,0]
        q_, k_, v_ = q[:,1:], k[:,1:], v[:,1:]
        ### cls token attends to all tokens
        cls_output = self.compute_attention(q_cls.unsqueeze(1), k, v)

        patch_num = q_.shape[1] // self.frame_num

        if mode == 'time':
            q_,k_,v_ = [i.reshape(batch_size*self.head_num, self.frame_num, patch_num, self.hidden_size_head).permute(2,0,1,3).reshape(-1,self.frame_num, self.hidden_size_head) for i in (q_,k_,v_)]
            ### p*b*h,f,c
            k_cls = k_cls.unsqueeze(0).expand(patch_num,-1,-1).reshape(-1, 1, self.hidden_size_head)
            k_ = torch.cat((k_cls,k_),dim=1)
            v_cls = v_cls.unsqueeze(0).expand(patch_num,-1,-1).reshape(-1, 1, self.hidden_size_head)
            v_ = torch.cat((v_cls,v_),dim=1)
            output = self.compute_attention(q_,k_,v_)  #### p*b, f, h*c
            output = output.reshape(patch_num,batch_size,self.frame_num,-1).permute(1,2,0,3).reshape(batch_size,-1,self.hidden_size)
            output = torch.cat((cls_output,output),dim=1)



        elif mode == 'space':
            q_,k_,v_ = [i.reshape(batch_size*self.head_num, self.frame_num, patch_num, self.hidden_size_head).permute(1,0,2,3).reshape(-1, patch_num, self.hidden_size_head) for i in (q_,k_,v_)]
            ### f*b*h,p,c
            k_cls = k_cls.unsqueeze(0).expand(self.frame_num,-1,-1).reshape(-1, 1, self.hidden_size_head)
            k_ = torch.cat((k_cls,k_),dim=1)
            v_cls = v_cls.unsqueeze(0).expand(self.frame_num,-1,-1).reshape(-1, 1, self.hidden_size_head)
            v_ = torch.cat((v_cls,v_),dim=1)
            output = self.compute_attention(q_,k_,v_)  #### f*b, p, h*c
            output = output.reshape(self.frame_num,batch_size, patch_num,-1).permute(1,0,2,3).reshape(batch_size,-1,self.hidden_size)
            output = torch.cat((cls_output,output),dim=1)

        else:
            raise NotImplementedError()

        return output 
        
    def compute_attention(self, q,k,v):
        q_len, norm_d = q.shape[1], q.shape[-1]
        att_map=torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(norm_d)       
        att_map=F.softmax(att_map,dim=-1)
        att_map=self.dropout(att_map)
        attn_output = self.linears[-1](torch.matmul(att_map,v).reshape(-1, self.head_num, q_len, self.hidden_size_head).permute(0,2,1,3).reshape(-1, q_len, self.hidden_size))
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1=nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2=nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = GELU()


    def forward(self,x):
        return self.linear2((self.activation(self.linear1(x))))

class TimesFormerEncoder(nn.Module):
    def __init__(self, config,video_cfg):
        super().__init__()
        layer = TimesFormerLayer(config,video_cfg)

        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

        self.last_layernorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, input_):
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)       
        hidden_states = self.last_layernorm(hidden_states)
        return hidden_states
