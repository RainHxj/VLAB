import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import *
from torch.nn import LayerNorm as FusedLayerNorm

from .model import VLABModel, VLABPreTrainedModel
from .head import ContrastiveHead
from .utils import txt_token_mask,TextMasker
from utils.distributed import ddp_allgather_with_grads
import ipdb

def pool_for_contra(feature, mode, text=None):
    if mode == 'cls':
        return feature[:,0]
    elif mode == 'distill':
        return 0.5*(feature[:,0] + feature[:,1])
    elif mode == 'average':
        return torch.mean(feature, dim=1) 
    elif mode == 'clip':
        return feature[torch.arange(feature.shape[0]), text.argmax(dim=-1)]
    else:
        raise NotImplementedError()


class OPTForPretraining(VLABPreTrainedModel):
    """ OPT pretraining """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.opt = VLABModel(config)
        video_cfg = config.vision_encoder
        txt_cfg = config.txt_encoder
        self.txt_cfg=txt_cfg
        multi_cfg = config.multi_encoder
        self.video_dim=video_cfg["video_dim"]
        self.txt_dim=txt_cfg["txt_dim"]
        self.proj_dim=video_cfg["proj_dim"]
        

        ## contra head for contrastive loss
        if txt_cfg.get("text_type", None) :
            self.contra_head_t = ContrastiveHead(self.txt_dim, self.proj_dim)
        else:
            self.contra_head_t = lambda cls_token_t : cls_token_t @ self.opt.vision_text_model.text_projection #
        
        if video_cfg.get("vision_type",None) == "cb_clip":
            self.contra_head_v = lambda cls_token_v : cls_token_v @ self.opt.vision_text_model.visual_proj #ContrastiveHead(self.txt_dim, self.proj_dim)
        else:
            self.contra_head_v = lambda cls_token_v : cls_token_v @ self.opt.vision_text_model.visual.proj #ContrastiveHead(self.txt_dim, self.proj_dim)
        
        if not self.config.vision_encoder["vision_type"].endswith('clip') and self.txt_cfg.get("text_type",None) is not None:
            self.contra_temp = nn.Parameter(torch.tensor(0.07))
        self.contra_sync = True
        
        ## multimodal encoder
        self.multimodal_dim=multi_cfg["multimodal_dim"]
        if self.video_dim!=self.multimodal_dim:
            self.proj_multi_input = nn.Sequential( 
                                        nn.Linear(self.video_dim, self.multimodal_dim), 
                                        FusedLayerNorm(self.multimodal_dim, eps=1e-12)) 

        self.video_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim))
        self.video_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.video_cfg["sample_num"], self.multimodal_dim))

        self.text_masker = TextMasker(self.config)



    def get_multimodal_forward_input_video(self, video_output):
        if self.video_dim!=self.multimodal_dim:
            video_output = self.proj_multi_input(video_output)
        b,_,_,_ = video_output.shape
        video_output =  video_output + self.video_frame_embedding.unsqueeze(-2)
        video_output = video_output.reshape(b, -1, self.multimodal_dim)  + self.video_type_embeddings
        return video_output   

    def forward(self, batch, task, compute_loss=True, batch_idx = 0):
        #### pretraining forward function ##
        if compute_loss:
            loss_dict = {}
        else:
            evaluation_dict = {}
        
        txt_tokens = batch['batch_2m']['txt_tokens']
        multi_tokens = batch['batch_2m']['multi_tokens']
        video_pixels = batch['batch_2m']['video_pixels']

        task = task.split('_')
        
        ## txt forward
        txt_output = self.opt.forward_txt_encoder(txt_tokens)

        ## video forward
        b,n,_,h,w = video_pixels.shape
        video_output = self.opt.forward_video_encoder(video_pixels)



        if 'contraTwo' in task:
            #### forward contra 
            if isinstance(video_output,(list,tuple)): 
                video_output_contra = video_output[0][:, 0, :].reshape(b, -1, video_output[0].shape[-1])
            else:
                video_output_contra = video_output[:, 0, :].reshape(b, -1, video_output.shape[-1])

            if self.txt_cfg.get("text_type",None):
                cls_token_t = pool_for_contra(txt_output,'cls')
            else:
                cls_token_t = pool_for_contra(txt_output,'clip', txt_tokens)
            cls_token_v = pool_for_contra(video_output_contra,'average')  

            feat_t = self.contra_head_t(cls_token_t)
            # feat_t =  F.normalize(feat_t,dim=-1)
            feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
            

            feat_v = self.contra_head_v(cls_token_v)
            # feat_v =  F.normalize(feat_v,dim=-1)
            feat_v = feat_v / feat_v.norm(dim=-1, keepdim=True)

            if compute_loss:
                loss_dict['contra_loss'] = self.contrastive_loss(feat_t, feat_v)
                
            else:
                evaluation_dict['feat_t'] = feat_t
                evaluation_dict['feat_v'] = feat_v

        if 'mlmTwo' in task:
            #### forward mlmTwo
            if isinstance(video_output,(list,tuple)):
                video_input = []
                for vec in video_output:
                    video_output_mlm = vec.reshape(b, -1, vec.shape[-2], vec.shape[-1])
                    video_input.append(self.get_multimodal_forward_input_video(video_output_mlm))
            else:
                video_output_mlm = video_output.reshape(b, -1, video_output.shape[-2], video_output.shape[-1])
                video_input = self.get_multimodal_forward_input_video(video_output_mlm)          

            txt_input, txt_labels = self.text_masker(multi_tokens)
            mlm_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, casual = False)
            mlm_output = mlm_output[:, :txt_tokens.shape[1], :]
            mlm_output = mlm_output[txt_labels != -1]
            prediction_scores_mlm = self.opt.multimodal_head(mlm_output)

            if compute_loss:
                masked_lm_loss = F.cross_entropy(prediction_scores_mlm,
                                             txt_labels[txt_labels != -1],
                                             reduction='mean')
                loss_dict['masked_lm_loss'] = masked_lm_loss
                
            else:
                evaluation_dict['prediction_scores'] = prediction_scores_mlm
                evaluation_dict['txt_labels'] = txt_labels
    
        if 'unimlmTwo' in task:
            #### forward caption
            # txt_input = multi_tokens
            # txt_lens = multi_tokens.shape[1]
            # txt_labels =  torch.zeros_like(multi_tokens)
            # txt_labels[:,:txt_lens-1] = multi_tokens[:,1:]
            # txt_labels[txt_labels==0] = -1
            if isinstance(video_output,(list,tuple)):
                video_input = []
                for vec in video_output:
                    video_output_mlm = vec.reshape(b, -1, vec.shape[-2], vec.shape[-1])
                    video_input.append(self.get_multimodal_forward_input_video(video_output_mlm))
            else:
                video_output_mlm = video_output.reshape(b, -1, video_output.shape[-2], video_output.shape[-1])
                video_input = self.get_multimodal_forward_input_video(video_output_mlm)    
            txt_input, txt_labels = self.text_masker(multi_tokens)
            caption_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, casual = True)
            caption_output = caption_output[:, :txt_tokens.shape[1], :]
            caption_output = caption_output[txt_labels != -1]
            prediction_scores_caption = self.opt.multimodal_head(caption_output)

            if compute_loss:
                masked_lm_loss_caption = F.cross_entropy(prediction_scores_caption,
                                             txt_labels[txt_labels != -1],
                                             reduction='mean')    
                loss_dict['masked_lm_loss_caption'] =  masked_lm_loss_caption
            else:
                evaluation_dict['prediction_scores_caption'] = prediction_scores_caption
                evaluation_dict['txt_labels_caption'] = txt_labels


        if compute_loss:
            return loss_dict
        else:
            return evaluation_dict


    def contrastive_loss(self, normalized_m1, normalized_m2):

        if self.config.vision_encoder["vision_type"].endswith('clip') and self.txt_cfg.get("text_type",None) is None:
            temp = 1./ self.opt.vision_text_model.logit_scale.exp()
        else:
            temp = self.contra_temp

        if self.contra_sync:
            normalized_m1 = ddp_allgather_with_grads.apply(normalized_m1)
            normalized_m2 = ddp_allgather_with_grads.apply(normalized_m2)
            torch.distributed.barrier()

        score_matrix = torch.matmul(normalized_m1, normalized_m2.permute(1,0))

        score_matrix = score_matrix / temp
        matrix1 = -F.log_softmax(score_matrix, dim=1)
        matrix2 = -F.log_softmax(score_matrix, dim=0)
        
        loss1 = matrix1.diag()
        loss2 = matrix2.diag()
        contra_loss = torch.mean(torch.cat((loss1,loss2), dim=0))
        return contra_loss


