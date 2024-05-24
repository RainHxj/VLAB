import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import *
from torch.nn import LayerNorm as FusedLayerNorm

from .model import VLABModel, VLABPreTrainedModel
from .head import ContrastiveHead
from .utils import txt_token_mask,TextMasker, PseduoTextMasker,TextMaskerWithoutReplace
from utils.distributed import ddp_allgather_with_grads,ddp_allgather
import torch.distributed as dist

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


class VLABForPretraining(VLABPreTrainedModel):
    """ VLAB pretraining """
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
        self.multi_cross=multi_cfg.get("multi_cross", 1)

        self.vision_type = [k for k in video_cfg['pretrained'].keys()]
        self.diff_view = video_cfg.get("diff_view",False)
        self.diff_view_num = len(self.vision_type)
        self.contra_sync = True

        self.mask_type = multi_cfg.get("mask_type", "common_mask")
        ## contra head for contrastive loss
        # if txt_cfg.get("text_type", None) :
        #     self.contra_head_t = ContrastiveHead(self.txt_dim, self.proj_dim)
        # else:
        #     self.contra_head_t = lambda cls_token_t : cls_token_t @ self.opt.vision_text_model.text_projection #
        
        # if video_cfg.get("vision_type",None) == "cb_clip":
        #     self.contra_head_v = lambda cls_token_v : cls_token_v @ self.opt.vision_text_model.visual_proj #ContrastiveHead(self.txt_dim, self.proj_dim)
        # else:
        #     self.contra_head_v = lambda cls_token_v : cls_token_v @ self.opt.vision_text_model.visual.proj #ContrastiveHead(self.txt_dim, self.proj_dim)
        
        # if not self.config.vision_encoder["vision_type"].endswith('clip') and self.txt_cfg.get("text_type",None) is not None:
        #     self.contra_temp = nn.Parameter(torch.tensor(0.07))
        
        # self.contra_head_t = []
        # self.contra_head_v = []

        
        # for i in range(self.multi_cross):
        #     text_proj = self.opt.vision_text_model.models[i].text_projection
        #     vision_proj = self.opt.vision_text_model.models[i].visual.proj
        #     self.contra_head_t.append(lambda cls_token_t : cls_token_t @ text_proj)
        #     self.contra_head_v.append(lambda cls_token_v : cls_token_v @ vision_proj)
            
        ## multimodal encoder

        self.multimodal_dim=multi_cfg["multimodal_dim"]

        self.video_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim))
        self.video_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.video_cfg["sample_num"], self.multimodal_dim))

        if self.mask_type=="common_mask":
            self.text_masker = TextMasker(self.config)
        elif self.mask_type=="pseudo_mask":
            self.text_masker = PseduoTextMasker(self.config)
        elif self.mask_type=="mask_woreplace":
            self.text_masker = TextMaskerWithoutReplace(self.config)

        if isinstance(self.video_dim, int):

            if self.video_dim!=self.multimodal_dim:
                self.proj_multi_input = nn.ModuleList()
                for i in range(self.multi_cross):
                    self.proj_multi_input.append(nn.Sequential( 
                                            nn.Linear(self.video_dim, self.multimodal_dim), 
                                            FusedLayerNorm(self.multimodal_dim, eps=1e-12)) )
        
        else:

            if self.video_dim[0]!=self.multimodal_dim:
                self.proj_multi_input = nn.ModuleList()
                for i in range(self.multi_cross):
                    self.proj_multi_input.append(nn.Sequential( 
                                            nn.Linear(self.video_dim[i], self.multimodal_dim), 
                                            FusedLayerNorm(self.multimodal_dim, eps=1e-12)) )

    def get_multimodal_forward_input_video(self, video_output, batch_size):
        
        assert isinstance(video_output,(list,tuple)), "check video output type"
        b = batch_size

        proj_output_global_local = []
        proj_output_global=[]
        proj_output_local=[]
        if self.diff_view_num>1 and self.diff_view:
            video_frame_embedding_diff_view = torch.chunk(self.video_frame_embedding, self.diff_view_num, dim=1)
        for i, v_data in enumerate(video_output):
            v_data = v_data.reshape(b, -1, v_data.shape[-2], v_data.shape[-1])
            if self.video_dim!=self.multimodal_dim:
                v_data = self.proj_multi_input[i](v_data)
            if self.diff_view_num>1 and self.diff_view:
                v_data =v_data + video_frame_embedding_diff_view[i].unsqueeze(-2)
            else:
                v_data =  v_data + self.video_frame_embedding.unsqueeze(-2)
            v_data_global_local = v_data.reshape(b, -1, self.multimodal_dim) + self.video_type_embeddings
            v_data_global = v_data[:,:,0,:] + self.video_type_embeddings
            v_data_local = v_data.reshape(b, -1, self.multimodal_dim) + self.video_type_embeddings

            proj_output_global_local.append(v_data_global_local)
            proj_output_global.append(v_data_global)
            proj_output_local.append(v_data_local)

        return proj_output_global_local, proj_output_global, proj_output_local

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

        # ipdb.set_trace()

        if 'contraTwo' in task:
            #### forward contra 

            assert isinstance(video_output,(list,tuple)), "please check video_output type"
            assert isinstance(txt_output,(list,tuple)), "please check text_output type"
            # assert len(video_output) == len(txt_output), "please check output"

            for i in range(len(video_output)):
                if len(video_output) != len(txt_output):
                    n=0
                else:
                    n=i
                    
                video_output_contra = video_output[i][:, 0, :].reshape(b, -1, video_output[i].shape[-1])

                cls_token_t = pool_for_contra(txt_output[n],'clip', txt_tokens)
                cls_token_v = pool_for_contra(video_output_contra,'average')  

                # feat_t = cls_token_t @ self.opt.vision_text_model.models[i].text_projection
                if self.vision_type[n].startswith("eva_clip"):
                    feat_t = cls_token_t @ self.opt.vision_text_model.models[i].text.text_projection
                    feat_v = self.opt.vision_text_model.models[i].visual.head(cls_token_v)
                    self.temp = 1./ self.opt.vision_text_model.models[i].text.logit_scale.exp()
                else:
                    feat_t = cls_token_t @ self.opt.vision_text_model.models[n].text_projection
                    feat_v = cls_token_v @ self.opt.vision_text_model.models[n].visual.proj
                    self.temp = 1./ self.opt.vision_text_model.models[n].logit_scale.exp()
                # feat_t =  F.normalize(feat_t,dim=-1)
                feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
                # feat_v =  F.normalize(feat_v,dim=-1)
                feat_v = feat_v / feat_v.norm(dim=-1, keepdim=True)

                # self.temp = 1./ self.opt.vision_text_model.models[i].logit_scale.exp()

                if compute_loss:
                    loss_dict['contra_loss_{}'.format(i)] = self.contrastive_loss(feat_t, feat_v, self.temp)
                    
                else:
                    if i ==0:
                        evaluation_dict['feat_t'] = feat_t
                        evaluation_dict['feat_v'] = feat_v

        if 'mlmTwo' in task:
            #### forward mlmTwo

            assert isinstance(video_output,(list,tuple)), "please check video_output type"
            video_input, video_global, video_patch = self.get_multimodal_forward_input_video(video_output, b)
        
            txt_input, txt_labels = self.text_masker(multi_tokens)
            # mlm_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, casual = False)
            mlm_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, video_global=video_global, video_patch=video_patch, casual = False)

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
            # if isinstance(video_output,(list,tuple)):
            #     video_input = []
            #     for vec in video_output:
            #         video_output_mlm = vec.reshape(b, -1, vec.shape[-2], vec.shape[-1])
            #         video_input.append(self.get_multimodal_forward_input_video(video_output_mlm))
            # else:
            #     video_output_mlm = video_output.reshape(b, -1, video_output.shape[-2], video_output.shape[-1])
            #     video_input = self.get_multimodal_forward_input_video(video_output_mlm)   
            assert isinstance(video_output,(list,tuple)), "please check video_output type"
            video_input, video_global, video_patch = self.get_multimodal_forward_input_video(video_output, b) 
            txt_input, txt_labels = self.text_masker(multi_tokens)
            # caption_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, casual = True)
            caption_output = self.opt.forward_multimodal_encoder(txt_input, video_feat=video_input, video_global=video_global, video_patch=video_patch, casual = True)

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


    def contrastive_loss(self, normalized_m1, normalized_m2, temp):

        # if self.config.vision_encoder["vision_type"].endswith('clip') and self.txt_cfg.get("text_type",None) is None:
        #     temp = 1./ self.opt.vision_text_model.logit_scale.exp()
        # else:
        #     temp = self.contra_temp

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

    def contrastive_loss_local(self, normalized_m1, normalized_m2, temp):
        image_feats_all = ddp_allgather(normalized_m1)
        text_feat_all = ddp_allgather(normalized_m2)
        sim_i2t = torch.matmul(normalized_m1, text_feat_all.permute(1, 0))
        sim_i2t = sim_i2t / temp
        sim_t2i = torch.matmul(normalized_m2, image_feats_all.permute(1, 0))
        sim_t2i = sim_t2i / temp
        rank = dist.get_rank()
        bs = normalized_m1.size(0)

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(normalized_m1.device)
        loss_itc = (
                           F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                           + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                   ) / 2
        return loss_itc



