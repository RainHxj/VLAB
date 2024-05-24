"""

Licensed under the MIT license.

VLAB for retrieval model
"""
from model.cb_pretrain import VLABForPretraining
import torch.nn.functional as F
from model.pretrain import pool_for_contra

class VLABForVideoRetrieval(VLABForPretraining):

    def __init__(self, config):
        super().__init__(config)
        self.local_loss = config.vision_encoder.get("local_loss",False)
        
        self.frozen()

    def frozen(self, ):
        
        del self.video_type_embeddings
        del self.video_frame_embedding
        
        del self.opt.multimodal_encoder
        del self.opt.multimodal_head
        if hasattr(self, "proj_multi_input"):
            del self.proj_multi_input
        
    def forward(self, batch, compute_loss=True):

        evaluation_dict = {}
        loss_dict={}
        sample_num_2m = len(batch['batch_2m']['ids'])


        assert sample_num_2m > 0, 'empty batch'
        txt_tokens = batch['batch_2m']['txt_tokens']
        video_pixels = batch['batch_2m']['video_pixels']

        txt_output = self.opt.forward_txt_encoder(txt_tokens)


        b,n,_,h,w = video_pixels.shape
        video_output = self.opt.forward_video_encoder(video_pixels)


        assert isinstance(video_output,(list,tuple)), "please check video_output type"
        assert isinstance(txt_output,(list,tuple)), "please check text_output type"
        assert len(video_output) == len(txt_output), "please check output"

        if len(video_output)>1:
            evaluation_dict['feat_t']=[]
            evaluation_dict['feat_v']=[]

        for i in range(len(video_output)):
            
            # ipdb.set_trace()
            video_output_contra = video_output[i][:, 0, :].reshape(b, -1, video_output[i].shape[-1])

            cls_token_t = pool_for_contra(txt_output[i],'clip', txt_tokens)
            cls_token_v = pool_for_contra(video_output_contra,'average')  

            # feat_t = cls_token_t @ self.opt.vision_text_model.models[i].text_projection
            if self.vision_type[i].startswith("eva_clip"):
                feat_t = cls_token_t @ self.opt.vision_text_model.models[i].text.text_projection
                feat_v = self.opt.vision_text_model.models[i].visual.head(cls_token_v)
                self.temp = 1./ self.opt.vision_text_model.models[i].text.logit_scale.exp()
            else:
                feat_t = cls_token_t @ self.opt.vision_text_model.models[i].text_projection
                feat_v = cls_token_v @ self.opt.vision_text_model.models[i].visual.proj
                self.temp = 1./ self.opt.vision_text_model.models[i].logit_scale.exp()
            # feat_t =  F.normalize(feat_t,dim=-1)
            feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
            # feat_v =  F.normalize(feat_v,dim=-1)
            feat_v = feat_v / feat_v.norm(dim=-1, keepdim=True)

            # self.temp = 1./ self.opt.vision_text_model.models[i].logit_scale.exp()

            if compute_loss:
                if not self.local_loss:

                    loss_dict['contra_loss_{}'.format(i)] = self.contrastive_loss(feat_t, feat_v, self.temp)
                else:
                    loss_dict['contra_loss_{}'.format(i)] = self.contrastive_loss_local(feat_t, feat_v, self.temp)
            else:

                if len(video_output) >1:
                    evaluation_dict['feat_t'].append(feat_t)
                    evaluation_dict['feat_v'].append(feat_v)
                else:
                    if i ==0:        
                        evaluation_dict['feat_t'] = feat_t
                        evaluation_dict['feat_v'] = feat_v
                
        if compute_loss:
            return loss_dict
        else:    
            return evaluation_dict





        '''
        # video_output = video_output[:, 0, :].reshape(b, -1, video_output.shape[-1])
        if isinstance(video_output,(list,tuple)): 
            video_output = video_output[0][:, 0, :].reshape(b, -1, video_output[0].shape[-1])
        else:
            video_output = video_output[:, 0, :].reshape(b, -1, video_output.shape[-1])
        # video_output = video_output.reshape(b,-1,video_output.shape[-1])
        # batch_size,_,hidden_size = video_output.shape
        # video_output = video_output.reshape(b,-1,video_output.shape[-1])

        if self.txt_cfg.get("text_type",None):
            if isinstance(txt_output,(list,tuple)):
                cls_token_t = pool_for_contra(txt_output[0],'cls')
            else:
                cls_token_t = pool_for_contra(txt_output,'cls')
        else:
            if isinstance(txt_output,(list,tuple)):
                cls_token_t = pool_for_contra(txt_output[0],'clip', txt_tokens)
            else:
                cls_token_t = pool_for_contra(txt_output,'clip', txt_tokens)
        cls_token_v = pool_for_contra(video_output,'average')
        feat_t = self.contra_head_t(cls_token_t)
        # feat_t = F.normalize(feat_t,dim=-1)
        feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
        feat_v = self.contra_head_v(cls_token_v)
        # feat_v = F.normalize(feat_v,dim=-1)
        feat_v = feat_v / feat_v.norm(dim=-1, keepdim=True)




        if self.vision_type[i].startswith("eva_clip"):
            feat_t = cls_token_t @ self.opt.vision_text_model.models[i].text.text_projection
            feat_v = self.opt.vision_text_model.models[i].visual.head(cls_token_v)
            self.temp = 1./ self.opt.vision_text_model.models[i].text.logit_scale.exp()
        else:
            feat_t = cls_token_t @ self.opt.vision_text_model.models[i].text_projection
            feat_v = cls_token_v @ self.opt.vision_text_model.models[i].visual.proj


            feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
            feat_v = feat_v / feat_v.norm(dim=-1, keepdim=True)

            if compute_loss:
                loss_dict['contra_loss_{}'.format(i)] = self.contrastive_loss(feat_t, feat_v, self.temp)


        if compute_loss:
            loss = self.contrastive_loss(feat_t, feat_v)
            loss_dict = {'ret_loss':loss}
            return loss_dict

        else:
            evaluation_dict = {}
            evaluation_dict['feat_t'] = feat_t
            evaluation_dict['feat_v'] = feat_v

            return evaluation_dict
        '''
    
    
    
    
    

