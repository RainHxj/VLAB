import torch
from torch import nn
from .clip import CLIP
from .clip import build_model

import ipdb

class CBCLIP(nn.Module):
    def __init__(self, models, diff_view=False):
        super().__init__()
    
        self.models = nn.ModuleList(models)
        self.diff_view = diff_view
        # self.logit_scale = models[0].logit_scale
        # self.text_projection = models[0].text_projection
        # self.visual_proj = models[0].visual.proj
        
        # # ## del some unused parameters
        # self.frozen()
        
    def encode_image(self, image):
        

        img=[]
        if self.diff_view:
            img = torch.chunk(image, len(self.models), dim=1)
        else:
            img=[image for i in range(len(self.models))]
        assert len(img)==len(self.models), "check image list for ensemble clip"
        if len(img)==2 and img[0].shape[1]<4:
            img[0]=img[0][:,[0],:,:,:]
        outputs = []
        # img = torch.chunk(image, 2, dim=1)
        # print("img shape {}".format(img[0].shape))
        for i, sub_model in enumerate(self.models):
            
            out = sub_model.encode_image(img[i])

            if isinstance(out,(list,tuple)):
                outputs.extend(out)
            else:
                outputs.append(out)

        return outputs

        # cls_token = outputs[0][:,[0]]
        # patch_tokens = outputs[0][:,1:]
        # for output in outputs[1:]:
        #     cls_token = cls_token + output[:,[0]]
        #     # patch_tokens = torch.cat([patch_tokens, output[:,1:]],dim=1)
        #     patch_tokens = patch_tokens + output[:,1:]

        # cls_token = cls_token/len(outputs)
        # patch_tokens = patch_tokens/len(outputs)
        # return torch.cat([cls_token, patch_tokens],dim=1)
        # return outputs

    def encode_text(self, text):

        outputs=[]
        for i,sub_model in enumerate(self.models):
            
            out = sub_model.encode_text(text)
            outputs.append(out)

        return outputs           
        
        # return self.models[0].encode_text(text)

    def forward(self, image, text):
        pass

def build_cb_model(state_dicts, resolution=224, grad_checkpointing=False,frozen_vision=False, diff_view=False, adaptor=[]):
    
    print("#######################")
    print(len(state_dicts))
    if len(adaptor)==0:
        adaptor=[False]*len(state_dicts)

    models = []
    for i, state_dict in enumerate(state_dicts):
        print("model type {}".format(type(state_dict)))
        if not isinstance(state_dict, dict):
            models.append(state_dict) 
        else:
            model = build_model(state_dict, resolution, grad_checkpointing, frozen_vision, adaptor[i])
            models.append(model)

    cb_model = CBCLIP(models, diff_view)

    return cb_model

