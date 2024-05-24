import ipdb
import torch
from torch import nn
from model.txt_encoders.bert import BertAttention

# 1. concat
# 2. stack
# 3. scale_sum
# 4. sum
# 5. share_sum
# 5. moe 
# 6. sa

class CrossEnsemble(nn.Module):
    def __init__(self, config, multi_cross, ensemble_type="concat"):
        super(CrossEnsemble, self).__init__()

        self.multi_cross=multi_cross
        self.ensemble_type=ensemble_type

        if ensemble_type=="concat":
            self.cross_attn = BertAttention(config, attn_type='cross')

        elif ensemble_type=="stack":
                self.cross_attn = nn.ModuleList()
                for i in range(self.multi_cross):
                    self.cross_attn.append(BertAttention(config, attn_type='cross'))            

        elif ensemble_type=="scale_sum":
            self.scales = nn.Parameter(torch.ones(multi_cross))
            for i in range(self.multi_cross):
                    self.cross_attn.append(BertAttention(config, attn_type='cross'))      
    
        elif ensemble_type=="sum":
            for i in range(self.multi_cross):
                self.cross_attn.append(BertAttention(config, attn_type='cross'))   
        
        elif ensemble_type=="share_sum":
            self.cross_attn = BertAttention(config, attn_type='cross')
        
        elif ensemble_type=="moe":
            for i in range(self.multi_cross):
                self.cross_attn.append(BertAttention(config, attn_type='cross'))     
        
        elif ensemble_type=="sa":
            for i in range(self.multi_cross):
                self.cross_attn.append(BertAttention(config, attn_type='cross'))  
        
        else:
            raise NotImplementedError


    def forward(self, att_feat, vision_feat):
        
        if ensemble_type=="concat":
            vision_feat = torch.cat(vision_feat, dim=1)
            output = self.cross_attn(att_feat, None, vision_feat)


        elif ensemble_type=="stack":
            
            for i in range(self.multi_cross):
                att_feat = self.cross_attn[i](att_feat, None, vision_feat[i])
            output=att_feat

        elif ensemble_type=="scale_sum":
            tmp = torch.zeros_like(att_feat)
            for i in range(self.multi_cross): 
                tmp+=self.cross_attn[i](att_feat, None, vision_feat[i])*self.scales[i]
            output=tmp
    
        elif ensemble_type=="sum":
            tmp = torch.zeros_like(att_feat)
            for i in range(self.multi_cross): 
                tmp+=self.cross_attn[i](att_feat, None, vision_feat[i])
            output=tmp

        elif ensemble_type=="share_sum":
            for i in range(self.multi_cross): 
                tmp+=self.cross_attn(att_feat, None, vision_feat[i])
            output=tmp

        elif ensemble_type=="moe":
            pass
        
        elif ensemble_type=="sa":
            pass
        
        else:
            raise NotImplementedError