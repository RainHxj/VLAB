# class TextMasker(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.mask_token = config.losses["mask_token"]
#         self.range = [106, config.multi_encoder["vocab_size"]]
#         self.mask_prob = config.losses["mask_prob"]

#     def forward(self, txt_tokens,  txt_mask_indicator = None):
#         # txt_tokens = txt_tokens['bert_tokens']
#         txt_tokens = txt_tokens.clone() ### important, must have

#         txt_tokens, txt_labels = self.perform_mask(txt_tokens, txt_mask_indicator=txt_mask_indicator)
#         return txt_tokens, txt_labels

    
#     def perform_mask(self, txt_tokens, txt_mask_indicator=None):


import numpy as np
import ipdb
txt_tokens = np.array([[101,45,23,452,567,22,1,102,0,0],[102,45,23,452,567,22,1,102,0,0]])


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
        
        if txt_tokens[i][j]!=0:
            tmp.append(103)
            tmp.append(txt_tokens[i][j])
            label.append(txt_tokens[i][j])
            label.append(-1)
            pos.append(j)
            pos.append(j)
        else: 
            tmp.append(txt_tokens[i][j])
            label.append(-1)
            pos.append(j)
    new_txt_tokens_list.append(tmp)
    label_list.append(label)
    pos_list.append(pos)

ipdb.set_trace()

new_txt_tokens = np.array(new_txt_tokens_list)
labels = np.array(label_list)
pos_ids = np.array(pos_list)

return new_txt_tokens, labels, pos_ids



# new_txt_tokens =torch.from_numpy(new_txt_tokens).long().cuda()
# labels =torch.from_numpy(labels).long().cuda()

# return new_txt_tokens, labels

       
                
                
