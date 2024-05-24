import torch
from collections import OrderedDict

# path = "/PATH/VLP/output/pretrained_weights/mf_clip/pretrain-webvid10m-va-512-512-last4-parallel_large_stage2_5witer/ckpt/model_step_50000.pt"
path = "/PATH/VLP/output/pretrained_weights/zc_clip/pretrain-s2-coyo150m_cc4m_cc12m-adaptor_coyo150m_cc4-12m_4ada_cliplarge-2_128gpus/ckpt/model_step_112832_last.pt"
weight = torch.load(path)

models_weight = OrderedDict()

# for na in weight.keys():
#     if 'opt.multimodal_encoder' not in na:
#         new_na = na.replace('opt.vision_text_model.','')

#         models_weight[new_na] = weight[na]

# clip_out={"state_dict":models_weight}
    
# torch.save(clip_out, "{}.pth".format("/PATH/VLP/output/pretrained_weights/mf_clip/pretrain-webvid10m-va-512-512-last4-parallel_large_stage2_5witer/ckpt/model_opt"))


for na in weight.keys():
    if 'opt.multimodal_encoder' not in na and 'opt.multimodal_head' not in na: 
        new_na = na.replace('opt.vision_text_model.','').replace("models.0.",'')

        if new_na not in ["video_type_embeddings", "video_frame_embedding", "proj_multi_input.0.0.weight", "proj_multi_input.0.0.bias", "proj_multi_input.0.1.weight", "proj_multi_input.0.1.bias"]:

            models_weight[new_na] = weight[na]

clip_out={"state_dict":models_weight}
    
torch.save(clip_out, "{}.pth".format("/PATH/VLP/output/pretrained_weights/zc_clip/pretrain-s2-coyo150m_cc4m_cc12m-adaptor_coyo150m_cc4-12m_4ada_cliplarge-2_128gpus/ckpt/model_opt"))