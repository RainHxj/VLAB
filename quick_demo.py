import torch
from model.vision_encoders.eva_vit import VisionTransformer


vit_giant = VisionTransformer(
    img_size=224, 
    patch_size=14, 
    num_classes=1024, 
    use_mean_pooling=False, 
    init_values=None,
    embed_dim=1408,
    depth=40,
    num_heads=16,
    mlp_ratio=4.3637,
    qkv_bias=True,
    drop_path_rate=0.4,
    adapt=None,
    batch_size=1,
    start_adapt_layer=-1,
    adapt_order='adapt_last',
    adapt_cls_dim=64,
    adapt_patch_dim=16,
    grad_checkpointing=False
)

ckpt = torch.load('eva_vit_giant.ckpt')

# for k, v in vit_giant.named_parameters():
#     print(k, v.shape)

# print(f'====='*4)

# for k, v in ckpt.items():
#     print(k, v.shape)

load = vit_giant.load_state_dict(ckpt)
print(load)
input = torch.randn(1,3,224,224)
out = vit_giant(input)
print(out.shape)

