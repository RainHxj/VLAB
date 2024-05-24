from pickle import NONE
import torch
from torch import nn
from .videoswin import SwinTransformer3D
from .clip import build_model
from .clip import Transformer
from .hf_clip import build_model_from_openai_state_dict, CLIP
from .eva_model import build_eva_model_and_transforms
from .adapter_eva_model import build_adapter_eva_model_and_transforms
from torch.nn import LayerNorm as FusedLayerNorm
from utils.logger import LOGGER
from .video_adapter import build_adapter
import json
import ipdb
import torch.nn.functional as F

def builder(cfg):
    
    vision_type=cfg["vision_type"]
    pretrained_path = cfg.get("pretrained",None)
    resolution = cfg.get("resolution",224)
    if vision_type == "video_swin":
        vision_encoder = SwinTransformer3D(vision_cfg)
        
        if pretrained_path:
            videoswin_weight = torch.load(pretrained_path,map_location='cpu')
            missing_keys, unexpected_keys = vision_encoder.load_state_dict(videoswin_weight, strict=False)
            del(videoswin_weight)

        #LOGGER.info(f'missing_keys in video encoder: {missing_keys}')
        LOGGER.info(f'unexpected_keys in video encoder: {unexpected_keys}')

        return vision_encoder
        
    elif vision_type == "clip":
        assert pretrained_path is not None, "PLEASE CHECK CLIP CHECKPOINT!!" 
        clip_weight = torch.jit.load(pretrained_path, map_location="cpu")
        clip_weight = clip_weight.state_dict()
        clip_model = build_model(clip_weight, resolution).float()
        del(clip_weight)
        return (clip_model, )

    elif vision_type == "our_clip":
        assert pretrained_path is not None, "PLEASE CHECK CLIP CHECKPOINT!!"
        clip_weight = torch.load(pretrained_path, map_location="cpu")
        clip_model = build_model(clip_weight).float()
        del(clip_weight)
        return (clip_model, )
    elif vision_type == "hf_clip":
        assert pretrained_path is not None, "PLEASE CHECK CLIP CHECKPOINT!!"
        pretrained_path=cfg["pretrained"]
        config=cfg["config"]
        model_cfg = json.load(open(config))
        clip_model = CLIP(**model_cfg)
        if cfg.get("checkpointing", None):
            clip_model.set_grad_checkpointing()
        hf_clip_weight = torch.load(pretrained_path, map_location="cpu")
        missing_keys, unexpected_keys = clip_model.load_state_dict(hf_clip_weight, strict=False)
        LOGGER.info(f'unexpected_keys in hf_clip: {unexpected_keys}')
        LOGGER.info(f'missing_keys in hf_clip: {missing_keys}')
        # clip_model = build_model_from_openai_state_dict(hf_clip_weight).float()
        del(hf_clip_weight)
        return (clip_model, )
    elif vision_type == "cb_clip":
        from .cb_clip import build_cb_model
        assert pretrained_path is not None, "PLEASE CHECK CLIP CHECKPOINT!!"
        paths = cfg["pretrained"]
        grad_checkpointing = cfg.get("grad_checkpointing", False)
        frozen_vision = cfg.get("frozen_vision", False)
        diff_view = cfg.get("diff_view", False)
        adaptor= cfg.get("adaptor", [])
        wo_text = cfg.get("wo_text", False)

        ### adapter
        adapt_type = cfg.get("adapt_type", None)
        start_adapt_layer = cfg.get('start_adapt_layer', -1)
        adapt_cls_dim = cfg.get('adapt_cls_dim', 64)
        adapt_patch_dim = cfg.get('adapt_patch_dim', 16)
        adapt_order = cfg.get('adapt_order', 'adapt_last')
        batch_size = cfg.get("batch_size", 16)

        assert isinstance(adaptor, list), "adaptor in config must be list"
        assert isinstance(paths, dict), "pretrained path must be list"
        clip_weights = []
        for model_type, path in paths.items():
            if model_type.startswith("clip"):
                clip_weight = torch.jit.load(path, map_location="cpu")
                clip_weight = clip_weight.state_dict()
                clip_weights.append(clip_weight)
            elif model_type.startswith("zc_clip"):
                clip_weight = torch.load(path, map_location="cpu")
                if "state_dict" in clip_weight.keys():
                    clip_weight = clip_weight["state_dict"]
                clip_weights.append(clip_weight)
            
            elif model_type.startswith("videoclip"):
                clip_weight = torch.jit.load(pretrained_path, map_location="cpu").state_dict()
                clip_model = build_adaptor(clip_weight, rolling_ratio = 0, use_checkpoint=use_checkpoint).float()

            elif model_type.startswith("swin"):
                vision_encoder = SwinTransformer3D(vision_cfg)
                
                if pretrained_path:
                    videoswin_weight = torch.load(pretrained_path,map_location='cpu')
                    missing_keys, unexpected_keys = vision_encoder.load_state_dict(videoswin_weight, strict=False)
                    del(videoswin_weight)

                #LOGGER.info(f'missing_keys in video encoder: {missing_keys}')
                LOGGER.info(f'unexpected_keys in video encoder: {unexpected_keys}')

            elif model_type.startswith("eva_clip"):
                clip_weight, _ = build_eva_model_and_transforms("EVA_CLIP_g_14", pretrained=path, image_size=cfg['resolution'])
                if grad_checkpointing:
                    clip_weight.set_grad_checkpointing()
                # frozen_vision=False
                # if frozen_vision:
                #     LOGGER.info(f'all vision parameters will be frozen.')
                #     pn=0
                #     for p in clip_weight.visual.parameters():
                #         pn+=1
                #         p.requires_grad=False
                #     LOGGER.info(f'{pn} vision parameters are frozen. Done!')
                clip_weights.append(clip_weight)

            elif model_type.startswith("adapter_eva"):
                print("xxxxxxxxxx")
                print(start_adapt_layer)
                clip_weight, _ = build_adapter_eva_model_and_transforms(
                    "EVA_CLIP_g_14",
                    pretrained=path,
                    adapt=adapt_type,
                    batch_size=batch_size,
                    start_adapt_layer=start_adapt_layer,
                    adapt_order=adapt_order,
                    adapt_cls_dim=adapt_cls_dim,
                    adapt_patch_dim=adapt_patch_dim,
                    grad_checkpointing=grad_checkpointing)
                if grad_checkpointing:
                    clip_weight.set_grad_checkpointing()
                if frozen_vision:
                    LOGGER.info(f'all vision parameters will be frozen.')
                    frozen_layer = cfg.get("frozen_layer", 100)
                    pn = 0
                    for n, p in clip_weight.visual.named_parameters():
                        # if 'blocks' in n:
                        #     layer = int(n.split('blocks.')[1][0])
                        # else:
                        #     layer = 100

                        # if 'adapt' not in n and layer < frozen_layer:
                        #     pn += 1
                        p.requires_grad = False
                    LOGGER.info(f'{pn} vision parameters are frozen. Done!')
                clip_weights.append(clip_weight)


            elif model_type.startswith("video_adapter"):
                assert path is not None, "PLEASE CHECK CLIP CHECKPOINT!!"
                clip_weight = torch.load(path,
                                             map_location="cpu")['state_dict']
                
                resolution = cfg['resolution']
                checkpoint = clip_weight
                for key, value in checkpoint.items():
                    if 'positional_embedding' in key and 'visual' in key:
                        w = checkpoint[key]
                        src_cls = w[0:1]
                        src_oth = w[1:]
                        grid = resolution // 14
                        dim = w.size(-1)
                        src_oth = F.interpolate(src_oth.reshape(16,16,dim).permute(2,0,1).unsqueeze(0), (grid, grid), mode='bilinear') # any
                        src_oth = src_oth[0].permute(1,2,0).reshape(-1,dim)
                        tgt = torch.cat((src_cls,src_oth),dim=0)
                        checkpoint[key] = tgt

                clip_weight = checkpoint

                # clip_weight = torch.load(path,
                                            #  map_location="cpu").state_dict()
                start_adapt_layer = cfg.get("start_adapt_layer", -1)
                adapt_order = cfg.get("adapt_order", 'adapt_last')
                adapt_cls_dim = cfg.get("adapt_cls_dim", 64)
                adapt_patch_dim = cfg.get("adapt_patch_dim",16)

                ## frozen vision encoder

                ## grad checkpointing
                
                clip_model = build_adapter(clip_weight,
                                           rolling_ratio=0,
                                           use_checkpoint=grad_checkpointing,
                                           start_adapt_layer=start_adapt_layer,
                                           adapt_order=adapt_order,
                                           adapt_cls_dim=adapt_cls_dim,
                                           adapt_patch_dim=adapt_patch_dim,
                                           frozen_vision=frozen_vision)
                del (clip_weight)
                clip_weights.append(clip_model)


        frozen_vision=False
        clip_model = build_cb_model(clip_weights, resolution, grad_checkpointing, frozen_vision, diff_view, adaptor).float()

        return (clip_model, )
        

    else:
        raise NotImplementedError



class Video_Token_Embeddings(nn.Module):
    def __init__(self, model_cfg_multimodal):
        super().__init__()
        self.token_embedding = nn.Embedding(model_cfg_multimodal.video_codebook_size + 1, model_cfg_multimodal.hidden_size)
        self.frame_num  = 10
        self.tokens_per_frame = 65
        self.position_embeddings = nn.Embedding(self.tokens_per_frame, model_cfg_multimodal.hidden_size)
        self.frame_embeddings = nn.Embedding(self.frame_num , model_cfg_multimodal.hidden_size)

    def forward(self,tokens):
        ### vqvae_tokens b, 650
        embeddings = self.token_embedding(tokens)
        video_pos_ids = list(range(self.tokens_per_frame)) * self.frame_num
        video_pos_ids = torch.tensor(video_pos_ids, dtype=torch.long, device=tokens.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(video_pos_ids)

        frame_ids = [i for i in range(self.frame_num) for j in range(self.tokens_per_frame)]
        frame_ids = torch.tensor(frame_ids, dtype=torch.long, device=tokens.device).unsqueeze(0)
        position_embeddings += self.frame_embeddings(frame_ids)
        embeddings += position_embeddings[:,:embeddings.shape[1]]

        return embeddings