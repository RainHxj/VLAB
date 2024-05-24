import torch
import argparse
from collections import OrderedDict

def convert_model(model,out_name=None):
    model_weight = torch.load(model)
    model_weight = model_weight["state_dict"]
    models_clip = OrderedDict()
    models_bert = OrderedDict()


    for key,value in model_weight.items():
        if key.startswith('vbackbone.vision_model.embeddings.class_embedding'):
            key_new= "visual.class_embedding"
            models_clip[key_new] = value
        elif key.startswith('vbackbone.vision_model.embeddings.position_embedding.weight'):
            key_new =  'visual.positional_embedding'
            models_clip[key_new] = value
        elif key.startswith('vbackbone.vision_model.embeddings.patch_embedding.weight'):
            key_new =  'visual.conv1.weight'
            models_clip[key_new] = value
        elif key.startswith('vbackbone.vision_model.pre_layrnorm.weight'): 
            key_new = 'visual.ln_pre.weight'
            models_clip[key_new] = value
        elif key.startswith('vbackbone.vision_model.pre_layrnorm.bias'):
            key_new =  'visual.ln_pre.bias'
            models_clip[key_new] = value
        elif key.startswith("vbackbone.visual_projection.weight"):
            key_new = "visual.proj"
            models_clip[key_new] = value.T

        elif key.startswith("vbackbone.vision_model.post_layernorm.weight"):
            key_new = "visual.ln_post.weight"
            models_clip[key_new] = value 
        elif key.startswith("vbackbone.vision_model.post_layernorm.bias"):
            key_new = "visual.ln_post.bias"
            models_clip[key_new] = value                
        elif key.startswith("head.logit_scale"):
            key_new = "logit_scale"
            models_clip[key_new] = value
        

        elif key.startswith("lbackbone.text_model.embeddings.position_embedding.weight"):
            key_new = "positional_embedding"
            models_clip[key_new] = value
        elif key.startswith("lbackbone.text_projection.weight"):
            key_new = "text_projection"
            models_clip[key_new] = value.T
        elif key.startswith("lbackbone.text_model.final_layer_norm.weight"):
            key_new = "ln_final.weight"
            models_clip[key_new] = value
        elif key.startswith("lbackbone.text_model.final_layer_norm.bias"):
            key_new = "ln_final.bias"
            models_clip[key_new] = value
        elif key.startswith("lbackbone.text_model.embeddings.token_embedding.weight"):
            key_new = "token_embedding.weight"
            models_clip[key_new] = value


        elif key.startswith('vbackbone.vision_model.encoder.layers'):
            num_layers = key.split(".")[4]

            if "self_attn.k_proj" in key or "self_attn.v_proj" in key:
                pass

            elif "self_attn.q_proj.weight" in key:
                q_w = model_weight['vbackbone.vision_model.encoder.layers.{}.self_attn.q_proj.weight'.format(num_layers)]
                k_w = model_weight['vbackbone.vision_model.encoder.layers.{}.self_attn.k_proj.weight'.format(num_layers)]
                v_w = model_weight['vbackbone.vision_model.encoder.layers.{}.self_attn.v_proj.weight'.format(num_layers)]                
                qkv_w  = torch.cat([q_w,k_w,v_w],dim=0)

                new_key= "visual.transformer.resblocks.{}.attn.in_proj_weight".format(num_layers)
                models_clip[new_key] = qkv_w

            elif "self_attn.q_proj.bias" in key:
                q_b = model_weight['vbackbone.vision_model.encoder.layers.{}.self_attn.q_proj.bias'.format(num_layers)]
                k_b = model_weight['vbackbone.vision_model.encoder.layers.{}.self_attn.k_proj.bias'.format(num_layers)]
                v_b = model_weight['vbackbone.vision_model.encoder.layers.{}.self_attn.v_proj.bias'.format(num_layers)]  
                qkv_b = torch.cat([q_b,k_b,v_b],dim=0) 
                new_key = "visual.transformer.resblocks.{}.attn.in_proj_bias".format(num_layers)
                models_clip[new_key] = qkv_b

            elif "self_attn.out_proj.weight" in key:
                new_key= "visual.transformer.resblocks.{}.attn.out_proj.weight".format(num_layers)
                models_clip[new_key]=value
            elif "self_attn.out_proj.bias" in key:
                new_key = "visual.transformer.resblocks.{}.attn.out_proj.bias".format(num_layers)
                models_clip[new_key] = value

            elif "layer_norm1.weight" in key:
                new_key = "visual.transformer.resblocks.{}.ln_1.weight".format(num_layers)
                models_clip[new_key] = value
            elif "layer_norm1.bias" in key:
                new_key = "visual.transformer.resblocks.{}.ln_1.bias".format(num_layers)
                models_clip[new_key] = value

            elif "mlp.fc1.weight" in key:
                new_key = "visual.transformer.resblocks.{}.mlp.c_fc.weight".format(num_layers)
                models_clip[new_key] = value
            elif 'mlp.fc1.bias' in key:
                new_key = "visual.transformer.resblocks.{}.mlp.c_fc.bias".format(num_layers)
                models_clip[new_key] = value
            elif 'mlp.fc2.weight' in key:
                new_key = "visual.transformer.resblocks.{}.mlp.c_proj.weight".format(num_layers)
                models_clip[new_key] = value
            elif 'mlp.fc2.bias' in key:
                new_key = "visual.transformer.resblocks.{}.mlp.c_proj.bias".format(num_layers)
                models_clip[new_key] = value
            elif "layer_norm2.weight" in key:
                new_key = "visual.transformer.resblocks.{}.ln_2.weight".format(num_layers)
                models_clip[new_key] = value
            elif "layer_norm2.bias" in key:
                new_key = "visual.transformer.resblocks.{}.ln_2.bias".format(num_layers)
                models_clip[new_key] = value
            else:
                print(key)


        elif key.startswith('lbackbone.text_model.encoder.layers'):
            num_layers = key.split(".")[4]

            if "self_attn.k_proj" in key or "self_attn.v_proj" in key:
                pass

            elif "self_attn.q_proj.weight" in key:
                q_w = model_weight['lbackbone.text_model.encoder.layers.{}.self_attn.q_proj.weight'.format(num_layers)]
                k_w = model_weight['lbackbone.text_model.encoder.layers.{}.self_attn.k_proj.weight'.format(num_layers)]
                v_w = model_weight['lbackbone.text_model.encoder.layers.{}.self_attn.v_proj.weight'.format(num_layers)]                
                qkv_w  = torch.cat([q_w,k_w,v_w],dim=0)

                new_key= "transformer.resblocks.{}.attn.in_proj_weight".format(num_layers)
                models_clip[new_key] = qkv_w

            elif "self_attn.q_proj.bias" in key:
                q_b = model_weight['lbackbone.text_model.encoder.layers.{}.self_attn.q_proj.bias'.format(num_layers)]
                k_b = model_weight['lbackbone.text_model.encoder.layers.{}.self_attn.k_proj.bias'.format(num_layers)]
                v_b = model_weight['lbackbone.text_model.encoder.layers.{}.self_attn.v_proj.bias'.format(num_layers)]  
                qkv_b = torch.cat([q_b,k_b,v_b],dim=0) 
                new_key = "transformer.resblocks.{}.attn.in_proj_bias".format(num_layers)
                models_clip[new_key] = qkv_b

            elif "self_attn.out_proj.weight" in key:
                new_key= "transformer.resblocks.{}.attn.out_proj.weight".format(num_layers)
                models_clip[new_key]=value
            elif "self_attn.out_proj.bias" in key:
                new_key = "transformer.resblocks.{}.attn.out_proj.bias".format(num_layers)
                models_clip[new_key] = value

            elif "layer_norm1.weight" in key:
                new_key = "transformer.resblocks.{}.ln_1.weight".format(num_layers)
                models_clip[new_key] = value
            elif "layer_norm1.bias" in key:
                new_key = "transformer.resblocks.{}.ln_1.bias".format(num_layers)
                models_clip[new_key] = value

            elif "mlp.fc1.weight" in key:
                new_key = "transformer.resblocks.{}.mlp.c_fc.weight".format(num_layers)
                models_clip[new_key] = value
            elif 'mlp.fc1.bias' in key:
                new_key = "transformer.resblocks.{}.mlp.c_fc.bias".format(num_layers)
                models_clip[new_key] = value
            elif 'mlp.fc2.weight' in key:
                new_key = "transformer.resblocks.{}.mlp.c_proj.weight".format(num_layers)
                models_clip[new_key] = value
            elif 'mlp.fc2.bias' in key:
                new_key = "transformer.resblocks.{}.mlp.c_proj.bias".format(num_layers)
                models_clip[new_key] = value
            elif "layer_norm2.weight" in key:
                new_key = "transformer.resblocks.{}.ln_2.weight".format(num_layers)
                models_clip[new_key] = value
            elif "layer_norm2.bias" in key:
                new_key = "transformer.resblocks.{}.ln_2.bias".format(num_layers)
                models_clip[new_key] = value
            else:
                print(key)
        else:
            print(key)


    clip_weight = torch.jit.load('/opt/tiger/fake_arnold/hexingjian/VLP/output/pretrained_weights/clip/ViT-L-14-336px.pt')
    clip_weight = clip_weight.state_dict()

    models_clip['input_resolution'] = clip_weight['input_resolution']
    models_clip['context_length'] = clip_weight['context_length']
    models_clip['vocab_size'] = clip_weight['vocab_size']

    clip_out={"state_dict":models_clip,"author":"svlp"}
    torch.save(clip_out, "{}.pth".format("vlp_clip_large_coyo"))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint from huggingface')
    parser.add_argument('in_file', help='input checkpoint filename')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=parse_args()
    convert_model(args.in_file)