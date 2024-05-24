import torch
import os
import numpy as np



### v2.0-singlevd contains a video encoder(VIT-base-32) and a multimodal encoder(bert-base-uncased)


###process ViT-16 weight 

def trans(x):
    return torch.from_numpy(x)
opt_weight={}
vit_weight = np.load("./output/pretrianed_weights/ViT-B_16.npz")

opt_weight['opt.video_embeddings.cls_token']  = trans(vit_weight['cls'])
opt_weight['opt.video_embeddings.first_conv.weight'] =  trans(vit_weight['embedding/kernel']).permute(3,2,0,1)  ### need to permute?
opt_weight['opt.video_embeddings.first_conv.bias'] = trans(vit_weight['embedding/bias'])
opt_weight['opt.video_embeddings.position_embeddings.weight'] = trans(vit_weight['Transformer/posembed_input/pos_embedding']).squeeze()
#'opt.video_embeddings.mask_embedding.weight', 
#'opt.video_embeddings.layernorm.weight', 
#'opt.video_embeddings.layernorm.bias'


for  i in range(12):
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_space.linears.0.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/kernel']).reshape(768,-1).permute(1,0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_space.linears.0.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/bias']).reshape(768)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_space.linears.1.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/kernel']).reshape(768,-1).permute(1,0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_space.linears.1.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/bias']).reshape(768)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_space.linears.2.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/kernel']).reshape(768,-1).permute(1,0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_space.linears.2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/bias']).reshape(768)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_space.linears.3.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/kernel']).reshape(-1,768).permute(1,0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_space.linears.3.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/bias'])
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_time.linears.0.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/kernel']).reshape(768,-1).permute(1,0).fill_(0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_time.linears.0.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/bias']).reshape(768).fill_(0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_time.linears.1.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/kernel']).reshape(768,-1).permute(1,0).fill_(0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_time.linears.1.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/bias']).reshape(768).fill_(0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_time.linears.2.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/kernel']).reshape(768,-1).permute(1,0).fill_(0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_time.linears.2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/bias']).reshape(768).fill_(0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_time.linears.3.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/kernel']).reshape(-1,768).permute(1,0).fill_(0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.attention_time.linears.3.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/bias']).fill_(0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.ff_layer.linear1.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/kernel']).permute(1,0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.ff_layer.linear1.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/bias'])
    opt_weight['opt.video_encoder.layer.'+str(i)+'.ff_layer.linear2.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/kernel']).permute(1,0)
    opt_weight['opt.video_encoder.layer.'+str(i)+'.ff_layer.linear2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/bias'])
    opt_weight['opt.video_encoder.layer.'+str(i)+'.layernorm2.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/scale'])
    opt_weight['opt.video_encoder.layer.'+str(i)+'.layernorm2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/bias'])
    opt_weight['opt.video_encoder.layer.'+str(i)+'.layernorm3.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/scale'])
    opt_weight['opt.video_encoder.layer.'+str(i)+'.layernorm3.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/bias'])
opt_weight['opt.video_encoder.last_layernorm.weight'] = trans(vit_weight['Transformer/encoder_norm/scale'])
opt_weight['opt.video_encoder.last_layernorm.bias'] = trans(vit_weight['Transformer/encoder_norm/bias'])


bert_weight = torch.load("./output/pretrianed_weights/bert-base-chinese.bin")

### word_embedding_weights:
opt_weight['opt.txt_embeddings.word_embeddings.weight'] = bert_weight['bert.embeddings.word_embeddings.weight']
### position_embedding weights:
opt_weight['opt.txt_embeddings.position_embeddings.weight'] = bert_weight['bert.embeddings.position_embeddings.weight']

opt_weight['opt.txt_embeddings.layernorm.weight'] = bert_weight['bert.embeddings.LayerNorm.gamma']
opt_weight['opt.txt_embeddings.layernorm.bias']  = bert_weight['bert.embeddings.LayerNorm.beta']

for  i in range(12):
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.0.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.query.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.0.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.query.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.1.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.key.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.1.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.key.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.2.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.value.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.2.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.value.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.3.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.dense.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.3.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.dense.bias'] 
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.ff_layer.linear1.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.intermediate.dense.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.ff_layer.linear1.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.intermediate.dense.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.ff_layer.linear2.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.dense.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.ff_layer.linear2.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.dense.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.layernorm1.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.gamma']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.layernorm1.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.beta']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.layernorm2.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.LayerNorm.gamma']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.layernorm2.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.output.LayerNorm.beta']
    



opt_weight['cls.dense.weight']  = bert_weight['cls.predictions.transform.dense.weight']
opt_weight['cls.dense.bias']  = bert_weight['cls.predictions.transform.dense.bias']
opt_weight['cls.layernorm.weight'] = bert_weight['cls.predictions.transform.LayerNorm.gamma' ]
opt_weight['cls.layernorm.bias'] =bert_weight['cls.predictions.transform.LayerNorm.beta']
opt_weight['cls.decoder.weight'] = bert_weight['cls.predictions.decoder.weight']
opt_weight['cls.decoder.bias'] = bert_weight['cls.predictions.bias']




if not os.path.exists('./output/pretrianed_weights'):
    os.makedirs('./output/pretrianed_weights')
torch.save(opt_weight,'./output/pretrianed_weights/double_modify_timesformerbase16_bertbasechinese_timezero.pt')





