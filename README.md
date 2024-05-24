# VLAB

## Get Started

### Model Weights

1. Necessary Weights

Unzip the weights under ```./output/pretrained_weights``` folder.

2. For Finetuning

Unzip the corresponding weights under ```./output/pretrain``` folder.

3. For Testing

Put the corresponding weights under ```./output/downstreams/{large, giant}``` folder.

### Run Scripts

```bash
# model_size: large, giant
# task: didemo_retrieval, msrvtt_caption, msrvtt_vqa, msrvtt_retrieval, msvd_caption, msvd_vqa, msvd_retrieval

# finetune
sh scripts/finetune/finetune.sh ${model_size} ${task}

# test
sh scripts/test/test.sh ${model_size} ${task}
```

