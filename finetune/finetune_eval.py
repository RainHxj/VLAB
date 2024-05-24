"""

Licensed under the MIT license.

OPT finetuning for video-Text Retrieval
"""
import argparse
import os
from os.path import  join
from time import time
import torch
from tqdm import tqdm
from utils.distributed import all_gather_list, ddp_allgather
from cococaption.pycocoevalcap.eval import COCOEvalCap
from cococaption.pycocotools.coco import COCO
from utils.distributed import all_gather_list
import torch.distributed as dist
from utils.misc import NoOp
import math
import torch.nn.functional as F
import ipdb
import json


def decode_sequence(opts, seq):
    vocab = [line.strip() for line in open(opts.caption_cfg["toker"])]
    i2v = {i:vocab[i]  for i in range(len(vocab))}
    eos_token = opts.caption_cfg['eos_token']
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t].item()
            if ix == eos_token:
                break
            words.append(i2v[ix])
        sent = ' '.join(words)
        sents.append(sent.replace(' ##', ''))
    return sents

def compute_caption_metric(results, opts):

    coco = COCO(opts.data_cfg["ann_path"])
    cocoRes = coco.loadRes(results)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    metric = cocoEval.eval
    metric = {k: round(v*100,2)  for k,v in metric.items()}
    return metric

@torch.no_grad()
def evaluate_caption(model, eval_loader, opts, LOGGER):
    st = time()
    LOGGER.info("start running Video Caption evaluation ...")


    model.eval()
    results_2m = []
    # if dist.get_rank() == 0:
        # pbar = tqdm(total=len(eval_loader))
    # else:
        # pbar = NoOp()
    pbar = tqdm(total=len(eval_loader))

    for batch in eval_loader:

        ids_2m = batch['batch_2m']['ids']
        sample_num_2m = len(ids_2m)
        evaluation_dict = model(batch, compute_loss=False)

        if sample_num_2m > 0 :
            sents = evaluation_dict['generated_sequence_2m']
            if opts.multi_encoder["multi_type"] == "bert":
                sents = decode_sequence(opts, sents.data)
            elif opts.multi_encoder["multi_type"] == "deberta_v2":
                from transformers import DebertaV2Tokenizer

                tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
                res=[]
                for i in range(sents.data.shape[0]):
                    res.append(tokenizer.decode(sents.data[i]))
                sents=res
            
            for  i in range(sample_num_2m):
                result = {'video_id':ids_2m[i], 'caption': sents[i]}
                results_2m.append(result)

        pbar.update(1)
    

    pbar.close()

    all_results_2m = [i for results_2m in all_gather_list(results_2m)  for i in results_2m]

    model.train()
    if dist.get_rank() != 0:
        return {},{}

    
    metric_2m = compute_caption_metric(all_results_2m, opts)   
    eval_log = {'metric_2m': metric_2m }
    tot_time = time()-st

    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds")
    return eval_log, all_results_2m



def decode_vqa_sequence(i2v, seq, eos_token):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t].item()
            if ix == eos_token:
                break
            words.append(i2v[ix])
        sent = ' '.join(words)
        sents.append(sent.replace(' ##', ''))
    return sents

@torch.no_grad()
def evaluate_vqa(model, eval_loader, opts, global_step, LOGGER):


    LOGGER.info("start running open-ended vqa evaluation ...")
    if opts.vqa_type == "rank":
        answer_candidate = json.load(open(opts.data_cfg["ann_path"]))
        answer_candidate = {v:k for k,v in answer_candidate.items()}

    vocab = [line.strip() for line in open(opts.caption_cfg["toker"])]
    i2v = {i:vocab[i]  for i in range(len(vocab))}
    eos_token = opts.caption_cfg['eos_token']

    model.eval()

    if dist.get_rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()

    results_2m=[]
    predicted_answers_2m=[]
    groundtruth_answers=[]
    for batch in eval_loader:

        ids_2m = batch['batch_2m']['ids']

        sample_num_2m = len(ids_2m)
        evaluation_dict = model(batch, compute_loss=False)
        if opts.vqa_type == "rank":
            indexes = evaluation_dict['logits_2m'].max(dim=1)[1].cpu().numpy().tolist()
            answers = [answer_candidate[i] for i in indexes]
        elif opts.vqa_type == "generate":
            # ipdb.set_trace()
            answers = evaluation_dict["generated_sequence_2m"]
            answers = decode_vqa_sequence(i2v, answers.data, eos_token)

        else:
            raise NotImplementedError

        predicted_answers_2m += answers
        groundtruth_answers += batch['batch_2m']['answers']

        ## save answer
        # for  i in range(sample_num_2m):
        #     result = {'video_id':ids_2m[i], 'question': batch['batch_2m']['questions'][i],'answer': answers[i], "gt":batch['batch_2m']['answers'][i]}
        #     results_2m.append(result)

        if opts.data_cfg["name"] in ["vqav2", "msrvtt", "msvd"] and opts.submit_file:
            for  i in range(sample_num_2m):
                if opts.data_cfg["name"] in ["vqav2"]:
                    result = {'question_id': batch['batch_2m']['question_ids'][i],'answer': answers[i]}
                else:
                    result = {'question_id': batch['batch_2m']['ids'][i] + batch['batch_2m']['questions'][i],'answer': answers[i]}
                results_2m.append(result)            

        pbar.update(1)
    pbar.close()



    all_results_2m = [i for results_2m in all_gather_list(results_2m)  for i in results_2m]
    if opts.data_cfg["name"] in ["vqav2", "msrvtt", "msvd"] and opts.submit_file:
        if dist.get_rank() != 0:
            return {},{}
        else:
            result_folder = join(opts.output_dir, 'results_test') 
            json.dump(all_results_2m,open(join(result_folder, 'step_{}_2m.json'.format(global_step)), 'w'))    

            return {},{}        

    all_predicted_answers_2m = [ans for ans_ls in all_gather_list(predicted_answers_2m)  for ans in ans_ls]

    all_groundtruth_answers = [ans for ans_ls in all_gather_list(groundtruth_answers)  for ans in ans_ls]
    
    model.train()
    if dist.get_rank() != 0:
        return {},{}
    
    total_num = len(all_predicted_answers_2m)
    assert len(all_groundtruth_answers) == total_num
    
    LOGGER.info('total {} questions has been tested'.format(total_num))
    accurate_num_2m = sum([all_predicted_answers_2m[i] == all_groundtruth_answers[i] for i in range(total_num)])
    accuracy_2m = accurate_num_2m / total_num
 
    return {'metric_2m':{'accuracy':round(accuracy_2m*100,2)}}, all_results_2m



def compute_retrieval_metric(score_matrix, ids, ids_txt):
    #ipdb.set_trace()
    assert score_matrix.shape == (len(ids_txt),len(ids))
    # video retrieval
    indice_matrix_1 = score_matrix.sort(dim=-1,descending=True)[1].tolist()
    rank = []
    for i in range(len(ids_txt)):
        gt_indice = ids.index(ids_txt[i])
        rank.append(indice_matrix_1[i].index(gt_indice))

        ## for vis

        # if indice_matrix_1[i].index(gt_indice) == 0:
        #     print([ids[vid] for vid in indice_matrix_1[i][:3]])
    
    rank = torch.tensor(rank).to(score_matrix)
    
    vr_r1 = (rank < 1).sum().item() / len(ids_txt)
    vr_r5 = (rank < 5).sum().item() / len(ids_txt)
    vr_r10 = (rank < 10).sum().item() / len(ids_txt)
    v_medianR = torch.median(rank).item() +1



    # text retrieval

    # indice_matrix_2 = score_matrix.sort(dim=0,descending=True)[1].permute(1,0).tolist()
    # rank = []
    # for i in range(len(ids)):
    #     gt_indices=[]
    #     for idx, id in enumerate(ids_txt):
    #         if id == ids[i]:
    #             gt_indices.append(idx)

    #     rank.append(min([indice_matrix_2[i].index(idx) for idx in gt_indices]))
    #     if min([indice_matrix_2[i].index(idx) for idx in gt_indices]) == 0:
    #         print([ids_txt[vid] for vid in indice_matrix_2[i][:3]])
    
    # rank = torch.tensor(rank).to(score_matrix)
    
    # tr_r1 = (rank < 1).sum().item() / len(ids)
    # tr_r5 = (rank < 5).sum().item() / len(ids)
    # tr_r10 = (rank < 10).sum().item() / len(ids)
    # t_medianR = torch.median(rank).item() +1

    eval_log = {'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                'forward_rsum': round((vr_r1 + vr_r5 + vr_r10)*100,1), 
                'forward_medianR': v_medianR,
                'forward_r1': f'{round(vr_r1*100,1)}'}
                # 'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                # 'backward_rsum': round((tr_r1 + tr_r5 + tr_r10)*100,1), 
                # 'backward_medianR': t_medianR}
    return eval_log


@torch.no_grad()
def evaluate_retrieval(model, val_loader, LOGGER):
    model.eval()
    LOGGER.info("start running contra validation...")

    txt_feature = []
    video_feature = []
    
    txt_feature2 = []
    video_feature2 = []
    
    ids = []
    ids_txt = []
    
    for batch in tqdm(val_loader):
        evaluation_dict = model(batch, compute_loss=False)
        feat_t = evaluation_dict['feat_t']
        feat_v = evaluation_dict['feat_v']
        if isinstance(feat_t, list):
            txt_feature.append(feat_t[0])
            video_feature.append(feat_v[0])    
            txt_feature2.append(feat_t[1])
            video_feature2.append(feat_v[1])     
        else:
            txt_feature.append(feat_t)
            video_feature.append(feat_v)
        
        ids += batch['batch_2m']['ids']
        ids_txt += batch['batch_2m']['ids_txt']
        

    txt_feature = torch.cat(txt_feature, dim = 0)
    video_feature = torch.cat(video_feature, dim = 0)
    
    if len(txt_feature2)!=0: 
        txt_feature2 = torch.cat(txt_feature2, dim = 0)
        video_feature2 = torch.cat(video_feature2, dim = 0)
        
        txt_feature2 = ddp_allgather(txt_feature2)
        video_feature2 = ddp_allgather(video_feature2)


    txt_feature = ddp_allgather(txt_feature)
    video_feature = ddp_allgather(video_feature)
    ids = [j for i in all_gather_list(ids) for j in i]
    ids_txt = [j for i in all_gather_list(ids_txt) for j in i]
    
    use_dual_softmax=True
    if dist.get_rank()==0:
        # temp = model.module.contra_temp builded_model
        # temp = 1./ model.module.opt.vision_text_model.logit_scale.exp()
        # temp = 1./ model.module.opt.builded_model.logit_scale.exp()
        score_matrix_t_v = torch.matmul(txt_feature, video_feature.permute(1,0))

        # else:
        if not isinstance(txt_feature2,list):
            score_matrix_t_v2 = F.softmax(torch.matmul(txt_feature2, video_feature2.permute(1,0)),dim=1)
            score_matrix_t_v = F.softmax(score_matrix_t_v,dim=1)+score_matrix_t_v2

        score_matrix_c1 = score_matrix_t_v.clone()
        # dual = F.softmax(score_matrix_c1/temp, dim=0)*len(score_matrix_c1)
        norm_matrix = score_matrix_c1
        # dual_matrix = torch.mul(dual, score_matrix_c1)
        eval_log = compute_retrieval_metric(norm_matrix, ids, ids_txt)
        eval_log_t_v = {k.replace('forward','video').replace('backward','txt') : v for k,v in eval_log.items()}

        eval_log_t_v_dual={}
        if use_dual_softmax:
            score_matrix_t_v = torch.matmul(txt_feature, video_feature.permute(1,0))
            try:
                temp1 = 1./ model.module.opt.vision_text_model.models[0].logit_scale.exp()
            except:
                temp1 = 1./ model.module.opt.vision_text_model.models[0].text.logit_scale.exp()
            if not isinstance(txt_feature2,list):
                temp2 = 1./ model.module.opt.vision_text_model.models[1].logit_scale.exp()

                score_matrix_t_v2 = F.softmax(torch.matmul(txt_feature2, video_feature2.permute(1,0)/temp2),dim=0)*len(score_matrix_t_v)
                score_matrix_t_v2_dual = torch.matmul(txt_feature2, video_feature2.permute(1,0))#F.softmax(score_matrix_t_v2/temp2,dim=1)
                dual_v2 = torch.mul(score_matrix_t_v2_dual, score_matrix_t_v2)


                score_matrix_t_v1 = F.softmax(score_matrix_t_v/temp1,dim=0)*len(score_matrix_t_v)
                score_matrix_t_v1_dual = score_matrix_t_v#F.softmax(score_matrix_t_v1/temp1,dim=1)
                dual_v1 = torch.mul(score_matrix_t_v1_dual, score_matrix_t_v1)
                
                dual_matrix=dual_v1+dual_v2
                # score_matrix_t_v = F.softmax(score_matrix_t_v/temp1,dim=0)+score_matrix_t_v2

                # score_matrix_c1 = score_matrix_t_v.clone()
                # dual = F.softmax(score_matrix_c1/temp, dim=0)*len(score_matrix_c1)
                # # dual_matrix = score_matrix_c1
                # dual_matrix = torch.mul(dual, score_matrix_c1)
                eval_log = compute_retrieval_metric(dual_matrix, ids, ids_txt)
                eval_log_t_v_dual = {k.replace('forward','dual_video').replace('backward','dual_text') : v for k,v in eval_log.items()}
            else:
                score_matrix_t_v1 = F.softmax(score_matrix_t_v/temp1,dim=0)*len(score_matrix_t_v)
                score_matrix_t_v1_dual = score_matrix_t_v#F.softmax(score_matrix_t_v1/temp1,dim=1)
                dual_v1 = torch.mul(score_matrix_t_v1_dual, score_matrix_t_v1)
                
                dual_matrix=dual_v1
                # score_matrix_t_v = F.softmax(score_matrix_t_v/temp1,dim=0)+score_matrix_t_v2

                # score_matrix_c1 = score_matrix_t_v.clone()
                # dual = F.softmax(score_matrix_c1/temp, dim=0)*len(score_matrix_c1)
                # # dual_matrix = score_matrix_c1
                # dual_matrix = torch.mul(dual, score_matrix_c1)
                eval_log = compute_retrieval_metric(dual_matrix, ids, ids_txt)
                eval_log_t_v_dual = {k.replace('forward','dual_video').replace('backward','dual_text') : v for k,v in eval_log.items()}
            for k, v in eval_log_t_v_dual.items():
                eval_log_t_v[k]=v


        eval_log = {'t_v': eval_log_t_v}
      
    else:
        eval_log = None
    model.train()
    return eval_log







# @torch.no_grad()
# def evaluate_retrieval(model, val_loader, LOGGER):
#     model.eval()
#     LOGGER.info("start running contra validation...")

#     txt_feature = []
#     video_feature = []
#     ids = []
#     ids_txt = []
    
#     for batch in tqdm(val_loader):
#         evaluation_dict = model(batch, compute_loss=False)
#         feat_t = evaluation_dict['feat_t']
#         feat_v = evaluation_dict['feat_v']
#         txt_feature.append(feat_t)
#         video_feature.append(feat_v)
        
#         ids += batch['batch_2m']['ids']
#         ids_txt += batch['batch_2m']['ids_txt']
        

#     txt_feature = torch.cat(txt_feature, dim = 0)
#     video_feature = torch.cat(video_feature, dim = 0)
    

#     txt_feature = ddp_allgather(txt_feature)
#     video_feature = ddp_allgather(video_feature)
#     ids = [j for i in all_gather_list(ids) for j in i]
#     ids_txt = [j for i in all_gather_list(ids_txt) for j in i]
    

#     if dist.get_rank()==0:
#         # temp = model.module.contra_temp builded_model
#         # temp = 1./ model.module.opt.vision_text_model.logit_scale.exp()
#         # temp = 1./ model.module.opt.builded_model.logit_scale.exp()
#         score_matrix_t_v = torch.matmul(txt_feature, video_feature.permute(1,0))
#         score_matrix_c1 = score_matrix_t_v.clone()
#         # dual = F.softmax(score_matrix_c1/temp, dim=0)*len(score_matrix_c1)
#         dual_matrix = score_matrix_c1
#         # dual_matrix = torch.mul(dual, score_matrix_c1)
#         eval_log = compute_retrieval_metric(dual_matrix, ids, ids_txt)
#         eval_log_t_v = {k.replace('forward','video').replace('backward','txt') : v for k,v in eval_log.items()}
#         eval_log = {'t_v': eval_log_t_v}
      
#     else:
#         eval_log = None
#     model.train()
#     return eval_log