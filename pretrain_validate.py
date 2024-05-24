from builtins import ValueError, isinstance
import json
import math
import os
from os.path import exists, join
from time import time, sleep
import ipdb
import torch
from torch.nn import functional as F
import torchvision
import shutil

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import ddp_allgather, all_gather_list
                            

import numpy as np
from PIL import Image
from torchvision.transforms import *
# import torch_fidelity
# import clip
# import misc.utils as utils
# from utils.aic_evaler import AICEvaler
# from utils.coco_evaler import COCOEvaler

def validate(model, val_dataloaders, opts, global_step):
    #ipdb.set_trace()
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        assert 'Two' in task or 'Three' in task
        if 'Two' in task: 
            val_log = validate_2m(model, loader, task.split('--')[0], opts, global_step)
        else:
            val_log = validate_3m(model, loader, task.split('--')[0], opts, global_step)
        val_log = {f'valid_{task}/{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(val_log)
    model.train()

@torch.no_grad()
def validate_2m(model, val_loader, task,opts, global_step):
    LOGGER.info("start running {} validation...".format(task))
    n_correct = 0
    n_word = 0
    n_correct_caption = 0
    n_word_caption = 0
    tot_score = 0
    n_ex = 0
    txt_feature = []
    video_feature = []
    video_generation_loss_valid = []
    val_log = {}
    if  'videogenTwo' in task.split('_'):
        generated_all_dir = os.path.join(opts.output_dir, f'gen_videos_all/step{global_step}')
        generated_ranked_dir = os.path.join(opts.output_dir, f'gen_videos_ranked/step{global_step}')
        gt_dir_for_FID = os.path.join(opts.output_dir,'gt_dir_for_FID')
        videogen_batches = model.videogen_batches
        for dir in (generated_all_dir, generated_ranked_dir, gt_dir_for_FID):
            if hvd.rank()==1:            
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                os.makedirs(dir,exist_ok=True)


        txt_mapper = val_loader.dataset.txt_mapper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        preprocess.transforms = [preprocess.transforms[0],preprocess.transforms[1],preprocess.transforms[-1]] 

        clip_scores = []

    for i, batch in enumerate(val_loader):
        evaluation_dict= model(batch, task=task, compute_loss=False, batch_idx = i)

        if 'contraTwo' in task.split('_'):
            feat_t = evaluation_dict['feat_t'] 
            feat_v = evaluation_dict['feat_v'] 
            txt_feature.append(feat_t)
            video_feature.append(feat_v)

        if 'mlmTwo' in task.split('_'):
            prediction_scores  = evaluation_dict['prediction_scores'] 
            txt_labels = evaluation_dict['txt_labels'] 
            txt_labels = txt_labels[txt_labels != -1]
            n_correct += (prediction_scores.max(dim=-1)[1] == txt_labels).sum().item()
            n_word += txt_labels.numel()

        if 'unimlmTwo' in task.split('_'):
            prediction_scores_caption = evaluation_dict['prediction_scores_caption'] 
            txt_labels_caption = evaluation_dict['txt_labels_caption'] 
            txt_labels_caption = txt_labels_caption[txt_labels_caption != -1]
            n_correct_caption += (prediction_scores_caption.max(dim=-1)[1] == txt_labels_caption).sum().item()
            n_word_caption += txt_labels_caption.numel()
        
        if 'matchTwo' in task.split('_'):
            vtm_scores =  evaluation_dict['vtm_scores'] 
            ground_truth = evaluation_dict['ground_truth'] 
            predictions = vtm_scores.max(dim = 1 )[1]
            tot_score += (predictions.cpu().numpy() == ground_truth.cpu().numpy()).sum()
            n_ex += len(ground_truth)
        
        if  'videogenTwo' in task.split('_') :
            video_generation_loss_valid.append(evaluation_dict['validation_loss'].item())
            if i< videogen_batches :  ### only generate the first ... batches 

                gts = evaluation_dict['gt']
                ids = batch['batch_2m']['ids']

                generated_videos = evaluation_dict['generated_videos']  ### b*ranking_size, 3, h, nw
                ranking_size = generated_videos.shape[0] // len(ids)
                ##### compute clip scores for ranking and CLIP metric                     
                txt_tokens = batch['batch_2m']['txt_tokens']
                txts = [ ' '.join(txt_mapper.detokenize(i[i!=0].tolist()[1:-1])).replace(' ##','') for i in txt_tokens ]
                txt_tokens = torch.cat([ clip.tokenize(i)  for i in txts ],dim=0).cuda()
                #####  expand txt_tokens
                b,dim = txt_tokens.shape
                txt_tokens = txt_tokens.unsqueeze(1).expand(-1,ranking_size,-1).reshape(-1,dim) ### b*ranking,size 512
                image = preprocess(generated_videos) ### b*ranking_size, 3, h, nw
                image_features = clip_model.encode_image(image)
                text_features = clip_model.encode_text(txt_tokens)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                score = image_features.unsqueeze(1) @ text_features.unsqueeze(-1)
                score = score.squeeze().reshape(b,ranking_size)
                score_max, score_max_idx = score.max(dim=1)
                clip_scores += score_max.tolist()
                score_max_idx = score_max_idx.tolist()
                

                for i in range(len(ids)):
                    for j in range(ranking_size):
                        generated_video = torchvision.transforms.ToPILImage()(generated_videos[i*ranking_size+j])
                        generated_video.save(os.path.join(generated_all_dir,f'{txts[i]}_sample{j}_clipsim_{score[i][j]:.3f}.jpg'))
                        if j == score_max_idx[i]:
                            generated_video.save(os.path.join(generated_ranked_dir,f'{txts[i]}_sample{j}_clipsim_{score[i][j]:.3f}.jpg'))
                    gt = torchvision.transforms.ToPILImage()(gts[i])
                    gt.save(os.path.join(gt_dir_for_FID,f'{txts[i]}.jpg'))
                    # if i>10:
                    #     break



    if 'mlmTwo' in task.split('_'):
        n_correct = sum(all_gather_list(n_correct))
        n_word = sum(all_gather_list(n_word))
        mlm_acc = n_correct / n_word
        val_log['mlm_acc'] = round(mlm_acc,2)
    if 'unimlmTwo' in task.split('_'):
        n_correct_caption = sum(all_gather_list(n_correct_caption))
        n_word_caption = sum(all_gather_list(n_word_caption))
        unimlm_acc = n_correct_caption / n_word_caption     
        val_log['unimlm_acc'] = round(unimlm_acc,2)
    if 'matchTwo' in task.split('_'):
        tot_score = sum(all_gather_list(tot_score))
        n_ex = sum(all_gather_list(n_ex))
        match_acc = tot_score / n_ex
        val_log['match_acc'] = match_acc
    if 'contraTwo' in task.split('_'):
        txt_feature = torch.cat(txt_feature, dim = 0)
        video_feature = torch.cat(video_feature, dim = 0)
        all_txt_feature = ddp_allgather(txt_feature)
        all_video_feature = ddp_allgather(video_feature)
        score_matrix_tv = torch.matmul(all_txt_feature, all_video_feature.permute(1,0))
        t2v_r1, v2t_r1 = compute_r1(score_matrix_tv)
        val_log['t2v_r1'] = round(t2v_r1*100,1)
        #val_log['v2t_r1'] = v2t_r1*100
    if  'videogenTwo' in task.split('_') :
        video_generation_loss_valid = np.mean(np.array(all_gather_list(video_generation_loss_valid)))
        val_log['video_generation_loss_valid'] = video_generation_loss_valid
        

        if videogen_batches > 0: 
            clip_scores = [ j  for i in all_gather_list(clip_scores) for j in i]
            clip_scores = np.mean(np.array(clip_scores))
            val_log['CLIP_score'] = round(clip_scores,3)

            if hvd.rank() == 1:
                #### compute FID, IS score
                metric = torch_fidelity.calculate_metrics(input1=generated_ranked_dir, input2=gt_dir_for_FID, 
                                                        cuda=True,isc=True,fid=True,kid=False,verbose=False)
                val_log['IS'] = round(metric['inception_score_mean'],3)
                val_log['FID'] = round(metric['frechet_inception_distance'],3)

    LOGGER.info(val_log)

    return val_log


@torch.no_grad()
def validate_3m(model, val_loader, task, opts, global_step):
    LOGGER.info("start running {} validation...".format(task))
    n_correct = 0
    n_correct_woaudio = 0
    n_word = 0
    n_word_woaudio = 0
    n_correct_caption = 0
    n_word_caption = 0
    n_correct_caption_video = 0
    n_word_caption_video = 0
    n_correct_caption_audio = 0
    n_word_caption_audio = 0
    txt_feature = []
    video_feature = []
    va_feature = []
    audio_feature=[]
    ### mvm
    mvm_raw_pixels_regression_loss = []
    mvm_feat_regression_loss = []
    n_correct_patches = 0
    n_patches = 0
    ###
    tot_score_three = 0
    tot_score_two = 0
    n_ex = 0
    video_generation_loss = []
    audio_generation_loss = []
    visual_vqvae_loss = []
    val_log = {}

    # if  'videogenThree' in task.split('_'):
    #     generated_all_dir = os.path.join(opts.output_dir, f'gen_videos_all/step{global_step}')
    #     generated_ranked_dir = os.path.join(opts.output_dir, f'gen_videos_ranked/step{global_step}')
    #     gt_dir_for_FID = os.path.join(opts.output_dir,'gt_dir_for_FID')
        
    #     for dir in (generated_all_dir, generated_ranked_dir, gt_dir_for_FID):
    #         if hvd.rank()==1:            
    #             if os.path.exists(dir):
    #                 shutil.rmtree(dir)
    #             os.makedirs(dir,exist_ok=True)


    #     txt_mapper = val_loader.dataset.txt_mapper
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     clip_model, preprocess = clip.load("ViT-B/32", device=device)
    #     clip_model.eval()
    #     preprocess.transforms = [preprocess.transforms[0],preprocess.transforms[1],preprocess.transforms[-1]] 

    #     clip_scores = []
    #     videogen_batches = model.videogen_batches 

    for i, batch in enumerate(val_loader):
        evaluation_dict= model(batch, task=task, compute_loss=False, batch_idx = i)

        if 'contraThree' in task.split('_'):
            feat_t = evaluation_dict['feat_t'] 
            feat_v = evaluation_dict['feat_v'] 
            feat_va = evaluation_dict['feat_va'] 
            feat_a = evaluation_dict['feat_a']
            txt_feature.append(feat_t)
            video_feature.append(feat_v)
            va_feature.append(feat_va)
            audio_feature.append(feat_a)
        if 'mlmThree' in task.split('_'):
            prediction_scores  = evaluation_dict['prediction_scores'] 
            txt_labels = evaluation_dict['txt_labels'] 
            prediction_scores_woaudio  = evaluation_dict['prediction_scores_woaudio']
            txt_labels_woaudio = evaluation_dict['txt_labels_woaudio'] 
            txt_labels = txt_labels[txt_labels != -1]
            txt_labels_woaudio = txt_labels_woaudio[txt_labels_woaudio != -1]
            n_correct += (prediction_scores.max(dim=-1)[1] == txt_labels).sum().item()
            n_correct_woaudio += (prediction_scores_woaudio.max(dim=-1)[1] == txt_labels_woaudio).sum().item()
            n_word += txt_labels.numel()
            n_word_woaudio += txt_labels_woaudio.numel()
        if 'unimlmThree' in task.split('_'):
            prediction_scores_caption = evaluation_dict['prediction_scores_caption'] 
            txt_labels_caption = evaluation_dict['txt_labels_caption'] 
            prediction_scores_caption_video = evaluation_dict['prediction_scores_caption_video']
            txt_labels_caption_video = evaluation_dict['txt_labels_caption_video']
            prediction_scores_caption_audio = evaluation_dict['prediction_scores_caption_audio']
            txt_labels_caption_audio = evaluation_dict['txt_labels_caption_audio']
            txt_labels_caption = txt_labels_caption[txt_labels_caption != -1]
            n_correct_caption += (prediction_scores_caption.max(dim=-1)[1] == txt_labels_caption).sum().item()
            n_word_caption += txt_labels_caption.numel()

            txt_labels_caption_video = txt_labels_caption_video[txt_labels_caption_video != -1]
            n_correct_caption_video += (prediction_scores_caption_video.max(dim=-1)[1] == txt_labels_caption_video).sum().item()
            n_word_caption_video += txt_labels_caption_video.numel()

            txt_labels_caption_audio = txt_labels_caption_audio[txt_labels_caption_audio != -1]
            n_correct_caption_audio += (prediction_scores_caption_audio.max(dim=-1)[1] == txt_labels_caption_audio).sum().item()
            n_word_caption_audio += txt_labels_caption_audio.numel()


        if  'videogenThree' in task.split('_') :
            video_generation_loss.append(evaluation_dict['video_generation_loss'].item())
            audio_generation_loss.append(evaluation_dict['audio_generation_loss'].item())
            # if i< videogen_batches :  ### only generate the first ... batches 

            #     gts = evaluation_dict['gt']
            #     ids = batch['batch_3m']['ids']

            #     generated_videos = evaluation_dict['generated_videos']  ### b*ranking_size, 3, h, nw
            #     ranking_size = generated_videos.shape[0] // len(ids)
            #     ##### compute clip scores for ranking and CLIP metric                     
            #     txt_tokens = batch['batch_3m']['txt_tokens']
            #     txts = [ ' '.join(txt_mapper.detokenize(i[i!=0].tolist()[1:-1])).replace(' ##','') for i in txt_tokens ]
            #     txt_tokens = torch.cat([ clip.tokenize(i)  for i in txts ],dim=0).cuda()
            #     #####  expand txt_tokens
            #     b,dim = txt_tokens.shape
            #     txt_tokens = txt_tokens.unsqueeze(1).expand(-1,ranking_size,-1).reshape(-1,dim) ### b*ranking,size 512
            #     image = preprocess(generated_videos) ### b*ranking_size, 3, h, nw
            #     image_features = clip_model.encode_image(image)
            #     text_features = clip_model.encode_text(txt_tokens)
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     score = image_features.unsqueeze(1) @ text_features.unsqueeze(-1)
            #     score = score.squeeze().reshape(b,ranking_size)
            #     score_max, score_max_idx = score.max(dim=1)
            #     clip_scores += score_max.tolist()
            #     score_max_idx = score_max_idx.tolist()
                
            #     for i in range(len(ids)):
            #         for j in range(ranking_size):
            #             generated_video = torchvision.transforms.ToPILImage()(generated_videos[i*ranking_size+j])
            #             generated_video.save(os.path.join(generated_all_dir,f'{txts[i]}_sample{j}_clipsim_{score[i][j]:.3f}.jpg'))
            #             if j == score_max_idx[i]:
            #                 generated_video.save(os.path.join(generated_ranked_dir,f'{txts[i]}_sample{j}_clipsim_{score[i][j]:.3f}.jpg'))
            #         gt = torchvision.transforms.ToPILImage()(gts[i])
            #         gt.save(os.path.join(gt_dir_for_FID,f'{txts[i]}.jpg'))
            #         # if i>10:
            #         #     break


    if 'mlmThree' in task.split('_'):
        n_correct = sum(all_gather_list(n_correct))
        n_correct_woaudio = sum(all_gather_list(n_correct_woaudio))
        n_word = sum(all_gather_list(n_word))
        n_word_woaudio = sum(all_gather_list(n_word_woaudio))
        mlm_acc = n_correct / n_word
        mlm_acc_woaudio = n_correct_woaudio / n_word_woaudio
        val_log['mlm_acc'] = round(mlm_acc,2)
        val_log['mlm_acc_woaudio'] = round(mlm_acc_woaudio,2)
        
    if 'unimlmThree' in task.split('_'):
        n_correct_caption = sum(all_gather_list(n_correct_caption))
        n_word_caption = sum(all_gather_list(n_word_caption))
        unimlm_acc = n_correct_caption / n_word_caption     

        n_correct_caption_video = sum(all_gather_list(n_correct_caption_video))
        n_word_caption_video = sum(all_gather_list(n_word_caption_video))
        unimlm_acc_video = n_correct_caption_video / n_word_caption_video   

        n_correct_caption_audio = sum(all_gather_list(n_correct_caption_audio))
        n_word_caption_audio = sum(all_gather_list(n_word_caption_audio))
        unimlm_acc_audio = n_correct_caption_audio / n_word_caption_audio      

        val_log['unimlm_acc'] = round(unimlm_acc,2)
        val_log['unimlm_acc_video'] = round(unimlm_acc_video,2)
        val_log['unimlm_acc_audio'] = round(unimlm_acc_audio,2)

    if 'contraThree' in task.split('_'):
        txt_feature = torch.cat(txt_feature, dim = 0)
        video_feature = torch.cat(video_feature, dim = 0)
        va_feature = torch.cat(va_feature, dim = 0)
        audio_feature = torch.cat(audio_feature, dim = 0)
        all_txt_feature = ddp_allgather(txt_feature)
        all_video_feature = ddp_allgather(video_feature)
        all_va_feature = ddp_allgather(va_feature)
        all_audio_feature = ddp_allgather(audio_feature)
        score_matrix_tv = torch.matmul(all_txt_feature, all_video_feature.permute(1,0))
        score_matrix_t_va = torch.matmul(all_txt_feature, all_va_feature.permute(1,0))
        score_matrix_ta = torch.matmul(all_txt_feature, all_audio_feature.permute(1,0))
        score_matrix_va = torch.matmul(all_video_feature, all_audio_feature.permute(1,0))
        t2v_r1, v2t_r1 = compute_r1(score_matrix_tv)
        t2va_r1, va2t_r1 = compute_r1(score_matrix_t_va)
        t2a_r1, a2t_r1 = compute_r1(score_matrix_ta)
        v2a_r1, a2v_r1 = compute_r1(score_matrix_va)
        val_log['t2v_r1'] = round(t2v_r1*100,1)
        val_log['t2va_r1'] = round(t2va_r1*100,1)
        val_log['t2a_r1'] = round(t2a_r1*100,1)
        val_log['v2a_r1'] = round(v2a_r1*100,1)
        #val_log['v2t_r1'] = v2t_r1*100

 
    if  'videogenThree' in task.split('_') :
        video_generation_loss = np.mean(np.array(all_gather_list(video_generation_loss)))
        audio_generation_loss = np.mean(np.array(all_gather_list(audio_generation_loss)))
        val_log['video_generation_loss'] = video_generation_loss
        val_log['audio_generation_loss'] = audio_generation_loss


        # if videogen_batches > 0: 
        #     clip_scores = [ j  for i in all_gather_list(clip_scores) for j in i]
        #     #print(100* len(clip_scores))
        #     clip_scores = np.mean(np.array(clip_scores))
        #     val_log['CLIP_score'] = round(clip_scores,3)

        #     if hvd.rank() == 1:
        #         #### compute FID, IS score
        #         metric = torch_fidelity.calculate_metrics(input1=generated_ranked_dir, input2=gt_dir_for_FID, 
        #                                                 cuda=True,isc=True,fid=True,kid=False,verbose=False)
        #         val_log['IS'] = round(metric['inception_score_mean'],3)
        #         val_log['FID'] = round(metric['frechet_inception_distance'],3)

    LOGGER.info(val_log)

    return val_log

def compute_r1(score_matrix):
    # video retrieval

    size = len(score_matrix)
    _, rank_txt = score_matrix.topk(size, dim=1)
    gt_video = torch.arange(size).long().to(rank_txt.device).unsqueeze(1).expand_as(rank_txt)
    rank = (rank_txt == gt_video).nonzero()[:,1]
    vr_r1 = (rank < 1).sum().item() / size
    vr_r5 = (rank < 5).sum().item() / size
    vr_r10 = (rank < 10).sum().item() / size
    v_medianR = torch.median(rank) +1

    # text retrieval
 
    _, rank_video = score_matrix.topk(size, dim=0)
    gt_video = torch.arange(size).long().to(rank_txt.device).unsqueeze(0).expand_as(rank_video)
    rank = (rank_video == gt_video).nonzero()[:,0]  
    tr_r1 = (rank < 1).sum().item() / size
    tr_r5 = (rank < 5).sum().item() / size
    tr_r10 = (rank < 10).sum().item() / size
    t_medianR = torch.median(rank) +1

    return vr_r1, tr_r1
