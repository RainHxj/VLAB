import json
from toolz.sandbox import unzip
import torch
from torch.utils.data import Dataset

from torchvision import transforms
import random
from os.path import join 
from utils.logger import LOGGER
import ipdb
from data import VideoMapper, AudioMapper



class VideoRecognitionDataset(Dataset):
    def __init__(self, ids_path, video_mapper, audio_mapper, label_json, split_id = True ):
        assert isinstance(video_mapper, VideoMapper)
        assert isinstance(audio_mapper, AudioMapper)
        self.video_mapper = video_mapper
        self.audio_mapper = audio_mapper
        self.ids = json.load(open(ids_path)) 
        self.idx = list(range(len(self.ids)))
        self.dataset_name = self.video_mapper.datatype.split('_')[-1]
        self.labels = json.load(open(label_json))
        
    def __len__(self):
        return len(self.ids)

    
    def __getitem__(self, i):
        id_ = self.ids[i]
        video_pixels = self.video_mapper[id_]
        audio_spectrograms = self.audio_mapper[id_]
        if video_pixels is None: ###wrong img/video and needs to resample 
            resample_idx = random.choice(self.idx)
            LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
            return self.__getitem__(resample_idx)
        if audio_spectrograms is None: ### wrong audio and needs to resample
            resample_idx = random.choice(self.idx)
            LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong audio, use {resample_idx} instead.')
            return self.__getitem__(resample_idx)

        label = torch.tensor(self.labels[id_]).long()

        return id_, video_pixels, audio_spectrograms, label


def videorecognition_collate(inputs):
    (ids , video_pixels, audio_spectrograms, labels) = map(list, unzip(inputs))
    ids_2m = []
    video_pixels_2m = []
    labels_2m = []

    ids_3m = []
    video_pixels_3m = []
    audio_spectrograms_3m = []
    labels_3m = []


    for i in range(len(audio_spectrograms)):
        if audio_spectrograms[i] != 'woaudio':
            ids_3m.append(ids[i])
            video_pixels_3m.append(video_pixels[i])
            audio_spectrograms_3m.append(audio_spectrograms[i])
            labels_3m.append(labels[i])
        else:
            ids_2m.append(ids[i])
            video_pixels_2m.append(video_pixels[i])
            labels_2m.append(labels[i])


    if ids_3m != []:
        video_pixels_3m = torch.stack(video_pixels_3m, dim=0)
        audio_spectrograms_3m = torch.stack(audio_spectrograms_3m, dim=0)
    
    if ids_2m != []:
        video_pixels_2m = torch.stack(video_pixels_2m, dim=0)
    
    batch_2m =   {'ids': ids_2m,
             'video_pixels': video_pixels_2m,
             'labels':labels_2m}
    
    
    batch_3m =   {'ids': ids_3m,
             'video_pixels': video_pixels_3m,
             'audio_spectrograms': audio_spectrograms_3m,
             'labels':labels_3m}


    batch={'batch_2m':batch_2m,
            'batch_3m':batch_3m}
    
    return batch


