import os
import random
from .base import BaseDataset
import numpy as np
import sys
import pdb


def normalize(audio_data, re_factor=0.8):
    EPS = 1e-3
    min_data = audio_data.min()
    audio_data -= min_data
    max_data = audio_data.max()
    audio_data /= max_data + EPS
    audio_data -= 0.5
    audio_data *= 2
    audio_data *= re_factor

    return audio_data.clip(-1, 1)

def swap(input, order):
    len_order = len(order)
    if len_order < 2:
        return input
    elif len_order == 2:
        return input[order[0]], input[order[1]]
    elif len_order == 3:
        return input[order[0]], input[order[1]], input[order[2]]
    elif len_order == 4:
        return input[order[0]], input[order[1]], input[order[2]], input[order[3]]
    else:
        print("error")



class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]
        path_audioN = infos[0][0]
        music_category = path_audioN.split('/')[1]
        music_lib = []
        type_1st = 0
        type_2nd = 0
        if '-' in music_category:
            type_1st = 1
            #print('1st is a duet: ', music_category)
            music_a = music_category.split('-')[0]
            music_b = music_category.split('-')[1]
            music_lib.append(music_a)
            music_lib.append(music_b)
        else:
            #print('1st is a solo: ', music_category) 
            type_1st = 0
            music_lib.append(music_category)


        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            second_video_idx = random.randint(0, len(self.list_sample)-1)
            second_music_category = self.list_sample[second_video_idx][0].split('/')[1]
            # if this is duet
            if '-' in second_music_category:
                type_2st = 1
                #print('2nd is a duet: ', second_music_category)
                music_2a = second_music_category.split('-')[0]
                music_2b = second_music_category.split('-')[1]
                while second_music_category in music_lib \
                        or music_2a in music_lib \
                        or music_2b in music_lib:
                    music_2a = ''
                    music_2b = ''
                    second_video_idx = random.randint(0, len(self.list_sample)-1)
                    second_music_category = self.list_sample[second_video_idx][0].split('/')[1]
                    if '-' in second_music_category:
                        type_2st = 1
                        #print('2nd is a duet: ', second_music_category)
                        music_2a = second_music_category.split('-')[0]
                        music_2b = second_music_category.split('-')[1]
                    else:
                        type_2st = 0


            # if this is solo
            else:
                type_2st = 0
                music_2a = ''
                music_2b = ''
                while second_music_category in music_lib \
                        or music_2a in music_lib \
                        or music_2b in music_lib:
                    music_2a = ''
                    music_2b = ''
                    second_video_idx = random.randint(0, len(self.list_sample)-1)
                    second_music_category = self.list_sample[second_video_idx][0].split('/')[1]
                    if '-' in second_music_category:
                        type_2st = 1
                        music_2a = second_music_category.split('-')[0]
                        music_2b = second_music_category.split('-')[1]
                    else:
                        type_2st = 0


            if type_2st==0:
                music_lib.append(second_music_category)
            else:
                music_lib.append(music_2a)
                music_lib.append(music_2b)
            infos[n] = self.list_sample[second_video_idx]

        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # data path
            data_path = '../../dataset/MUSIC/'
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(data_path + path_frameN+'/{:06d}.jpg'.format(center_frameN + idx_offset))
            path_audios[n] = data_path + path_audioN
        
        
        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)
        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
        return ret_dict
