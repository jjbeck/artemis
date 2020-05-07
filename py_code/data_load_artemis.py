#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:49:24 2019

@author: alekhka
"""


import numpy as np
import pickle
import os
import torch
from collections import Counter
from random import shuffle

BEHAVIOR_INDICES = {
    "drink":0,
    "eat":1,
    "groom":2,
    "hang":3,
    "sniff":4,
    "rear":5,
    "rest":6,
    "walk":7,
    "eathand":8
    }

allowed_labels = ["drink","eat","groom","hang","sniff","rear","rest","walk","eathand"]
#if you batch size is 32 then you have 32 labels
#for every training samples 16 timestamps (in each of these 1024 embeddings)
labels_file = '/media/data_cifs/alekh/lstm/all_labels.p'
train_emb_file = '/media/data_cifs/alekh/lstm/all_video_embs.p'
val_labels_file = '/media/data_cifs/alekh/lstm/valid_labels_all.p'

test_labels_file = '/media/data_cifs/alekh/lstm/all_test_labels.p'
test_emb_file = '/media/data_cifs/alekh/lstm/all_test_embs_newloadnocropDIV_GRAY.p'

with open(labels_file, 'rb') as f:
    all_labels = pickle.load(f)


with open(val_labels_file, 'rb') as f:
    valid_labels = pickle.load(f)

shuffle(valid_labels)

all_videos = list(all_labels.keys())

#embs_dir = '/media/data_cifs/alekh/embeddings/'
#for video in all_videos:
#    labls = all_labels[video]
#    clss = Counter(labls)
#    if clss['none']>7700:
#        all_videos.remove(video)
#all_vid_embs = {}
#for emb_file in os.listdir(embs_dir):
#    with open(embs_dir + emb_file, 'rb') as f:
#        u = pickle._Unpickler(f)
#        u.encoding = 'latin1'
#        vid_embs = u.load()
#
#    vid_name = emb_file.split("__")[1][:-2]
#    all_vid_embs[vid_name] = vid_embs
#    print(vid_name)

with open(train_emb_file,'rb') as f:
    all_vid_embs = pickle.load(f)



epochs = 0
label_idx = 0

def load_data(batchsize, frame_steps=64, frame_stride=1, is_test=False):
    global epochs, label_idx
    i=0
    batch_labels = []
    batch_embs = []
    while i<batchsize:
        #test_selected_list = []
        if is_test:
            with open(test_labels_file, 'rb') as f:
                all_test_labels = pickle.load(f)

            with open(test_emb_file, 'rb') as f:
                all_test_vid_embs = pickle.load(f)

            for video in all_test_labels.keys():
                if i>=batchsize:
                    break
                labels = all_test_labels[video]
                test_vid_embs = all_test_vid_embs[video]

                for frame_start_idx in range(0,len(labels),1):
                    unallowed = False
                    for l in labels[frame_start_idx : frame_start_idx + frame_steps]:
                        if l not in allowed_labels:
                            unallowed = True
                            break
                    if unallowed:
                        continue

                    seq_labels = labels[frame_start_idx : frame_start_idx + frame_steps : frame_stride]
                    frame_embs = [test_vid_embs[x] for x in list(range(frame_start_idx, frame_start_idx + frame_steps, frame_stride))]
                                                                                                                                                                                          105,1         43%

                    batch_labels.append([BEHAVIOR_INDICES[l] for l in seq_labels])
                    batch_embs.append(frame_embs)
                    #test_selected_list.append([video, frame_start_idx])
                    i=i+1
                    if i>=batchsize:
                        break
            del all_test_labels, all_test_vid_embs
            break

        else:
#            video_idx = 30
#            while(video_idx == 30 or video_idx==51):
#            video_idx = np.random.randint(0,92)
#                
#            frame_start_idx = np.random.randint(16, 8000-frame_steps)
#            
#            video_name = all_videos[video_idx]
#            vid_labels = all_labels[video_name][frame_start_idx : frame_start_idx + frame_steps : frame_stride]

            #label_idx = np.random.randint(0, 438607)

            if(label_idx>=len(valid_labels)):
                epochs+=1
                label_idx = 0
                print("Completed epoch: ", epochs, " with ", len(valid_labels), " labels")

            sel_label = valid_labels[label_idx]
            label_idx+=1

            frame_start_idx = sel_label[1]
            video_name = sel_label[0]

            vid_labels = all_labels[video_name][frame_start_idx : frame_start_idx + frame_steps : frame_stride]

            if 'none' in vid_labels or 'dig' in vid_labels:
                #i-=1
                continue

            if (vid_labels[-1] == 'groom' or vid_labels[-1] == 'sniff') and np.random.randint(0,2):
                #print('skipped: ', vid_labels[-1])
                continue

            batch_labels.append([BEHAVIOR_INDICES[l] for l in vid_labels])

            try:
                vid_embs = all_vid_embs[video_name]
            except:
                i-=1
                print("vid not found")
                continue
            frame_embs = [vid_embs[x] for x in list(range(frame_start_idx, frame_start_idx + frame_steps, frame_stride))]
            batch_embs.append(frame_embs)



            i = i + 1
            # print(label_idx)

#    if is_test:
#        with open('test_selected_list.p', 'wb') as f:
#            pickle.dump(test_selected_list, f)

    batch_embs = torch.tensor(np.stack(batch_embs), dtype=torch.float)
    batch_labels = torch.tensor(np.stack(batch_labels), dtype=torch.long)
    return batch_embs, batch_labels


