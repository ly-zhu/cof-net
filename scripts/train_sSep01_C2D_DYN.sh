#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "

# S: solo
# D: Duet
OPTS+="--list_train data/music_TrainS335_D65_f8fps_11k.csv "
#OPTS+="--list_train data/music_TrainS335_f8fps_11k.csv "
OPTS+="--list_val data/music_ValValS100_f8fps_11k.csv "
#OPTS+="--list_val data/music_ValTestS130_f8fps_11k.csv "
#OPTS+="--list_val data/music_TestD84_f8fps_11k.csv "


# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 16 "

# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "

# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 12 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 48 "
OPTS+="--batch_size_per_gpu 10 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "
OPTS+="--dup_trainset 100 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

# checkpoint
OPTS+="--ckpt ./ckpt_sSep01_C2D_DYN_N2_f12_trainOnce_SD_bs10_dup100_f8fps_11k "

# execute
CUDA_VISIBLE_DEVICES="2" python -u main_sSep01_C2D_DYN.py $OPTS
