# System libs
import os
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
#from scipy.misc import imsave
from mir_eval.separation import bss_eval_sources

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
from viz import plot_sSep01_loss_metrics, HTMLVisualizer
from dynamicimage import get_dynamic_image


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_sound1, self.net_frame, self.net_frame1, self.net_synthesizer, self.net_synthesizer1 = nets

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10


        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # The 1st stage only needs single frame
        # select one frame from the start of stream
        frame = [None for n in range(N)]
        indice = torch.LongTensor([0]).to(args.device)
        for n in range(N):
            frame[n] = torch.index_select(frames[n], 2, indice)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)



        # ##############  First stage  ###############
        # we name 1st stage as stage0 in this implementation, which is slightly different from paper (stage1)

        # LOG magnitude for mixture
        log_mag_mix = torch.log(mag_mix).detach()


        # 1. forward net_sound Bx1xHSxWS  --> BxCxHSxWS
        # No nonlinear operation is applied to sound features here
        # args.sound_activation is None
        feat_sound_stage0 = self.net_sound(log_mag_mix)
        feat_sound_stage0 = activate(feat_sound_stage0, args.sound_activation)


        # 2. forward net_frame  Bx3xTxHIxWI  --> Bx1xC
        # args.img_activation is sigmoid operation
        feat_frame_stage0 = [None for n in range(N)]
        for n in range(N):
            feat_frame_stage0[n] = self.net_frame.forward_multiframe(frame[n])
            feat_frame_stage0[n] = activate(feat_frame_stage0[n], args.img_activation)
        
        
        # 3. sound synthesizer vision: Bx1xC, sound: BxCxHSxWS, --> Bx1xHSxWS
        # args.output_activation is sigmoid operation
        pred_masks_stage0 = [None for n in range(N)]
        pred_masks_stage0_sigmoid = [None for n in range(N)]
        for n in range(N):
            pred_masks_stage0[n] = self.net_synthesizer(feat_frame_stage0[n], feat_sound_stage0)
            pred_masks_stage0_sigmoid[n] = activate(pred_masks_stage0[n], args.output_activation)


        # ##############  Second stage  ###############
        # we name 2nd stage as stage1 in this implementation, which is slightly different from paper (stage2)
        # stage 2 takes sound mixture and the sound separation prediction from previous stage as inputs

        # get separated spectrogram from previpus stage
        mag_mix_filter = [None for n in range(N)]
        for n in range(N):
            mag_mix_filter[n] = pred_masks_stage0_sigmoid[n]*mag_mix

        # LOG magnitude of each separated sounds from previous stage
        log_mag_mix_filter = [None for n in range(N)]
        for n in range(N):
            log_mag_mix_filter[n] = torch.log(mag_mix_filter[n]).detach()

        # get the dynamic image from the loaded frames
        # here we simply get one single dynamic image from all loaded sT frames
        # you can make corresponding changes by modifing the WINDOW_LENGTH and STRIDE to get more dynamic images
        sB, sC, sT, sH, sW = frames[0].size()   
        dynamic_frames = torch.zeros((N, sB, 1, sH, sW, sC)).float().to(args.device)
        WINDOW_LENGTH = sT
        #STRIDE = 6

        #dynamic_frames = torch.zeros((N, sB, int((sT-WINDOW_LENGTH)/STRIDE), sH, sW, sC)).float().to(args.device)
        for n in range(N):
            frames[n] = frames[n].permute(0, 2, 3, 4, 1).contiguous()
            for j in range(sB):
                current_video_frames = frames[n][j].cpu().numpy()
                dynamic_count = 0
                #for k in range(0, args.num_frames - WINDOW_LENGTH, STRIDE):
                #chunk = current_video_frames[k:k + WINDOW_LENGTH]
                chunk = current_video_frames[0:0 + WINDOW_LENGTH]
                assert len(chunk) == WINDOW_LENGTH
                dynamic_image = get_dynamic_image(chunk)
                dynamic_frames[n][j][dynamic_count] = torch.from_numpy(dynamic_image).float().to(args.device)
                dynamic_count += 1

        # permute channels to B, C, T, HI, WI)
        dynamic_frames_c = [None for n in range(N)]
        for n in range(N):
            dynamic_frames_c[n] = dynamic_frames[n].permute(0, 4, 1, 2, 3).contiguous()


        # 1. forward net_sound  Bx1xHSxWS  --> BxCxHSxWS
        # No nonlinear operation is applied to sound features here
        # args.sound_activation is None
        feat_sound_stage1 = [None for n in range(N)]
        for n in range(N):
            feat_sound_stage1[n] = self.net_sound1(log_mag_mix_filter[n])
            feat_sound_stage1[n] = activate(feat_sound_stage1[n], args.sound_activation)


        # 2. forward net_frame  Bx3xTxHIxWI  --> Bx1xC
        # args.img_activation is sigmoid operation
        feat_frame_stage1 = [None for n in range(N)]
        for n in range(N):
            feat_frame_stage1[n] = self.net_frame1.forward_multiframe(dynamic_frames_c[n])
            feat_frame_stage1[n] = activate(feat_frame_stage1[n], args.img_activation)

        # 3. sound synthesizer vision: Bx1xC, sound: BxCxHSxWS, --> Bx1xHSxWS
        # args.output_activation is sigmoid operation
        pred_masks_stage1 = [[] for n in range(N)]
        for n in range(N):
            pred_masks_tmp = [None for n in range(N)]
            for m in range(N):
                if m!=n:
                    # m-th visual cue look for relative component from n-th sounds (m!=n)
                    pred_masks_tmp[m] = self.net_synthesizer1(feat_frame_stage1[n], feat_sound_stage1[m])
                elif m==n:
                    pred_masks_tmp[m] = torch.zeros(mag_mix.size()).to(args.device)
                pred_masks_stage1[n].append(pred_masks_tmp[m])

        # initialize the final mask as the prediction from stage0
        final_pred_masks_ = [None for n in range(N)]
        for n in range(N):
            final_pred_masks_[n] = pred_masks_stage0[n].clone()


        # move sound components between different sounds
        for n in range(N):
            for j in range(N):
                if n != j:
                    # add missing component (recover from other sounds)
                    final_pred_masks_[n] = final_pred_masks_[n] + pred_masks_stage1[n][j]
                    # remove irrelative component (relative to other sounds)
                    final_pred_masks_[n] = final_pred_masks_[n] - pred_masks_stage1[j][n]

        # afterward nonlinear activation
        for n in range(N):
            final_pred_masks_[n] = activate(final_pred_masks_[n], args.output_activation)
            for j in range(N):
                if n != j:
                    pred_masks_stage1[n][j] = activate(pred_masks_stage1[n][j], args.output_activation)


        return {'pred_masks_stage0_sigmoid': pred_masks_stage0_sigmoid, 'filters': pred_masks_stage1, 'final_pred_masks_': final_pred_masks_, 'gt_masks': gt_masks, 'mag_mix': mag_mix, 'mags': mags, 'weight': weight}


# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]



# Visualize predictions
def output_visuals(vis_rows, batch_data, outputs_netWrapper, outputs1, outputs2, mid_pred_masks_, args):
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    pred_masks1_ = outputs1
    pred_masks2_ = outputs2
    gt_masks_ = outputs_netWrapper['gt_masks']
    mag_mix_ = outputs_netWrapper['mag_mix']
    weight_ = outputs_netWrapper['weight']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)

    mid_mag = [None for n in range(N)]
    mid_pred_masks_linear = [[] for n in range(N)]
    pred_masks1_linear = [None for n in range(N)]
    pred_masks2_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks1_linear[n] = F.grid_sample(pred_masks1_[n], grid_unwarp)
            pred_masks2_linear[n] = F.grid_sample(pred_masks2_[n], grid_unwarp)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)
            for m in range(N):
               #if n != m:
                mid_pred_masks_linear[n].append(F.grid_sample(mid_pred_masks_[n][m], grid_unwarp))

        else:
            pred_masks1_linear[n] = pred_masks1_[n]
            pred_masks2_linear[n] = pred_masks2_[n]
            gt_masks_linear[n] = gt_masks_[n]

            for m in range(N):
                #if n != m:
                mid_pred_masks_linear[n][m] = mid_pred_masks_[n][m]


    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    for n in range(N):
        pred_masks1_[n] = pred_masks1_[n].detach().cpu().numpy()
        pred_masks2_[n] = pred_masks2_[n].detach().cpu().numpy()
        pred_masks1_linear[n] = pred_masks1_linear[n].detach().cpu().numpy()
        pred_masks2_linear[n] = pred_masks2_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()
        mid_mag[n] = pred_masks1_linear[n] * mag_mix
        for m in range(N):
            if n != m:
                mid_pred_masks_[n][m] = mid_pred_masks_[n][m].detach().cpu().numpy()
                mid_pred_masks_linear[n][m] = mid_pred_masks_linear[n][m].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks1_[n] = (pred_masks1_[n] > args.mask_thres).astype(np.float32)
            pred_masks1_linear[n] = (pred_masks1_linear[n] > args.mask_thres).astype(np.float32)
            pred_masks2_[n] = (pred_masks2_[n] > args.mask_thres).astype(np.float32)
            pred_masks2_linear[n] = (pred_masks2_linear[n] > args.mask_thres).astype(np.float32)

            for m in range(N):
                if n != m:
                    mid_pred_masks_linear[n][m] = (mid_pred_masks_linear[n][m] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        matplotlib.image.imsave(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        matplotlib.image.imsave(os.path.join(args.vis, filename_weight), weight[::-1, :])
        
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav1 = [None for n in range(N)]
        preds_wav2 = [None for n in range(N)]
        for n in range(N):
            # GT and predicted audio recovery
            gt_mag_ = mag_mix_[j, 0] * gt_masks_[n][j, 0]
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag1_ = mag_mix_[j, 0] * pred_masks1_[n][j, 0]
            pred_mag2_ = mag_mix_[j, 0] * pred_masks2_[n][j, 0]
            pred_mag1 = mag_mix[j, 0] * pred_masks1_linear[n][j, 0]
            pred_mag2 = mag_mix[j, 0] * pred_masks2_linear[n][j, 0]
            preds_wav1[n] = istft_reconstruction(pred_mag1, phase_mix[j, 0], hop_length=args.stft_hop)
            preds_wav2[n] = istft_reconstruction(pred_mag2, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask1 = os.path.join(prefix, 'predmaska{}.jpg'.format(n+1))
            filename_predmask2 = os.path.join(prefix, 'predmaskb{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask1 = (np.clip(pred_masks1_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask2 = (np.clip(pred_masks2_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            matplotlib.image.imsave(os.path.join(args.vis, filename_gtmask), gt_mask[::-1, :])
            matplotlib.image.imsave(os.path.join(args.vis, filename_predmask1), pred_mask1[::-1, :])
            matplotlib.image.imsave(os.path.join(args.vis, filename_predmask2), pred_mask2[::-1, :])
            for m in range(N):
                if n != m:
                    filename_predmask = os.path.join(prefix, 'predmask_{}_{}.jpg'.format(n+1, m+1))
                    mid_pred_mask_tmp = (np.clip(mid_pred_masks_[n][m][j, 0], 0, 1) * 255).astype(np.uint8)
                    matplotlib.image.imsave(os.path.join(args.vis, filename_predmask), mid_pred_mask_tmp[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag1 = os.path.join(prefix, 'predampa{}.jpg'.format(n+1))
            filename_predmag2 = os.path.join(prefix, 'predampb{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag_)
            pred_mag1 = magnitude2heatmap(pred_mag1_)
            pred_mag2 = magnitude2heatmap(pred_mag2_)
            matplotlib.image.imsave(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            matplotlib.image.imsave(os.path.join(args.vis, filename_predmag1), pred_mag1[::-1, :, :])
            matplotlib.image.imsave(os.path.join(args.vis, filename_predmag2), pred_mag2[::-1, :, :])
	    #for m in range(N):
            #    if n != m:
            #        mid_pred_mag = mid_mag[m][j, 0] * mid_pred_masks_linear[n][m][j, 0]
            #        mid_pred_mag = magnitude2heatmap(mid_pred_mag)
            #        filename_predmag = os.path.join(prefix, 'predmag_{}_{}.jpg'.format(n+1, m+1))
            #        matplotlib.image.imsave(os.path.join(args.vis, filename_predmag), mid_pred_mag[::-1, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav1 = os.path.join(prefix, 'preda{}.wav'.format(n+1))
            filename_predwav2 = os.path.join(prefix, 'predb{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav1), args.audRate, preds_wav1[n])
            wavfile.write(os.path.join(args.vis, filename_predwav2), args.audRate, preds_wav2[n])

            #row_elements += [
            #    #{'video': filename_av},
            #    {'image': filename_predmag2, 'audio': filename_predwav2},
            #    {'image': filename_gtmag, 'audio': filename_gtwav},
            #    {'image': filename_predmask2},
            #    {'image': filename_gtmask}]

        #row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)



def evaluate(crit, netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()
    loss_meter = AverageMeter()
    sdr_mix1_meter = AverageMeter()
    sdr1_meter = AverageMeter()
    sir1_meter = AverageMeter()
    sar1_meter = AverageMeter()
    sdr_mix2_meter = AverageMeter()
    sdr2_meter = AverageMeter()
    sir2_meter = AverageMeter()
    sar2_meter = AverageMeter()

    # initialize HTML header
    #visualizer = HTMLVisualizer(os.path.join(args.vis, 'index.html'))
    #header = ['Filename', 'Input Mixed Audio']
    #for n in range(1, args.num_mix+1):
    #    header += [#'Video {:d}'.format(n),
    #               'Predicted Audio {:d}'.format(n),
    #               'GroundTruth Audio {}'.format(n),
    #               'Predicted Mask {}'.format(n),
    #               'GroundTruth Mask {}'.format(n)]
    #header += ['Loss weighting']
    #visualizer.add_header(header)
    vis_rows = []

    
    for i, batch_data in enumerate(loader):
        # forward pass
        outputs_netWrapper = netWrapper.forward(batch_data, args)
        pred_masks0_ = outputs_netWrapper['pred_masks_stage0_sigmoid']
        pred_masks1_ = outputs_netWrapper['final_pred_masks_']
        filters = outputs_netWrapper['filters'] # 2nd stage (stage1) output


        loss1 = crit(pred_masks0_, outputs_netWrapper['gt_masks'], outputs_netWrapper['weight']).reshape(1)
        loss2 = crit(pred_masks1_, outputs_netWrapper['gt_masks'], outputs_netWrapper['weight']).reshape(1)
        loss = loss1 + loss2
        err = loss.mean()

        loss1_meter.update(loss1.item())
        loss2_meter.update(loss2.item())
        loss_meter.update(err.item())
        print('[Eval] iter {}, loss: {:.4f} loss1: {:.4f} loss2: {:.4f}'.format(i, err.item(), loss1.item(), loss2.item()))
        
 
        # calculate metrics
        sdr_mix1, sdr1, sir1, sar1 = calc_metrics(batch_data, pred_masks0_, args)
        sdr_mix1_meter.update(sdr_mix1)
        sdr1_meter.update(sdr1)
        sir1_meter.update(sir1)
        sar1_meter.update(sar1)
        sdr_mix2, sdr2, sir2, sar2 = calc_metrics(batch_data, pred_masks1_, args)
        sdr_mix2_meter.update(sdr_mix2)
        sdr2_meter.update(sdr2)
        sir2_meter.update(sir2)
        sar2_meter.update(sar2)


        # output visualization
        if len(vis_rows) < args.num_vis:
            output_visuals(vis_rows, batch_data, outputs_netWrapper, pred_masks0_, pred_masks1_, filters, args)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, loss1: {:.4f} loss2: {:.4f}, '
          'SDR_mixture1: {:.4f}, SDR1: {:.4f}, SIR1: {:.4f}, SAR1: {:.4f} '
          'SDR_mixture2: {:.4f}, SDR2: {:.4f}, SIR2: {:.4f}, SAR2: {:.4f} '
          .format(epoch, loss_meter.average(), loss1_meter.average(), loss2_meter.average(),
                  sdr_mix1_meter.average(),
                  sdr1_meter.average(),
                  sir1_meter.average(),
                  sar1_meter.average(),
                  sdr_mix2_meter.average(),
                  sdr2_meter.average(),
                  sir2_meter.average(),
                  sar2_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['err1'].append(loss1_meter.average())
    history['val']['err2'].append(loss2_meter.average())
    history['val']['sdr1'].append(sdr1_meter.average())
    history['val']['sir1'].append(sir1_meter.average())
    history['val']['sar1'].append(sar1_meter.average())
    history['val']['sdr2'].append(sdr2_meter.average())
    history['val']['sir2'].append(sir2_meter.average())
    history['val']['sar2'].append(sar2_meter.average())

    #print('Plotting html for visualization...')
    #visualizer.add_rows(vis_rows)
    #visualizer.write_html()

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_sSep01_loss_metrics(args.ckpt, history)
    print('this evaluation round is done!')

# train one epoch
def train(crit, netWrapper,  loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # forward pass
        netWrapper.zero_grad()
        outputs_netWrapper = netWrapper.forward(batch_data, args)
        pred_masks0_ = outputs_netWrapper['pred_masks_stage0_sigmoid']
        pred_masks1_ = outputs_netWrapper['final_pred_masks_']

        loss1 = crit(pred_masks0_, outputs_netWrapper['gt_masks'], outputs_netWrapper['weight']).reshape(1)
        loss2 = crit(pred_masks1_, outputs_netWrapper['gt_masks'], outputs_netWrapper['weight']).reshape(1)
        loss = loss1 + loss2
        err = loss.mean()


        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_synthesizer: {}, '
                  'loss: {:.4f}, loss1: {:.4f}, loss2: {:.4f} '
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_synthesizer,
                          err.item(), loss1.item(), loss2.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())
            history['train']['err1'].append(loss1.item())
            history['train']['err2'].append(loss2.item())


def checkpoint_full(nets, optimizer, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_sound1, net_frame, net_frame1, net_synthesizer, net_synthesizer1) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    state = {'epoch': epoch, \
             'state_dict_net_sound': net_sound.state_dict(), \
             'state_dict_net_sound1': net_sound1.state_dict(), \
             'state_dict_net_frame': net_frame.state_dict(),\
             'state_dict_net_frame1': net_frame1.state_dict(),\
             'state_dict_net_synthesizer': net_synthesizer.state_dict(),\
             'state_dict_net_synthesizer1': net_synthesizer1.state_dict(),\
             'optimizer': optimizer.state_dict(), \
             'train_history': history, }

    torch.save(state, '{}/checkpoint_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(state, '{}/checkpoint_{}'.format(args.ckpt, suffix_best))


def load_checkpoint(nets, optimizer, history, filename):
    (net_sound, net_sound1, net_frame, net_frame1, net_synthesizer, net_synthesizer1) = nets
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        net_sound.load_state_dict(checkpoint['state_dict_net_sound'])
        net_sound1.load_state_dict(checkpoint['state_dict_net_sound1'])
        net_frame.load_state_dict(checkpoint['state_dict_net_frame'])
        net_frame1.load_state_dict(checkpoint['state_dict_net_frame1'])
        net_synthesizer.load_state_dict(checkpoint['state_dict_net_synthesizer'])
        net_synthesizer1.load_state_dict(checkpoint['state_dict_net_synthesizer1'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        history = checkpoint['train_history']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    nets = (net_sound, net_sound1, net_frame, net_frame1, net_synthesizer, net_synthesizer1)
    return nets, optimizer, start_epoch, history

def load_checkpoint_from_train(nets, filename):
    (net_sound, net_sound1, net_frame, net_frame1, net_synthesizer, net_synthesizer1) = nets
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        print('epoch: ', checkpoint['epoch'])
        net_sound.load_state_dict(checkpoint['state_dict_net_sound'])
        net_sound1.load_state_dict(checkpoint['state_dict_net_sound1'])
        net_frame.load_state_dict(checkpoint['state_dict_net_frame'])
        net_frame1.load_state_dict(checkpoint['state_dict_net_frame1'])
        net_synthesizer.load_state_dict(checkpoint['state_dict_net_synthesizer'])
        net_synthesizer1.load_state_dict(checkpoint['state_dict_net_synthesizer1'])

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    nets = (net_sound, net_sound1, net_frame, net_frame1, net_synthesizer, net_synthesizer1)
    return nets


def create_optimizer(nets, args):
    (net_sound, net_sound1, net_frame, net_frame1, net_synthesizer, net_synthesizer1) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_sound1.parameters(), 'lr': args.lr_sound},
                    {'params': net_synthesizer.parameters(), 'lr': args.lr_synthesizer},
                    {'params': net_synthesizer1.parameters(), 'lr': args.lr_synthesizer},
                    {'params': net_frame.features.parameters(), 'lr': args.lr_frame},
                    {'params': net_frame.fc.parameters(), 'lr': args.lr_sound},
                    {'params': net_frame1.features.parameters(), 'lr': args.lr_frame},
                    {'params': net_frame1.fc.parameters(), 'lr': args.lr_sound}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_synthesizer *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame)
    net_synthesizer = builder.build_synthesizer(
        arch=args.arch_synthesizer,
        fc_dim=args.num_channels,
        weights=args.weights_synthesizer)

    net_sound1 = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)
    net_frame1 = builder.build_frame(
        arch=args.arch_frame,#'dynamic_res18',#
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame)
    net_synthesizer1 = builder.build_synthesizer(
        arch=args.arch_synthesizer,
        fc_dim=args.num_channels,
        weights=args.weights_synthesizer)
    nets = (net_sound, net_sound1, net_frame, net_frame1, net_synthesizer, net_synthesizer1)
    crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),#2,
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))


    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': [], 'err1': [], 'err2': []},
        'val': {'epoch': [], 'err': [], 'err1': [], 'err2': [], 'sdr1': [], 'sir1': [], 'sar1': [], 'sdr2': [], 'sir2': [], 'sar2': []}}


    # Training loop
    
    # Load checkpoint if needed!
    start_epoch = 1
    model_name = args.ckpt + '/checkpoint.pth'
    if os.path.exists(model_name):
        if args.mode == 'eval':
            nets = load_checkpoint_from_train(nets, model_name)
        elif args.mode == 'train':
            model_name = args.ckpt + '/checkpoint_latest.pth'
            nets, optimizer, start_epoch, history = load_checkpoint(nets, optimizer, history, model_name)
        print("Loading from checkpoint successfully.")

    # Wrap networks
    netWrapper = NetWrapper(nets)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)


    # Eval mode
    #evaluate(crit, netWrapper, loader_val, history, start_epoch-1, args)
    if args.mode == 'eval':
        evaluate(crit, netWrapper, loader_val, history, start_epoch-1, args)
        print('Evaluation Done!')
        return


    for epoch in range(start_epoch, args.num_epoch + 1):    
        train(crit, netWrapper,  loader_train, optimizer, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

        ## Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(crit, netWrapper, loader_val, history, epoch, args)

            # checkpointing
            checkpoint_full(nets, optimizer, history, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    #args.device = torch.device("cuda")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    if args.mode == 'train':
        args.vis = os.path.join(args.ckpt, 'visualization_train/')
        makedirs(args.ckpt, remove=True)
    elif args.mode == 'eval':
        args.vis = os.path.join(args.ckpt, 'visualization_val/')

    # initialize best error with inf
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
