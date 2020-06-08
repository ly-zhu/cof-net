import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from .synthesizer_net import InnerProd, Bias
from .audio_net import Unet
from .vision_net import ResnetFC, ResnetDilated
from .criterion import BCELoss, LogL1Loss, L1Loss, L2Loss, CELoss, BCELoss_noWeight, MSELoss
from .net_mask import drn_d_22, AutoED
import numpy as np
import torch.nn.init as init
import math

def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.001)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            #m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            #m.weight.data.normal_(0.0, 0.0001)
            y = m.in_features
            m.weight.data.normal_(0.0, 1/np.sqrt(y))#0.0001)

    def weights_init_bu1(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            y = m.in_features
            m.weight.data.normal_(0.0, 1/np.sqrt(y))#0.0001)


    def build_sound(self, arch='unet5', input_channel=1, fc_dim=64, weights=''):
        # 2D models
        if arch == 'unet5':
            net_sound = Unet(fc_dim=fc_dim, num_downs=5)
        elif arch == 'unet6':
            net_sound = Unet(fc_dim=fc_dim, num_downs=6)
        elif arch == 'unet7':
            net_sound = Unet(input_channel=input_channel, fc_dim=fc_dim, num_downs=7)
        else:
            raise Exception('Architecture undefined!')

        net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound


    # builder for vision
    def build_frame(self, arch='resnet18', fc_dim=64, pool_type='avgpool',
                    weights=''):
        pretrained=True
        if arch == 'resnet18fc':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetFC(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetDilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_synthesizer(self, arch, fc_dim=64, weights=''):
        if arch == 'linear':
            net = InnerProd(fc_dim=fc_dim)
        elif arch == 'bias':
            net = Bias()
        else:
            raise Exception('Architecture undefined!')

        net.apply(self.weights_init_bu1)
        if len(weights) > 0:
            print('Loading weights for net_synthesizer')
            net.load_state_dict(torch.load(weights))
        return net


    # builder for vision
    def build_mask(self, arch='drn_d_22', fc_dim=64,
                    weights=''):
        pretrained=True
        if arch == 'drn_d_22':
            original_model2 = drn_d_22(pretrained=pretrained)
            net_mask = AutoED(original_model2)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net_mask.load_state_dict(torch.load(weights))
        return net_mask



    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        elif arch == 'ce':
            net = CELoss()
        elif arch == 'logl1':
            net = LogL1Loss()
        elif arch == 'mse':
            net = MSELoss()
        else:
            raise Exception('Architecture undefined!')
        return net
