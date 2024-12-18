# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import (
    surrogate,
    neuron,
    functional,
    layer
)
from configs import config
import numpy as np


bn_mom = 0.1
algc = False

class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = True

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

    def forward(self, x):
        # shape correction
        if self._spiking is not True and len(x.shape) == 5:
            x = x.mean([0])
        return x

def spike_activation(x, ste=False, temp=1.0):
    out_s = torch.gt(x, 0.5) # torch.gt:逐元素对比x>0.5?
    if ste:
        out_bp = torch.clamp(x, 0, 1) # 将x中的元素限制[0,1]范围内
    else:
        out_bp = torch.clamp(x, 0, 1)
        out_bp = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
    return (out_s.float() - out_bp).detach() + out_bp


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def mem_update(x_in, mem, V_th, decay, grad_scale=1., temp=1.0):
    mem = mem * decay + x_in
    #if mem.shape[1]==256:
    #    embed()
    #V_th = gradient_scale(V_th, grad_scale)
    spike = spike_activation(mem / V_th, temp=temp)
    mem = mem * (1 - spike)
    #mem = 0
    #spike = spike * Fire_ratio
    return mem, spike


class LIFAct(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, step):
        super(LIFAct, self).__init__()
        self.step = step
        #self.V_th = nn.Parameter(torch.tensor(1.))
        self.V_th = 1.0
        # self.tau = nn.Parameter(torch.tensor(-1.1))
        self.temp = 3.0
        #self.temp = nn.Parameter(torch.tensor(1.))
        self.grad_scale = 0.1

    def forward(self, x):
        if self._spiking is not True:
            return F.relu(x)
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel()*self.step)
        u = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            u, out_i = mem_update(x_in=x[i], mem=u, V_th=self.V_th,
                                  grad_scale=self.grad_scale, decay=0.25, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out


class tdBatchNorm(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0
        VTH = 1.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1, 3, 4])
            # use biased var in train
            var = input.var([0, 1, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * VTH * (input - mean[None, None, :, None, None]) / (torch.sqrt(var[None, None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, None, :, None, None] + self.bias[None, None, :, None, None]

        return input

    
class BasicInterpolate(nn.Module):
    def __init__(self, size, mode, align_corners):
        super(BasicInterpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        times_window, batch_size = x.shape[0], x.shape[1]
        # [t,b,c,h,w,]->[t*b,c,h,w]
        x = x.reshape(-1, *x.shape[2:])
        x = F.interpolate(x, size=self.size, mode=self.mode,
                          align_corners=self.align_corners)
        # [t*b,c,h,w]->[t,b,c,h,w]
        x = x.view(times_window, batch_size, *x.shape[1:])
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False, connect_f=None):
        super(BasicBlock, self).__init__()
        # self.connect_f = connect_f

        self.conv1 = layer.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(planes, momentum=bn_mom)
        self.lif1 = LIFAct(step = config.DATASET.nr_temporal_bins)
        #self.relu = nn.ReLU(inplace=True)

        self.conv2 = layer.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(planes, momentum=bn_mom)
        self.lif2 = LIFAct(step = config.DATASET.nr_temporal_bins)

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # if self.connect_f == 'ADD':
        #     out += residual
        # elif self.connect_f == 'AND':
        #     out *= residual
        # elif self.connect_f == 'IAND':
        #     out = residual * (1. - out)
        # else:
        #     raise NotImplementedError(self.connect_f)

        if self.no_relu:
            return out
        else:
            return self.lif2(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True, connect_f=None,
                 spiking_neuron="LIFNode", surrogate_function="ATan", detach_reset=True, v_reset=None):
        super(Bottleneck, self).__init__()
        # self.connect_f = connect_f

        self.conv1 = layer.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = layer.BatchNorm2d(planes, momentum=bn_mom)
        self.lif1 = LIFAct(step = config.DATASET.nr_temporal_bins)

        self.conv2 = layer.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(planes, momentum=bn_mom)
        self.lif2 = LIFAct(step = config.DATASET.nr_temporal_bins)

        self.conv3 = layer.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = layer.BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.lif3 = LIFAct(step = config.DATASET.nr_temporal_bins)
        #self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.relu(out)
        out = self.lif2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # if self.connect_f == 'ADD':
        #     out += residual
        # elif self.connect_f == 'AND':
        #     out *= residual
        # elif self.connect_f == 'IAND':
        #     out = residual * (1. - out)
        # else:
        #     raise NotImplementedError(self.connect_f)

        if self.no_relu:
            return out
        else:
            return self.lif3(out)


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None,
                 spiking_neuron="LIFNode", surrogate_function="ATan", detach_reset=True, v_reset=None):
        super(segmenthead, self).__init__()

        self.bn1 = layer.BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = layer.Conv2d(
            inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.lif1 = LIFAct(step = config.DATASET.nr_temporal_bins)

        self.bn2 = layer.BatchNorm2d(interplanes, momentum=bn_mom)
        self.conv2 = layer.Conv2d(
            interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.lif2 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif3 = LIFAct(step = config.DATASET.nr_temporal_bins)

        self.scale_factor = scale_factor

    def forward(self, x):

        x = self.conv1(self.lif1(self.bn1(x)))
        out = self.conv2(self.lif2(self.bn2(x)))
        # x = self.conv1(self.relu(self.bn1(x)))
        # out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = BasicInterpolate(size=[height, width], mode='bilinear',
                                   align_corners=algc)(out)
            # out = F.interpolate(out,
            #             size=[height, width],
            #             mode='bilinear', align_corners=algc)
        # out = self.lif3(out)
        return out


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes,
                 spiking_neuron="LIFNode", surrogate_function="ATan", detach_reset=True, v_reset=None):
        super(DAPPM, self).__init__()

        bn_mom = 0.1
        self.scale1 = nn.Sequential(
            layer.AvgPool2d(kernel_size=5, stride=2, padding=2),
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            layer.AvgPool2d(kernel_size=9, stride=4, padding=4),
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            layer.AvgPool2d(kernel_size=17, stride=8, padding=8),
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            layer.BatchNorm2d(branch_planes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            layer.BatchNorm2d(branch_planes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            layer.BatchNorm2d(branch_planes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            layer.BatchNorm2d(branch_planes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            layer.BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x:[t,b,c,h,w]
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((BasicInterpolate(size=[height, width],
                                                      mode='bilinear', align_corners=algc)(self.scale1(x))+x_list[0])))
        x_list.append((self.process2((BasicInterpolate(size=[height, width],
                                                       mode='bilinear', align_corners=algc)(self.scale2(x))+x_list[1]))))
        x_list.append(self.process3((BasicInterpolate(size=[height, width],
                                                      mode='bilinear', align_corners=algc)(self.scale3(x))+x_list[2])))
        x_list.append(self.process4((BasicInterpolate(size=[height, width],
                                                      mode='bilinear', align_corners=algc)(self.scale4(x))+x_list[3])))

        out = self.compression(torch.cat(x_list, 2)) + self.shortcut(x)
        # 输出的是conv后的real value
        # out:[t,b,c,h,w]
        return out


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes,
                 spiking_neuron="LIFNode", surrogate_function="ATan", detach_reset=True, v_reset=None):
        super(PAPPM, self).__init__()
        bn_mom = 0.1

        self.scale1 = nn.Sequential(
            layer.AvgPool2d(kernel_size=5, stride=2, padding=2),
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            layer.AvgPool2d(kernel_size=9, stride=4, padding=4),
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            layer.AvgPool2d(kernel_size=17, stride=8, padding=8),
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale0 = nn.Sequential(
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(
                inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale_process = nn.Sequential(
            layer.BatchNorm2d(branch_planes*4, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(branch_planes*4, branch_planes*4,
                         kernel_size=3, padding=1, groups=4, bias=False),
        )

        self.compression = nn.Sequential(
            layer.BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(branch_planes * 5, outplanes,
                         kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            layer.BatchNorm2d(inplanes, momentum=bn_mom),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(BasicInterpolate(size=[height, width], mode='bilinear',
                                           align_corners=algc)(self.scale1(x))+x_)
        scale_list.append(BasicInterpolate(size=[height, width], mode='bilinear',
                                           align_corners=algc)(self.scale2(x))+x_)
        scale_list.append(BasicInterpolate(size=[height, width], mode='bilinear',
                                           align_corners=algc)(self.scale3(x))+x_)
        scale_list.append(BasicInterpolate(size=[height, width], mode='bilinear',
                                           align_corners=algc)(self.scale4(x))+x_)
        # [t,b,c,h,w]
        scale_out = self.scale_process(torch.cat(scale_list, 2))

        out = self.compression(
            torch.cat([x_, scale_out], 2)) + self.shortcut(x)
        return out


class HammingDistancesimilarity(nn.Module):
    def __init__(self):
        super(HammingDistancesimilarity, self).__init__()

    def forward(self, q, k):

        #  --- spike相似度计算+融合时间维度---
        #  input is 5 dimension tensor : [t, b, c, h, w]
        #  改变维度：[t, b, c, h, w] -> [b, t, c, h, w] -> [b, t*c, h, w]
        # t, b, c, h, w = q.size()
        # q = q.permute(1, 0, 2, 3, 4).contiguous().view(b, t*c, h, w)
        # k = k.permute(1, 0, 2, 3, 4).contiguous().view(b, t*c, h, w)
        # d = q.shape[1]

        # # score: [b, t*c, h, w]
        # score = (q == k) * 1.0
        # # score: [b,t*c,h,w]->[b,h,w]->[b,1,h,w]
        # score = (torch.sum(score, dim=1) / d)
        # score = score.unsqueeze(1)
        
        # --- spike相似度计算，不融合时间维度 ---
        d = q.shape[2]
        # score=q=k: [t,b,c,h,w]
        score = (q == k) * 1.0
        # score: [t,b,c,h,w]->[t,b,1,h,w]
        score = (torch.sum(score, dim=2) / d)
        score = score.unsqueeze(2)
        
        #  --- 实值相似度计算 ---
        # input: [t, b, c, h, w], output: [t, b, 1, h, w]       
        # score = torch.sigmoid(torch.sum(q*k, dim=2).unsqueeze(2))
        
        return score


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.similarity = HammingDistancesimilarity()
        self.after_relu = after_relu

        self.f_x = nn.Sequential(
            layer.Conv2d(in_channels, mid_channels,
                         kernel_size=1, bias=False),
            layer.BatchNorm2d(mid_channels),
        )
        self.f_y = nn.Sequential(
            layer.Conv2d(in_channels, mid_channels,
                         kernel_size=1, bias=False),
            layer.BatchNorm2d(mid_channels),
        )
        self.lif1 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif2 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif3 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif4 = LIFAct(step = config.DATASET.nr_temporal_bins)

        # if with_channel:
        #     self.up = nn.Sequential(
        #                             layer.Conv2d(mid_channels, in_channels,
        #                                       kernel_size=1, bias=False),
        #                             layer.BatchNorm2d(in_channels),
        #                            )

        # if after_relu:
        #     self.lif = neuron.LIFNode(step_mode='m')

    def forward(self, x, y):
        input_size = x.size()  # input_size=[t,b,c,h,w]
        if self.after_relu:
            y = self.lif1(y)
            x = self.lif2(x)

        y_q = self.f_y(y)  # y_q:[t,b,c,h,w]
        y_q = BasicInterpolate(size=[input_size[-2], input_size[-1]],
                               mode='bilinear', align_corners=False)(y_q)
        x_k = self.f_x(x)

        # if self.with_channel:
        #     sim_map = torch.sigmoid(self.up(x_k * y_q))
        # else:
        #     sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))


        # -- spike相似度计算 --
        # sim_map: [t,b,c,h,w]
        sim_map = self.similarity(self.lif3(y_q), self.lif4(x_k))
        # x/y: [t,b,c,h,w]
        y = BasicInterpolate(size=[input_size[-2], input_size[-1]], mode='bilinear',
                             align_corners=False)(y)
        
        # # --- 融合t维度 ---
        # t, b, c, h, w = x.shape
        # # [t,b,c,h,w]->[b,t,c,h,w]->[b,t*c,h,w]
        # x = x.permute(1, 0, 2, 3, 4).contiguous().view(b, t*c, h, w)
        # y = y.permute(1, 0, 2, 3, 4).contiguous().view(b, t*c, h, w)
        # # [b,t*c,h,w] * [b, t*c, h, w]
        # x = (1-sim_map)*x + sim_map*y
        # # [b, t*c, h, w] -> [b, t, c, h, w] -> [t, b, c, h, w]
        # x = x.view(b, t, c, h, w).permute(1, 0, 2, 3, 4).contiguous()
        
        
        # --- 不融合t维度 ---
        x = (1-sim_map)*x + sim_map*y
        
        
        # --- 实值相似度计算，不融合t维度 ---
        # sim_map:[t,b,1,h,w]
        # sim_map = self.similarity(y_q, x_k)
        
        # y = BasicInterpolate(size=[input_size[-2], input_size[-1]], mode='bilinear', 
        #                      align_corners=False)(y)
        # # x/y:[t,b,c,h,w]
        # x = (1-sim_map)*x + sim_map*y
        return x


class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Light_Bag, self).__init__()

        self.conv_p = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            layer.BatchNorm2d(out_channels)
        )
        self.conv_i = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            layer.BatchNorm2d(out_channels)
        )
        # add
        self.conv = nn.Sequential(
            layer.BatchNorm2d(in_channels),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(in_channels, out_channels,
                         kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i, d):
        # d:[t,b,c,h,w]
        # 浮点数乘法
        edge_att = torch.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        return p_add + i_add
        
        # 浮点相加
        # edge_att = torch.sigmoid(d)
        # mask = (edge_att > 0.5).float()
        # return self.conv(mask * p + (1 - mask) * i)
        
        # mid = torch.median(d)
        # mask = (d > mid).float()
        # return self.conv(mask * p + (1 - mask) * i)

        


class DDFMv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDFMv2, self).__init__()

        self.conv_p = nn.Sequential(
            layer.BatchNorm2d(in_channels),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            layer.BatchNorm2d(out_channels)
        )
        self.conv_i = nn.Sequential(
            layer.BatchNorm2d(in_channels),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            layer.BatchNorm2d(out_channels)
        )
        self.conv = nn.Sequential(
            layer.BatchNorm2d(in_channels),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(in_channels, out_channels,
                         kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)

        return p_add + i_add
        
        #浮点相加
        # edge_att = torch.sigmoid(d)
        # mask = (edge_att > 0.5).float()
        # return self.conv(mask * p + (1 - mask) * i)
        
        # mid = torch.median(d)
        # mask = (d > mid).float()
        # return self.conv(mask * p + (1 - mask) * i)


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
            layer.BatchNorm2d(in_channels),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            # nn.ReLU(inplace=True),
            layer.Conv2d(in_channels, out_channels,
                         kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i, d):   
        # 浮点数乘法
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att*p + (1-edge_att)*i)
        
        # 只有浮点数加法
        # edge_att = torch.sigmoid(d)
        # mask = (edge_att > 0.5).float()
        # return self.conv(mask * p + (1 - mask) * i)
        
        # mid = torch.median(d)
        # mask = (d > mid).float()
        # return self.conv(mask * p + (1 - mask) * i)


if __name__ == '__main__':

    x = torch.rand(4, 64, 32, 64).cuda()
    y = torch.rand(4, 64, 32, 64).cuda()
    z = torch.rand(4, 64, 32, 64).cuda()
    net = PagFM(64, 16, with_channel=True).cuda()

    out = net(x, y)
