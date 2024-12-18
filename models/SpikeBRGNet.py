# ------------------------------------------------------------------------------
# Written by Xiaxin Zhu (xiaxinZhu@cqu.stu.edu.cn)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
from .model_utils import BasicInterpolate
import logging
from .model_utils import LIFAct, tdBatchNorm
from configs import config

from spikingjelly.activation_based import (
    surrogate,
    neuron,
    functional,
    layer
)

bn_mom = 0.1
algc = False



class SpikeBRGNet(nn.Module):

    def __init__(self, input_channel, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, 
                 head_planes=128, augment=True, connect_f=None):
        super(SpikeBRGNet, self).__init__()
        self.augment = augment
        
        self.lif1 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif2 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif3 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif4 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif5 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif6 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif7 = LIFAct(step = config.DATASET.nr_temporal_bins)
        self.lif8 = LIFAct(step = config.DATASET.nr_temporal_bins)
        
        # I Branch
        self.conv1 =  nn.Sequential(
            layer.Conv2d(input_channel, planes, kernel_size=3, stride=2, padding=1),
            layer.BatchNorm2d(planes),
            LIFAct(step = config.DATASET.nr_temporal_bins),
            #   nn.ReLU(inplace=True),
            layer.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            layer.BatchNorm2d(planes),
            LIFAct(step = config.DATASET.nr_temporal_bins)
            #   nn.ReLU(inplace=True),
        )

        #self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m, connect_f=connect_f)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2, connect_f=connect_f)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2, connect_f=connect_f)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2, connect_f=connect_f)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2, connect_f=connect_f)
        
        # P Branch
        self.compression3 = nn.Sequential(
                layer.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
                layer.BatchNorm2d(planes*2),
        )
        self.compression4 = nn.Sequential(
                layer.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
                layer.BatchNorm2d(planes*2),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m, connect_f=connect_f)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m, connect_f=connect_f)
        # self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1, connect_f=connect_f)
        self.layer5_ = self._make_single_layer(Bottleneck, planes * 2, planes * 2, no_relu=True, connect_f=connect_f)
        
        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes, no_relu=True, connect_f=connect_f)
            # self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1, connect_f=connect_f)
            self.layer4_d = self._make_single_layer(Bottleneck, planes, planes, no_relu=True, connect_f=connect_f)
            self.diff3 = nn.Sequential(
                    layer.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                    layer.BatchNorm2d(planes),
            )
            self.diff4 = nn.Sequential(
                    layer.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                    layer.BatchNorm2d(planes*2),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2, no_relu=True, connect_f=connect_f)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2, no_relu=True, connect_f=connect_f)
            self.diff3 = nn.Sequential(
                    layer.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                    layer.BatchNorm2d(planes*2),
            )
            self.diff4 = nn.Sequential(
                    layer.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                    layer.BatchNorm2d(planes*2),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
            
            
        # self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1, connect_f=connect_f)
        self.layer5_d = self._make_single_layer(Bottleneck, planes * 2, planes * 2, no_relu=True, connect_f=connect_f)
        
        
        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)           

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, layer.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1, connect_f=None):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(planes*block.expansion),
                # neuron.LIFNode()
            )
        
        # 无论如何添加conv1x1
        # else:
        #     # 等维度变换
        #     downsample = nn.Sequential(
        #         layer.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
        #         layer.BatchNorm2d(inplanes),
        #     )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, connect_f))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True, connect_f=connect_f))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False, connect_f=connect_f))
        # for i in range(1, blocks):
        #     layers.append(block(inplanes, planes, stride=1, connect_f=connect_f))

        return nn.Sequential(*layers)
    
    def _make_single_layer(self, block, inplanes, planes, stride=1, no_relu=False, connect_f=None):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(planes*block.expansion),
                # neuron.LIFNode()
            )
        
        # 无论如何添加conv1x1
        # else:
        #     # 等维度变换
        #     downsample = nn.Sequential(
        #         layer.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
        #         layer.BatchNorm2d(inplanes),
        #     )
            
        single_layer = block(inplanes, planes, stride, downsample, no_relu, connect_f)
        
        return single_layer

    def forward(self, x):
        
        # x:[b,c(t),h,w]
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        
        x = self.conv1(x)
        x = self.layer1(x)
        
        x = self.lif2(self.layer2(self.lif1(x)))
        
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)
        x = self.lif3(self.layer3(x))
        
        x_ = self.pag3(x_, self.compression3(x))
        # input_size = x_.size()
        # x_ += BasicInterpolate(size=[input_size[-2], input_size[-1]],
        #                        mode='bilinear', align_corners=False)(self.compression3(x))
        
        x_d = x_d + BasicInterpolate(size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)(self.diff3(x))
        if self.augment:
            temp_p = x_
        
        x = self.lif4(self.layer4(x))
        x_ = self.layer4_(self.lif5(x_))
        x_d = self.layer4_d(self.lif6(x_d))
        
        x_ = self.pag4(x_, self.compression4(x))
        # input_size = x_.size()
        # x_ += BasicInterpolate(size=[input_size[-2], input_size[-1]],
        #                        mode='bilinear', align_corners=False)(self.compression4(x))
        
        x_d = x_d + BasicInterpolate(size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)(self.diff4(x))

        if self.augment:
            temp_d = x_d
            
        x_ = self.layer5_(self.lif7(x_))
    
        x_d = self.layer5_d(self.lif8(x_d))
       
        x = BasicInterpolate(size=[height_output, width_output],mode='bilinear', 
                             align_corners=algc)(self.spp(self.layer5(x)))
       
        x_ = self.final_layer(self.dfm(x_, x, x_d))
        # x_ = self.final_layer(x_+ x+ x_d)
        
        

        if self.augment: 
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            # 处理时间维度
            # [t,b,c,h,w]->[b,c,h,w](average on t demension)
            x_extra_p = x_extra_p.mean(dim=0)
            x_extra_d = x_extra_d.mean(dim=0)
            x_ = x_.mean(dim=0)
            return [x_extra_p, x_, x_extra_d]
        else:
            x_ = x_.mean(dim=0)   
            return x_     

def get_seg_model(cfg, imgnet_pretrained):
    
    if cfg.DATASET.event_representation == 'voxel_grid':
        input_channel = 1
    elif cfg.DATASET.event_representation == 'voxel_grid_2':
        input_channel = cfg.DATASET.nr_temporal_bins
    elif cfg.DATASET.event_representation in ['SBT_1', 'SBE_1']:
        input_channel = 2
    elif cfg.DATASET.event_representation in ['SBT_2', 'SBE_2']:
        input_channel = cfg.DATASET.nr_temporal_bins * 2
    elif cfg.DATASET.event_representation == 'ev_segnet':
        input_channel = 6
    elif cfg.DATASET.event_representation == 'MDOE':
        input_channel = 4
    else:
        input_channel = 1
    
    
    model = SpikeBRGNet(input_channel, m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, 
                    ppm_planes=96, head_planes=128, augment=True)
    functional.set_step_mode(model, step_mode='m')
    
    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict'] 
        # pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        logging.info('Train from scratch!')
        # pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        # if 'state_dict' in pretrained_dict:
        #     pretrained_dict = pretrained_dict['state_dict']
        # model_dict = model.state_dict()
        # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        # msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        # logging.info('Attention!!!')
        # logging.info(msg)
        # logging.info('Over!!!')
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict, strict = False)
    return model

def get_pred_model(cfg, name, num_classes):
    
    if cfg.DATASET.event_representation == 'voxel_grid':
        input_channel = 1
    elif cfg.DATASET.event_representation == 'voxel_grid_2':
        input_channel = cfg.DATASET.nr_temporal_bins
    elif cfg.DATASET.event_representation in ['SBT_1', 'SBE_1']:
        input_channel = 2
    elif cfg.DATASET.event_representation in ['SBT_2', 'SBE_2']:
        input_channel = cfg.DATASET.nr_temporal_bins * 2
    elif cfg.DATASET.event_representation == 'ev_segnet':
        input_channel = 6
    elif cfg.DATASET.event_representation == 'MDOE':
        input_channel = 4
    else:
        input_channel = 1
    

    model = SpikeBRGNet(input_channel, m=2, n=3, num_classes=num_classes, planes=32, 
                    ppm_planes=96, head_planes=128, augment=False)
    functional.set_step_mode(model, step_mode='m')
    
    return model

if __name__ == '__main__':
    
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    # 测试模型预测时的推理速度（FPS）
    model = get_pred_model(name='SpikeBRGNet_s', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            # ？？为什么乘6
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        # latency:ms
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    # FPS:Frame(iteration) per second(s)
    FPS = 1000 / latency
    print(FPS)