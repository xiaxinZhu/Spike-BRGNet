# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from PIL import Image

from spikingjelly.activation_based import functional
from torchvision.utils import save_image

def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, scheduler_lr, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_ba_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        # images:[B,C,H,W] [12, 3, 1024, 1024]
        # labels:[B,H,W] [12, 1024, 1024]
        # bd_gts:[B,H,W] [12, 1024, 1024]
        # images, labels, bd_gts, _, _ = batch
        events, labels, bd_gts = batch
        events = events.cuda(config.GPU_DEVICE)
        labels = labels.long().cuda(config.GPU_DEVICE)
        bd_gts = bd_gts.float().cuda(config.GPU_DEVICE)
        
        # losses=[1,b,h,w], acc=1, loss_list=[loss_s, loss_b]
        # 合并并行化之后：
        # losses: [2, 6, 1024, 1024]
        # acc: [2]
        # loss_list=[loss_s, loss_b]
        
        losses, _, acc, loss_list = model(events, labels, bd_gts)
        functional.reset_net(model)
        
        loss = losses.mean()
        acc  = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        # (Semantic loss) loss_list[0]:loss_s=[b*2,h,w]:[12,1024,1024]
        # (BCE loss) loss_list[1]:loss_b=[2]
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_ba_loss.update(loss_list[1].mean().item())

        lr = scheduler_lr.get_last_lr()

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BA loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_ba_loss.average())
            logging.info(msg)

    scheduler_lr.step()
    
    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            # image, label, bd_gts, _, _ = batch
            event, label, bd_gts = batch
            size = label.size()
            event = event.cuda(config.GPU_DEVICE)
            label = label.long().cuda(config.GPU_DEVICE)
            bd_gts = bd_gts.float().cuda(config.GPU_DEVICE)

            losses, pred, _, _ = model(event, label, bd_gts)
            # pred = model(image, label, bd_gts)
            functional.reset_net(model)
            
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    
    writer = writer_dict['writer']
    # global_steps = writer_dict['valid_global_steps']
    global_steps = writer_dict['train_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    # writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array
    # return mean_IoU, IoU_array


def testval(config, testloader, model, sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            event, label, _ = batch
            # event, image, label, _, _ = batch
            size = label.size()
            event = event.cuda(config.GPU_DEVICE)
            label = label.long().cuda(config.GPU_DEVICE)
            
            pred = model(event)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
                    
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, testloader, model, sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            # event, label, _ = batch
            event, image, label, _, _ = batch
            size = label.size()
            event = event.cuda(config.GPU_DEVICE)
            
            pred = model(event)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            label = label.squeeze(0).cpu().numpy()
            
            if sv_pred:
                if config.DATASET.DATASET == 'DDD17_events':
                    # DDD17:['flat','background','object','vegetation','human','vehicle']
                    color_map = [(128, 64,128), #紫
                                (70 , 70, 70), #灰
                                (220,220,  0), #黄
                                (107,142, 35), #绿
                                (220, 20, 60), #红
                                (0  ,  0,142)] #蓝
                elif config.DATASET.DATASET == 'DESC_events':
                    # DSEC:['background','building','fence','person','pole','road','sidewalk','vegetation','car','wall','traffic sign']
                    color_map = [(0,  0,  0),
                                (70 ,70, 70),
                                (190,153,153),
                                (220, 20,60),
                                (153,153,153),
                                (128, 64,128),
                                (244, 35,232),
                                (107,142, 35),
                                (0,  0,  142),
                                (102,102,156),
                                (220,220,  0)]
                    
                sv_predict = np.zeros((size[-2], size[-1], 3), dtype=np.uint8)
                sv_label = np.zeros((size[-2], size[-1], 3), dtype=np.uint8)
                
                sv_path = os.path.join(sv_dir,'test_results')
                sv_path_label = os.path.join(sv_path,'label')
                sv_path_pred = os.path.join(sv_path,'predict')
                sv_path_image = os.path.join(sv_path,'image')
                
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                if not os.path.exists(sv_path_label):
                    os.mkdir(sv_path_label)
                if not os.path.exists(sv_path_pred):
                    os.mkdir(sv_path_pred)
                if not os.path.exists(sv_path_image):
                    os.mkdir(sv_path_image)
                
                if config.DATASET.split in ['valid', 'train'] :
                    # 验证全部valid数据集
                    for idx in range(pred.shape[0]):
                        for i, color in enumerate(color_map):
                            for j in range(3):
                                sv_predict[:,:,j][pred[idx]==i] = color_map[i][j]
                                sv_label[:,:,j][label[idx]==i] = color_map[i][j]
                        
                        sv_predict_event = Image.fromarray(sv_predict)
                        sv_label_event = Image.fromarray(sv_label)
                        sv_predict_event.save(sv_path_label+"/predict{}.png".format(index * pred.shape[0] + idx))
                        sv_label_event.save(sv_path_pred+"/label{}.png".format(index * pred.shape[0] + idx))
                        save_image(image[idx], sv_path_image+'/image{}.png'.format(index * pred.shape[0] + idx))
                        
                elif config.DATASET.split == 'test':
                    for i, color in enumerate(color_map):
                        for j in range(3):
                            sv_predict[:,:,j][pred==i] = color_map[i][j]
                            sv_label[:,:,j][label==i] = color_map[i][j]
                        
                    sv_predict_event = Image.fromarray(sv_predict)
                    sv_label_event = Image.fromarray(sv_label)
                    sv_predict_event.save(sv_path_pred+"/predict{}.png".format(index))
                    sv_label_event.save(sv_path_label+"/label{}.png".format(index))
                    save_image(image, sv_path_image+'/image{}.png'.format(index))
                    
