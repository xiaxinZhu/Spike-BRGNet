# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    # score：[x_extra_p,x_]->[6,19,1024,1024],[6,19,1024,1024]
    # target: labels->[6,1024,1024]
    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        # return:预测损失+辅助语义损失
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")

        


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        # nn.CrossEntropyLoss:computes the cross entropy loss between input logits and target.
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        # target:[b,h,w]
        # pred/score:[b,c,h,w] 
        pred = F.softmax(score, dim=1) 
        # [b,c,h,w]->b*c*h*w (展平为一位向量)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        # mask:b*c*h*w，掩码
        mask = target.contiguous().view(-1) != self.ignore_label 

        # tmp_target:[b,h,w]
        tmp_target = target.clone()
        # 将label值为无效值255的label赋值为0
        tmp_target[tmp_target == self.ignore_label] = 0
        # 使用 gather() 方法根据 tmp_target 张量的值，在 pred 张量的第一维度上
        # 进行索引，获取对应位置的预测概率。这将得到一个形状为 [6, 1, 1024, 1024]
        # 的张量，并将其重新赋值给 pred
        # tmp_target.unsqueeze:[6,1,1024,1024]
        # pred:[6,19,1024,1024]->[6,1,1024,1024]
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        # 将 pred 张量展平为一维张量，并根据 mask 掩码选取有效位置的预测概率。
        # 然后，使用 .contiguous().sort()对选取的预测概率进行排序，并返回排序后的
        # 结果 pred 和对应的索引 ind
        # pred：一维张量
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        # config: thres=OHEMTHRES->0.9, min_kept=OHEMKEEP->131072
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        # 使用mask掩码和ind索引，选取有效位置的像素损失
        # pixel_losses：一维张量
        pixel_losses = pixel_losses[mask][ind]
        # 根据阈值threshold进一步筛选像素损失，只保留对应位置的预测概率小于阈值的损失
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean() # len=1

    def forward(self, score, target):
        # score=[x_extra_p,x_],target
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]
        # BALANCE_WEIGHTS=[0.4, 1.0], SB_WEIGHTS=1.0, OHEMTHRES=0.9, OHEMKEEP=131072
        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    # [b,c,h,w]->[b,h,w,c]->[1,b*h*w*c] (c=1)
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    # [b,h,w]->[1,b*h*w]
    target_t = target.view(1, -1) 

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    # input(Tensor) – Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
    # target(Tensor) – Tensor of the same shape as input with values between 0 and 1
    # weight(Tensor) – a manual rescaling weight if provided it’s repeated to match input tensor shape
    # reduction – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
    #             'mean': the sum of the output will be divided by the number of elements in the output, 
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
    # len(loss)=1
    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
if __name__ == '__main__':
    a = torch.zeros(2,64,64)
    a[:,5,:] = 1
    pre = torch.randn(2,1,16,16)
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))

        
        
        


