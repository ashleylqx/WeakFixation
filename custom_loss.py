import os
import numpy as np
import math
import torch
import torch.nn.functional as F
import torchvision

import scipy.misc
from config import *

# def kl_div(pred_maps, target_maps):
#     pass
# torch.nn.KLDivLoss

def cc(pred_maps, target_maps):
    pass

def nss(pred_maps, target_fixs):
    pass

def sim(pred_maps, target_maps):
    pass

def loss_fn(pred_maps, target_maps, target_fixs):
    pass

class InfoEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(InfoEntropyLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        input_normalized = self.softmax(input)

        self_info = -torch.log(input_normalized)
        entropy = torch.mul(input_normalized, self_info).sum(dim=-1)

        return entropy.mean()

class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x = x/x.max()
        # b = F.log_softmax(x, dim=-1)
        b = F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)
        b = -1.0 * b.sum(dim=-1)
        return b.mean()


class HLoss_th(torch.nn.Module):
    def __init__(self):
        super(HLoss_th, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x = x/x.max(-1)
        # assert x.sum()!=0
        x = torch.softmax(x, dim=-1)
        ind = torch.ones_like(x, requires_grad=True)
        ind = torch.div(ind , ind.sum(-1, keepdim=True))
        # print('x', x.size())

        pos_mask = torch.nn.ReLU()((x - x.mean(-1, keepdim=True).expand_as(x)))  # /sal_map_hot.var()
        pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (x.var(-1, keepdim=True)+1e-8)) * 1e3)

        # saliency value of region
        pos_x = torch.mul(pos_mask, x).sum(-1, keepdim=True)
        neg_x = torch.mul((1.-pos_mask), x).sum(-1, keepdim=True)
        p = torch.cat([pos_x, neg_x], dim=-1)

        # pixel percentage of region
        pos_ind = torch.mul(pos_mask, ind).sum(-1, keepdim=True)
        neg_ind = torch.mul((1. - pos_mask), ind).sum(-1, keepdim=True)
        ratio_w = torch.cat([pos_ind, neg_ind], dim=-1)

        b = F.softmax(p, dim=-1) * F.log_softmax(ratio_w, dim=-1) # recent best
        # b = F.softmax(ratio_w, dim=-1) * F.log_softmax(p, dim=-1) # _2
        # b = F.softmax(p, dim=-1) * F.log_softmax(p, dim=-1) # _3
        # b = F.softmax(ratio_w, dim=-1) * F.log_softmax(ratio_w, dim=-1) # _4
        # b = F.softmax(p, dim=-1) * F.log_softmax(p, dim=-1) + \
        #     F.softmax(ratio_w, dim=-1) * F.log_softmax(ratio_w, dim=-1) #_5
        b = -1.0 * b.sum(dim=-1)
        return b.mean()
# *** default ***
class HLoss_th_2(torch.nn.Module):
    def __init__(self):
        super(HLoss_th_2, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x = x/x.max(-1)
        # assert x.sum()!=0
        x = torch.softmax(x, dim=-1)
        ind = torch.ones_like(x, requires_grad=True)
        ind = torch.div(ind , ind.sum(-1, keepdim=True))
        # print('x', x.size())

        pos_mask = torch.nn.ReLU()((x - x.mean(-1, keepdim=True).expand_as(x)))  # /sal_map_hot.var()
        pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (x.var(-1, keepdim=True)+1e-8)) * 1e3)

        # saliency value of region
        pos_x = torch.mul(pos_mask, x).sum(-1, keepdim=True)
        neg_x = torch.mul((1.-pos_mask), x).sum(-1, keepdim=True)
        p = torch.cat([pos_x, neg_x], dim=-1)

        # pixel percentage of region
        pos_ind = torch.mul(pos_mask, ind).sum(-1, keepdim=True)
        neg_ind = torch.mul((1. - pos_mask), ind).sum(-1, keepdim=True)
        ratio_w = torch.cat([pos_ind, neg_ind], dim=-1)

        # b = F.softmax(p, dim=-1) * F.log_softmax(ratio_w, dim=-1) # recent best
        b = F.softmax(ratio_w, dim=-1) * F.log_softmax(p, dim=-1) # _2
        # b = F.softmax(p, dim=-1) * F.log_softmax(p, dim=-1) # _3
        # b = F.softmax(ratio_w, dim=-1) * F.log_softmax(ratio_w, dim=-1) # _4
        # b = F.softmax(p, dim=-1) * F.log_softmax(p, dim=-1) + \
        #     F.softmax(ratio_w, dim=-1) * F.log_softmax(ratio_w, dim=-1) #_5
        b = -1.0 * b.sum(dim=-1)
        return b.mean()

class HLoss_th_210423(torch.nn.Module):
    def __init__(self):
        super(HLoss_th_210423, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x = x/x.max(-1)
        # assert x.sum()!=0
        x = torch.softmax(x, dim=-1)
        ind = torch.ones_like(x, requires_grad=True)
        ind = torch.div(ind , ind.sum(-1, keepdim=True))
        # print('x', x.size())

        # pos_mask = torch.nn.ReLU()((x - x.mean(-1, keepdim=True).expand_as(x)))  # /sal_map_hot.var()
        # pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (x.var(-1, keepdim=True)+1e-8)) * 1e3)
        pos_mask = x - x.mean(-1, keepdim=True).expand_as(x)  # /sal_map_hot.var()
        pos_mask = pos_mask > 0


        # saliency value of region
        pos_x = torch.masked_select(x, pos_mask).sum(-1, keepdim=True)
        neg_x = torch.masked_select(x, (1.-pos_mask)).sum(-1, keepdim=True)
        p = torch.cat([pos_x, neg_x], dim=-1)

        # pixel percentage of region
        pos_ind = torch.masked_select(ind, pos_mask).sum(-1, keepdim=True)
        neg_ind = torch.masked_select(ind, (1.-pos_mask)).sum(-1, keepdim=True)
        ratio_w = torch.cat([pos_ind, neg_ind], dim=-1)

        # b = F.softmax(p, dim=-1) * F.log_softmax(ratio_w, dim=-1) # recent best
        b = F.softmax(ratio_w, dim=-1) * F.log_softmax(p, dim=-1) # _2
        # b = F.softmax(p, dim=-1) * F.log_softmax(p, dim=-1) # _3
        # b = F.softmax(ratio_w, dim=-1) * F.log_softmax(ratio_w, dim=-1) # _4
        # b = F.softmax(p, dim=-1) * F.log_softmax(p, dim=-1) + \
        #     F.softmax(ratio_w, dim=-1) * F.log_softmax(ratio_w, dim=-1) #_5
        b = -1.0 * b.sum(dim=-1)
        return b.mean()

class HLoss_th_3(torch.nn.Module):
    def __init__(self):
        super(HLoss_th_3, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x = x/x.max(-1)
        # assert x.sum()!=0
        x = torch.softmax(x, dim=-1)
        ind = torch.ones_like(x, requires_grad=True)
        ind = torch.div(ind , ind.sum(-1, keepdim=True))
        # print('x', x.size())

        pos_mask = torch.nn.ReLU()((x - x.mean(-1, keepdim=True).expand_as(x)))  # /sal_map_hot.var()
        pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (x.var(-1, keepdim=True)+1e-8)) * 1e3)

        # saliency value of region
        pos_x = torch.mul(pos_mask, x).sum(-1, keepdim=True)
        neg_x = torch.mul((1.-pos_mask), x).sum(-1, keepdim=True)
        p = torch.cat([pos_x, neg_x], dim=-1)

        # pixel percentage of region
        # pos_ind = torch.mul(pos_mask, ind).sum(-1, keepdim=True)
        # neg_ind = torch.mul((1. - pos_mask), ind).sum(-1, keepdim=True)
        # ratio_w = torch.cat([pos_ind, neg_ind], dim=-1)

        # b = F.softmax(p, dim=-1) * F.log_softmax(ratio_w, dim=-1) # recent best
        # b = F.softmax(ratio_w, dim=-1) * F.log_softmax(p, dim=-1) # _2
        b = F.softmax(p, dim=-1) * F.log_softmax(p, dim=-1) # _3
        # b = F.softmax(ratio_w, dim=-1) * F.log_softmax(ratio_w, dim=-1) # _4
        # b = F.softmax(p, dim=-1) * F.log_softmax(p, dim=-1) + \
        #     F.softmax(ratio_w, dim=-1) * F.log_softmax(ratio_w, dim=-1) #_5
        b = -1.0 * b.sum(dim=-1)
        return b.mean()

# modified from https://github.com/xiaoboCASIA/SV-X-Softmax/blob/master/loss.py
def loss_HM(pred, label, save_rate=0.9, gamma=2.0):
    # elif loss_type == 'FocalLoss':
    #     assert (gamma >= 0)
    #     input = F.cross_entropy(pred, label, reduce=False)
    #     pt = torch.exp(-input)
    #     loss = (1 - pt) ** gamma * input
    #     loss_final = loss.mean()
    # loss_type == 'HardMining':
    batch_size = pred.shape[0]
    # loss = F.cross_entropy(pred, label, reduce=False)
    # ind_sorted = torch.argsort(-loss) # from big to small
    # num_saved = int(save_rate * batch_size)
    # ind_update = ind_sorted[:num_saved]
    # loss_final = torch.sum(F.cross_entropy(pred[ind_update], label[ind_update]))

    loss = F.binary_cross_entropy_with_logits(pred, label, reduce=False)
    loss = torch.sum(loss, dim=-1)
    ind_sorted = torch.argsort(-loss) # from big to small
    num_saved = int(save_rate * batch_size)
    ind_update = ind_sorted[:num_saved]
    loss_final = torch.sum(F.binary_cross_entropy_with_logits(pred[ind_update,:], label[ind_update,:]))

    return loss_final


class Compress_percent(torch.nn.Module):
    def __init__(self):
        super(Compress_percent, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # xm = x.mean(-1)
        # mask = (x>xm.unsqueeze(1)).float()
        ind = torch.ones_like(x, requires_grad=True)
        # ratios = (torch.mul(mask, ind)).sum(-1) / ind.sum(-1)

        pos_mask = torch.nn.ReLU()((x - x.mean() - 1e-8))  # /sal_map_hot.var()
        pos_mask = torch.nn.Softsign()(pos_mask / (x.var()+1e-8) *1e3)

        pos_x = torch.mul(pos_mask, ind).sum(-1, keepdim=True)
        neg_x = torch.mul((1. - pos_mask), ind).sum(-1, keepdim=True)

        ratios = pos_x/neg_x
        return ratios.mean()

# class Compress_percent(torch.nn.Module):
#     def __init__(self):
#         super(Compress_percent, self).__init__()
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)
#         xm = x.mean(-1)
#         xc = x.size(-1)
#         ratios = (x>xm.unsqueeze(1)).sum(-1).float()/xc
#         return ratios.mean()
#


#============ vae loss ===================
# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(logit, y, mu, std, n_classes, kl_weight=1e-3):
    # # BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, n_classes), reduction='sum')
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, n_classes), reduction='sum').div(math.log(2))
    #
    # # see Appendix B from VAE paper:
    # # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # # https://arxiv.org/abs/1312.6114
    # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -kl_weight * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).div(math.log(2))

    # return BCE + KLD

    # class_loss = F.cross_entropy(logit, y).div(math.log(2))
    class_loss = F.binary_cross_entropy(logit, y).div(math.log(2))
    info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
    total_loss = class_loss + kl_weight * info_loss

    return total_loss



if __name__ == '__main__':
    # torch.set_grad_enabled(True)
    # a = torch.rand(2, 8, 8)
    # b = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)(a)
    # el_loss = InfoEntropyLoss()
    # losses = el_loss(b)
    # losses.backward()
    # print(losses.item())

    # criterion = HLoss_th()# backpropgate
    criterion = HLoss_th_210423()# backpropgate
    # criterion = torch.nn.L1Loss()# backpropgate
    # criterion = HLoss() # backpropgate
    # criterion = Compress_percent() # do not propagate

    x = torch.rand(2, 10, 10)
    x2 = torch.ones(2, 10, 10)

    w_m = torch.rand(2, 10, 10, requires_grad=True)
    mask = torch.bmm(x, w_m)
    pos_mask = torch.nn.ReLU()((mask - mask.view(2, -1).mean(-1, keepdim=True).unsqueeze(-1).expand_as(mask)))  # /sal_map_hot.var()
    pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (mask.view(2, -1).var(-1, keepdim=True).unsqueeze(-1)+1e-8))* 1e3)
    # pos_mask = torch.ge(mask, mask.view(2, -1).mean(-1).unsqueeze(-1).unsqueeze(-1).expand_as(mask))

    x = torch.mul(x, pos_mask.float())
    # x3 = torch.mul(x2, pos_mask)


    # pos_mask = mask - mask.view(2, -1).mean(-1, keepdim=True).unsqueeze(-1).expand_as(mask)
    # pos_mask = pos_mask > 0
    # x = torch.masked_select(x, pos_mask)


    # x = torch.randn(2, 10, 10)
    w = torch.ones(2, 10, 3, requires_grad=True) #requires_grad=True
    print(w.requires_grad)
    w_pre = torch.ones(2, 10, 3, requires_grad=True)
    output = torch.bmm(x, w)
    # output = torch.bmm(x3, w)

    optimizer = torch.optim.Adam([w, w_m], lr=1e-4)

    # x = torch.ones(2, 3, 224, 224)
    # model = torchvision.models.vgg16_bn(pretrained=False)
    # output = model.features(x)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    # loss = criterion(output)
    loss = criterion(output)
    # loss = criterion(output, torch.zeros_like(output))

    output = output.view(output.size(0), -1)
    xm = output.mean(-1)
    mask = (output>xm.unsqueeze(1)).float()

    # loss = criterion(output, mask, torch.ones_like(output, requires_grad=True))
    loss.backward()
    optimizer.step()
    print(loss.item())
    if (w==w_pre).all():
        print('weight not change')


    print('w grad', w.grad.sum())
    print('w_m grad', w_m.grad.sum())

    # print('x grad', x.grad.sum())
    # print('x2 grad', x2.grad.sum())


    # print(w.grad)
    # #
    # # criterion2 = InfoEntropyLoss()
    # # loss2 = criterion2(output)
    # # print(loss2.item()) # loss1.item()==loss.item()
    #
    # files = ['i169636965', 'i176365896', 'i2170813265', 'i2247811037']
    # # gt_postfix = '_fixMap.jpg'
    # # gt_path = PATH_MIT1003 + 'ALLFIXATIONMAPS'
    # gt_postfix = '.png'
    # gt_path = base_path + 'WF/Preds/best_sa_wk5_epoch00'
    #
    # for f in files:
    #     map = scipy.misc.imread(os.path.join(gt_path, f+gt_postfix))
    #     # map = map/255.
    #     # map = np.ones_like(map).astype('float')
    #     # map = np.zeros_like(map).astype('float')
    #     map = torch.rand(map.shape[0], map.shape[1])
    #     print(criterion(map.unsqueeze(0)).item())
    #     # print(criterion(torch.tensor(np.expand_dims(map, axis=0))).item())




