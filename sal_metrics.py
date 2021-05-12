# https://github.com/rdroste/unisal/blob/192abd2affbb1824895118a914806663bb897aa1/unisal/train.py#L710
# https://github.com/rdroste/unisal/blob/192abd2affbb1824895118a914806663bb897aa1/unisal/train.py#L607
# SIM, AUC-J, s-AUC, https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py
# KLD, NSS, CC, https://github.com/rdroste/unisal/blob/master/unisal/utils.py

import pdb
import torch
import torch.nn.functional as F
import numpy as np


# numpy array ----------------
def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    # pdb.set_trace()
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)))
    return norm_s_map


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = normalize_map(s_map)
    assert np.max(gt) == 1.0,\
        'Ground truth not discretized properly max value > 1.0'
    assert np.max(s_map) == 1.0,\
        'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map[gt > 0].tolist()

    num_fixations = len(thresholds)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map,
        # keep only those pixels with values above threshold
        temp = s_map >= thresh
        num_overlap = np.sum(np.logical_and(temp, gt))
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        # fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[:2]) - num_fixations)
        fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[-2:]) - num_fixations)

        area.append((round(tp, 4) ,round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list, fp_list = list(zip(*area))
    # pdb.set_trace()
    return np.trapz(np.array(tp_list), np.array(fp_list))

# pending ... need to sample other maps
def auc_shuff_acl(s_map, gt, other_map, n_splits=100, stepsize=0.1):

    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()
    # pdb.set_trace()
    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((Nfixations_oth, n_splits), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy())
        # sal map values at random fixation locations of other random images
        randfix[:, i] = S[randind[:Nfixations_oth]]

    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    for s in range(n_splits):

        curfix = randfix[:, s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(np.array(tp), np.array(fp))

    return np.mean(auc)


def similarity(s_map, gt):
    # pdb.set_trace()
    shape = gt.shape
    new_shape = (-1, shape[-1] * shape[-2])
    s_map = s_map.reshape(new_shape)
    gt = gt.reshape(new_shape)

    norm_s_map = (s_map - s_map.min(1)) / ((s_map.max(1) - s_map.min(1))+1e-8)
    norm_s_map = norm_s_map / (norm_s_map.sum()+1e-8)
    norm_gt = (gt - gt.min(1)) / ((gt.max(1) - gt.min(1))+1e-8)
    norm_gt = norm_gt / (norm_gt.sum()+1e-8)

    return np.sum(np.minimum(norm_s_map, norm_gt))

    # norm_s_map = (s_map - torch.min(s_map)) / ((torch.max(s_map) - torch.min(s_map)))
    # norm_s_map = norm_s_map / norm_s_map.sum()
    # norm_gt = (gt - torch.min(gt)) / ((torch.max(gt) - torch.min(gt)))
    # norm_gt = norm_gt / norm_gt.sum()
    #
    # return torch.sum(torch.minimum(norm_s_map, norm_gt))

# torch tensor ---------------
def nss(pred, fixations):
    size = fixations.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    fixations = fixations.reshape(new_size)
    # pdb.set_trace()
    pred_normed = (pred - pred.mean(-1, True)) / (pred.std(-1, keepdim=True)+1e-9)
    results = []
    for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                      torch.unbind(fixations, 0)): # delete dimention 0
    # for this_pred_normed, mask in zip(pred_normed ,fixations):
        if mask.sum() == 0:
            print("No fixations.")
            results.append(torch.ones([]).float().to(fixations.device))
            continue
        nss_ = torch.masked_select(this_pred_normed, mask)
        nss_ = nss_.mean(-1)
        results.append(nss_)
    results = torch.stack(results)
    # results = results.reshape(size[:2])
    results = results.reshape(size[:1])
    return results


def corr_coeff(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        r_num = torch.mean(xm * ym)
        r_den = torch.sqrt(
            torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
        r = r_num / r_den
        cc.append(r)

    cc = torch.stack(cc)
    # cc = cc.reshape(size[:2])
    cc = cc.reshape(size[:1])
    return cc  # 1 - torch.square(r)


def kld_loss(pred, target):
    loss = F.kl_div(pred, target, reduction='none')
    loss = loss.sum(-1).sum(-1).sum(-1)
    return loss