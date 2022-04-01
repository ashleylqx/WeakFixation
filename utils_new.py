import os
import cv2
import numpy as np
import shutil
import scipy.misc
import pdb
import torch

from config_new import *

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map

def postprocess_prediction(prediction, size=None, printinfo=False):
    """
    Postprocess saliency maps by resizing and applying gaussian blurringself.
    args:
        prediction: numpy array with saliency postprocess_prediction
        size: original (H,W) of the image
    returns:
        numpy array with saliency map normalized 0-255 (int8)
    """
    prediction = prediction - np.min(prediction)
    # pdb.set_trace()
    # prediction = prediction - np.mean(prediction)
    # prediction[prediction<0] = 0
    if printinfo:
        print('max %.4f min %.4f'%(np.max(prediction), np.min(prediction)))

    if np.max(prediction) != 0:
        saliency_map = (prediction/np.max(prediction) * 255).astype(np.uint8)
    else:
        saliency_map = prediction.astype(np.uint8)

    if size is None:
        size = SALGAN_RESIZE

    # resize back to original size
    saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)
    saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

    # saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    # saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)

    # clip again
    # saliency_map = np.clip(saliency_map, 0, 255)
    if np.max(saliency_map)!=0:
        saliency_map = saliency_map.astype('float') / np.max(saliency_map) * 255.
    else:
        print('Zero saliency map.')

    return saliency_map

def postprocess_prediction_thm(prediction, size=None):
    """
    Postprocess saliency maps by resizing and applying gaussian blurringself.
    args:
        prediction: numpy array with saliency postprocess_prediction
        size: original (H,W) of the image
    returns:
        numpy array with saliency map normalized 0-255 (int8)
    """
    # prediction = prediction - np.min(prediction)

    prediction = prediction - np.mean(prediction)
    prediction[prediction<0] = 0

    print('max %.4f min %.4f'%(np.max(prediction), np.min(prediction)))
    if np.max(prediction) != 0:
        saliency_map = (prediction/np.max(prediction) * 255).astype(np.uint8)
    else:
        saliency_map = prediction.astype(np.uint8)

    if size is None:
        size = SALGAN_RESIZE

    # resize back to original size
    saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)
    saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

    # saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    # saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)

    # clip again
    # saliency_map = np.clip(saliency_map, 0, 255)
    if np.max(saliency_map)!=0:
        saliency_map = saliency_map.astype('float') / np.max(saliency_map) * 255.
    else:
        print('Zero saliency map.')

    return saliency_map

def postprocess_prediction_salgan(prediction, size=None):
    """
    Postprocess saliency maps by resizing and applying gaussian blurringself.
    args:
        prediction: numpy array with saliency postprocess_prediction
        size: original (H,W) of the image
    returns:
        numpy array with saliency map normalized 0-255 (int8)
    """
    # prediction = prediction - np.min(prediction) # makes no difference

    print('max %.4f min %.4f'%(np.max(prediction), np.min(prediction)))
    saliency_map = (prediction * 255).astype(np.uint8)

    blur_size = 5
    # resize back to original size
    saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    # blur
    saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)

    return saliency_map

# preserve aspect ratio
def postprocess_hd_prediction(prediction, size=None):
    """
    Postprocess saliency maps by resizing and applying gaussian blurringself.
    args:
        prediction: numpy array with saliency postprocess_prediction
        size: original (H,W) of the image
    returns:
        numpy array with saliency map normalized 0-255 (int8)
    """
    print('max %.4f min %.4f'%(np.max(prediction), np.min(prediction)))
    saliency_map = (prediction * 255).astype(np.uint8)

    if size is None:
        size = SALGAN_RESIZE

    # resize back to original size
    # saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)
    # saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

    saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    # saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0) # hd_map has its own gaussian blur process

    # clip again
    # saliency_map = np.clip(saliency_map, 0, 255)
    if np.max(saliency_map)!=0:
        saliency_map = saliency_map.astype('float') / np.max(saliency_map) * 255.
    else:
        print('Zero saliency map.')

    return saliency_map

# preserve aspect ratio
def postprocess_prediction_my(pred, shape_r, shape_c):
    # pred = sigmoid(pred)
    pred = pred.astype('float')
    predictions_shape = pred.shape
    rows_rate = shape_r*1.0 / predictions_shape[0]
    cols_rate = shape_c*1.0 / predictions_shape[1]

    pred = pred / np.max(pred) * 255.

    if rows_rate > cols_rate:
        new_cols = int(predictions_shape[1] * float(shape_r) // predictions_shape[0])
        # pred = cv2.resize(pred, (new_cols, shape_r))
        pred = scipy.misc.imresize(pred, (shape_r, new_cols))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = int(predictions_shape[0] * float(shape_c) // predictions_shape[1])
        # pred = cv2.resize(pred, (shape_c, new_rows))
        pred = scipy.misc.imresize(pred, (new_rows, shape_c))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r),:]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma=7)
    img = img.astype('float') / np.max(img) * 255

    return img


# Method to save trained model
def save_model(net, optim, epoch, p_out, eval_loss, name_model=None, results=None, is_best=False, best_name='best.pt'):

    if name_model is None:
        name_model = epoch

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    # opt_state_dict = optim.state_dict()
    # for key in opt_state_dict.keys():
    #     opt_state_dict[key] = opt_state_dict[key].cpu()

    model_dict = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optim.state_dict(),
            'eval_loss': eval_loss}

    for k in results.keys():
        model_dict[k] = results[k].mean

    # filepath = os.path.join(p_out,'{}.pt'.format(name_model))
    filepath = os.path.join(p_out, name_model)
    torch.save(model_dict, filepath)

    if is_best:
        # shutil.copyfile(filepath, os.path.join(p_out, '{}_best.pt'.format(name_model)))
        shutil.copyfile(filepath, os.path.join(p_out, best_name))

    # torch.save({
    #         'epoch': epoch,
    #         'state_dict': state_dict,
    #         'optimizer': optim,
    #         'eval_loss': eval_loss},
    #         os.path.join(p_out,'{}_epoch{:02d}.pt'.format(name_model, epoch)))



def get_lr_optimizer( optimizer ):
    """ Get learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        yield param_group['lr']

# -------from dim.py ---------------
def sample_locations(enc, n_samples):
    '''Randomly samples locations from localized features.
    Used for saving memory.
    Args:
        enc: Features.
        n_samples: Number of samples to draw.
    Returns:
        torch.Tensor
    '''
    n_locs = enc.size(2)
    batch_size = enc.size(0)
    weights = torch.tensor([1. / n_locs] * n_locs, dtype=torch.float)
    idx = torch.multinomial(weights, n_samples * batch_size, replacement=True) \
        .view(batch_size, n_samples)
    enc = enc.transpose(1, 2)
    adx = torch.arange(0, batch_size).long()
    enc = enc[adx[:, None], idx].transpose(1, 2)
    return enc

def sample_locations_my(enc, n_samples):
    '''Randomly samples locations from localized features.
    Used for saving memory.
    Args:
        enc: Features. (N, C, H, W)
        n_samples: Number of samples to draw.
    Returns:
        torch.Tensor
    '''
    enc = enc.view(enc.size(0), enc.size(1), -1)
    n_locs = enc.size(2)
    batch_size = enc.size(0)
    weights = torch.tensor([1. / n_locs] * n_locs, dtype=torch.float)
    idx = torch.multinomial(weights, n_samples * batch_size, replacement=True) \
        .view(batch_size, n_samples)
    enc = enc.transpose(1, 2)
    adx = torch.arange(0, batch_size).long()
    enc = enc[adx[:, None], idx].transpose(1, 2)
    return enc