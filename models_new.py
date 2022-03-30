import numpy as np
import math
import pdb

import torch
import torchvision.models as models
import torch.nn.functional as F

from torchvision.models.resnet import Bottleneck

from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torch.autograd.function import once_differentiable

from collections import OrderedDict

from wildcat_pooling import WildcatPool2d, ClassWisePool

from config_new import *

def gen_grid_boxes(in_h, in_w, N=14):
    x = torch.linspace(0., in_w, N+1)
    y = torch.linspace(0., in_h, N+1)
    xv, yv = torch.meshgrid(x, y)

    grid_boxes = list()
    for i in range(N):
        for j in range(N):
            grid_boxes.append(torch.tensor([[[xv[i, j], yv[i, j], xv[i+1, j+1], yv[i+1, j+1]]]]))

    return torch.cat(grid_boxes, dim=1) # leave dim 0 for batch size repeat

class CenterBias_G(torch.nn.Module):
    def __init__(self, n=16, in_h=14, in_w=14): # let output size equal to input size
        super(CenterBias_G, self).__init__()
        # define some layer members
        self.n = n
        self.output_h = in_h
        self.output_w = in_w
        # # print(input_c, in_h, in_w, input_c*in_h*in_w)
        # self.fc1 = torch.nn.Linear(input_c, 200)
        # # self.fc1 = torch.nn.Linear(input_c*in_h*in_w, 200)
        # self.fc2 = torch.nn.Linear(200, 4*n)  # mu_x, mu_y, sigma_x, sigma_y
        self.params = torch.nn.Parameter(data=torch.rand(1, self.n*4), requires_grad=True)
        # torch.nn.init.constant_(self.params.weight, 0.5)
        # self.params

    def gen_gaussian_map(self, params):
        mu_x = params[:, :self.n]
        mu_y = params[:, self.n:2*self.n]
        sigma_x = params[:, 2*self.n:3*self.n]
        sigma_y = params[:, 3*self.n:]

        mu_x = mu_x.clamp(min=0.25, max=0.75)
        mu_y = mu_y.clamp(min=0.25, max=0.75)
        sigma_x = sigma_x.clamp(min=0.1, max=0.9)
        sigma_y = sigma_y.clamp(min=0.2, max=0.8)

        x_t = torch.mm(torch.ones(self.output_h, 1),
                       torch.linspace(0, 1.0, self.output_w).unsqueeze(0))
        y_t = torch.mm(torch.linspace(0, 1.0, self.output_h).unsqueeze(1),
                       torch.ones(1, self.output_w))

        x_t = x_t.unsqueeze(0).unsqueeze(0).repeat(mu_x.size(0), self.n, 1, 1).to(params.device)
        y_t = y_t.unsqueeze(0).unsqueeze(0).repeat(mu_x.size(0), self.n, 1, 1).to(params.device)
        # x_t = x_t.unsqueeze(0).unsqueeze(0).repeat(mu_x.size(0), self.n, 1, 1)
        # y_t = y_t.unsqueeze(0).unsqueeze(0).repeat(mu_x.size(0), self.n, 1, 1)

        gaussian = 1./(2*np.pi*torch.mul(sigma_x, sigma_y).unsqueeze(-1).unsqueeze(-1).expand_as(x_t))*\
            torch.exp(-(torch.div(torch.pow(x_t-mu_x.unsqueeze(-1).unsqueeze(-1).expand_as(x_t), 2),
                                  2*torch.pow(sigma_x.unsqueeze(-1).unsqueeze(-1),2)+1e-8) +
                        torch.div(torch.pow(y_t-mu_y.unsqueeze(-1).unsqueeze(-1).expand_as(y_t), 2),
                                  2*torch.pow(sigma_y.unsqueeze(-1).unsqueeze(-1),2)+1e-8)))
        max_gauss = torch.max(gaussian.view(gaussian.size(0), gaussian.size(1), -1), -1, keepdim=True).values
        gaussian = torch.div(gaussian, max_gauss.unsqueeze(-1))

        return gaussian

    def forward(self, x):
        # # define forward process
        # # x = F.relu(self.fc1(x.view(x.size(0), -1))) # ori_cb
        # x = F.adaptive_avg_pool2d(x, (1, 1)) # use GAP for _cb2
        # x = F.relu(self.fc1(x.view(x.size(0), -1)))
        # params = F.sigmoid(self.fc2(x))

        maps = self.gen_gaussian_map(self.params.to(x.device))
        return maps

class CenterBias_A(torch.nn.Module):
    def __init__(self, n=16, input_c=coco_num_classes, in_h=14, in_w=14): # let output size equal to input size
        super(CenterBias_A, self).__init__()
        # define some layer members
        self.n = n
        self.output_h = in_h
        self.output_w = in_w
        print(input_c, in_h, in_w, input_c)
        self.fc1 = torch.nn.Linear(input_c, input_c//2)
        # self.fc1 = torch.nn.Linear(input_c*in_h*in_w, 200)
        self.fc2 = torch.nn.Linear(input_c//2, 4*n)  #mu_x, mu_y, sigma_x, sigma_y

        # torch.nn.init.constant_(self.fc1.weight, 1./input_c)
        # torch.nn.init.constant_(self.fc2.weight, 2./input_c)

    def gen_gaussian_map(self, params):
        mu_x = params[:, :self.n]
        mu_y = params[:, self.n:2*self.n]
        sigma_x = params[:, 2*self.n:3*self.n]
        sigma_y = params[:, 3*self.n:]

        mu_x = mu_x.clamp(min=0.25, max=0.75)
        mu_y = mu_y.clamp(min=0.25, max=0.75)
        sigma_x = sigma_x.clamp(min=0.1, max=0.9)
        sigma_y = sigma_y.clamp(min=0.2, max=0.8)

        x_t = torch.mm(torch.ones(self.output_h, 1),
                       torch.linspace(0, 1.0, self.output_w).unsqueeze(0))
        y_t = torch.mm(torch.linspace(0, 1.0, self.output_h).unsqueeze(1),
                       torch.ones(1, self.output_w))

        x_t = x_t.unsqueeze(0).unsqueeze(0).repeat(mu_x.size(0), self.n, 1, 1).to(params.device)
        y_t = y_t.unsqueeze(0).unsqueeze(0).repeat(mu_x.size(0), self.n, 1, 1).to(params.device)
        # x_t = x_t.unsqueeze(0).unsqueeze(0).repeat(mu_x.size(0), self.n, 1, 1)
        # y_t = y_t.unsqueeze(0).unsqueeze(0).repeat(mu_x.size(0), self.n, 1, 1)

        gaussian = 1./(2*np.pi*torch.mul(sigma_x, sigma_y).unsqueeze(-1).unsqueeze(-1).expand_as(x_t))*\
            torch.exp(-(torch.div(torch.pow(x_t-mu_x.unsqueeze(-1).unsqueeze(-1).expand_as(x_t), 2),
                                  2*torch.pow(sigma_x.unsqueeze(-1).unsqueeze(-1),2)+1e-8) +
                        torch.div(torch.pow(y_t-mu_y.unsqueeze(-1).unsqueeze(-1).expand_as(y_t), 2),
                                  2*torch.pow(sigma_y.unsqueeze(-1).unsqueeze(-1),2)+1e-8)))
        max_gauss = torch.max(gaussian.view(gaussian.size(0), gaussian.size(1), -1), -1, keepdim=True).values
        gaussian = torch.div(gaussian, max_gauss.unsqueeze(-1))

        return gaussian

    def forward(self, x):
        # define forward process
        # x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        params = torch.sigmoid(self.fc2(x))

        maps = self.gen_gaussian_map(params)
        return maps

# dilate layer3 (2)
class resnet50_dilate(torch.nn.Module):
    def __init__(self):
        super(resnet50_dilate, self).__init__()

        blocks = models.resnet50(pretrained=False)
        self.conv1 = blocks.conv1
        self.bn1 = blocks.bn1
        self.relu = blocks.relu
        self.maxpool = blocks.maxpool
        self.layer1 = blocks.layer1
        self.layer2 = blocks.layer2

        # change to dilated non-downsampled version
        downsample_ly3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer3 = torch.nn.Sequential(
            Bottleneck(inplanes=512, planes=256, stride=1, downsample=downsample_ly3, groups=1,
                       base_width=64, dilation=2, norm_layer=None),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, groups=1,
                       base_width=64, dilation=2, norm_layer=None),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, groups=1,
                       base_width=64, dilation=2, norm_layer=None),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, groups=1,
                       base_width=64, dilation=2, norm_layer=None),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, groups=1,
                       base_width=64, dilation=2, norm_layer=None),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, groups=1,
                       base_width=64, dilation=2, norm_layer=None)
        )

        # downsample_ly4 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
        #     torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
        # self.layer4 = torch.nn.Sequential(
        #     Bottleneck(inplanes=1024, planes=512, stride=1, downsample=downsample_ly4, groups=1,
        #                base_width=64, dilation=4, norm_layer=None),
        #     Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, groups=1,
        #                base_width=64, dilation=4, norm_layer=None),
        #     Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, groups=1,
        #                base_width=64, dilation=4, norm_layer=None)
        # )
        self.layer4 = blocks.layer4

        # keep original
        self.avgpool = blocks.avgpool
        self.fc = blocks.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class attention_module_multi_head_RN_cls(torch.nn.Module):
    # fc_new_1 [num_rois, 1024]
    # attention, [num_rois, feat_dim]
    # attention_module_multi_head, eq. 5,4,3,2, 6(last half)
    # get attention_1 [num_rois, 1024] to concat with the fc_new_1 in eq.6
    # fc_all_1 = fc_new_1 + attention_1
    """ Attetion module with vectorized version

        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:

        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
    """
    def __init__(self, emb_dim=64, fc_dim=16, feat_dim=1024,
                      cls_num=coco_num_classes, dim=(1024, 1024, 1024), group=16, index=1):
        super(attention_module_multi_head_RN_cls, self).__init__()
        assert dim[0] == dim[1]
        assert fc_dim == group

        self.cls_num = cls_num
        self.emb_dim = emb_dim # position_embedding.size(-1)
        self.fc_dim = fc_dim
        self.feat_dim = feat_dim
        self.dim = dim
        self.group = group
        self.index = index
        self.dim_group = (self.dim[0] // self.group, self.dim[1] // self.group, self.dim[2] // self.group) # (64,64,64)

        # eval('self.pair_pos_fc1_%d = torch.nn.Linear(self.emb_dim, self.fc_dim)'%self.index)
        self.pair_pos_fc1 = torch.nn.Linear(self.emb_dim, self.fc_dim) # WG
        self.query = torch.nn.Linear(self.feat_dim, self.dim[0]) # WQ, roi_feat is the fA in eq.4 [num_rois, feat_dim]
        self.key = torch.nn.Linear(self.feat_dim, self.dim[1]) # WK, nongt_roi_feat is the fA in eq.4 [nongt_dim, feat_dim]

        self.linear_out = torch.nn.Conv2d(self.fc_dim*self.feat_dim, self.dim[2], kernel_size=(1,1), groups=self.fc_dim)
        # groups param == Nr in the paper, conv contains the concat op in eq.6

        self.read_out = torch.nn.Linear(self.dim[2], self.cls_num)

    def forward(self, bbox, roi_feat):
        nongt_dim = roi_feat.size(0)
        # nongt_roi_feat = roi_feat[:nongt_dim, :]
        # nongt_roi_feat = roi_feat.detach()
        nongt_roi_feat = roi_feat
        # bbox contains negative values
        position_matrix = self.extract_position_matrix(bbox, nongt_dim=nongt_dim)
        # print('position_matrix', position_matrix.max(), position_matrix.min())
        # position_matrix contains nan
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=self.emb_dim)

        # [num_rois, nongt_dim, emb_dim]-->[num_rois*nongt_dim, emb_dim]
        position_embedding_reshape = position_embedding.view(-1, position_embedding.size(2))

        # WG in eq.5
        # [num_rois*nongt_dim, emb_dim] position_embedding_reshape --> [num_rois*nongt_dim, fc_dim] position_feat_1
        # position_embedding_reshape contains nan
        position_feat_1 = self.pair_pos_fc1(position_embedding_reshape)
        if torch.isnan(position_feat_1).any():
            pdb.set_trace()
        position_feat_1_relu = F.relu(position_feat_1)
        if torch.isnan(position_feat_1_relu).any():
            pdb.set_trace()
        # position_feat_1_relu = F.relu(self.pair_pos_fc1(position_embedding_reshape))
        # aff_weight, [num_rois, nongt_dim, fc_dim]
        aff_weight = position_feat_1_relu.view(-1, nongt_dim, self.fc_dim)
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = aff_weight.permute(0, 2, 1)

        q_data = self.query(roi_feat)
        q_data_batch = q_data.view(-1, self.group, self.dim_group[0]) #[num_rois, group, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2) #[group, num_rois, dim_group[0]]

        k_data = self.key(nongt_roi_feat)
        k_data_batch = k_data.view(-1, self.group, self.dim_group[1])
        k_data_batch = k_data_batch.permute(1, 0, 2) # [group, nongt_dim, dim_group[0]]

        # v_data = nongt_roi_feat.detach() # version 1
        v_data = nongt_roi_feat # version 2

        aff = torch.bmm(q_data_batch, k_data_batch.permute(0, 2, 1))
        # aff_scale, [group, num_rois, nongt_dim]
        aff_scale = torch.mul(aff, 1./math.sqrt(float(self.dim_group[1])))
        aff_scale = aff_scale.permute(1, 0, 2) # input of the log function before wA

        # eq.5 e^(log(wG)+wA)=wG+e^(wA)
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
        aff_weight[aff_weight<1e-6] = 1e-6
        weighted_aff = torch.log(aff_weight) + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)
        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = aff_softmax.view(-1, aff_softmax.size(2))

        # output_t, [num_rois * fc_dim, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = output_t.view(-1, self.fc_dim*self.feat_dim, 1, 1)

        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.linear_out(output_t)
        # output [num_rois, dim[2]]
        linear_out = linear_out.squeeze(-1).squeeze(-1)
        linear_out = roi_feat + linear_out  # TODO: the article use residual to enhance the input feature.
        # return linear_out.squeeze(-1).squeeze(-1)

        # output = self.read_out(linear_out.mean(dim=0, keepdim=True)) # not good
        output = self.read_out(linear_out.sum(dim=0, keepdim=True)) # all the previous experiments
        # print('output', output.max(), output.min())

        output = torch.sigmoid(output)  # output [b_s, num_rois, 1]
        # output = torch.softmax(output, 1)  # output [b_s, num_rois, 1]
        return output
        # return output.view(1, self.grid_N, self.grid_N)

    def extract_position_matrix(self, bbox, nongt_dim):
        '''
        relation network for object detection cvpr2018
        Extract position matrix
        Args:
            bbox: [num_boxes, 4]
        returns:
            position_matrix: [num_boxes, nongt_dim, 4]
        '''

        # xmin, ymin, xmax, ymax are [num_boxes, 1]
        xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=1)

        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        # delta_x [num_boxes, num_boxes]
        delta_x = center_x.repeat(1, xmin.size(0)) - torch.transpose(center_x, 1, 0).repeat(xmin.size(0), 1)
        delta_x = torch.div(delta_x, bbox_width)
        delta_x[delta_x.abs() < 1e-3] = 1e-3
        delta_x = torch.log(delta_x.abs())

        if torch.isnan(delta_x).any():
            pdb.set_trace()

        delta_y = center_y.repeat(1, xmin.size(0)) - torch.transpose(center_y, 1, 0).repeat(xmin.size(0), 1)
        delta_y = torch.div(delta_y, bbox_height)
        delta_y[delta_y.abs() < 1e-3] = 1e-3
        delta_y = torch.log(delta_y.abs())

        if torch.isnan(delta_y).any():
            pdb.set_trace()

        delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 0))
        delta_width[delta_width.abs() < 1e-3] = 1e-3
        delta_width = torch.log(delta_width)

        if torch.isnan(delta_width).any():
            pdb.set_trace()

        delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 0))
        delta_height[delta_height.abs() < 1e-3] = 1e-3
        delta_height = torch.log(delta_height)

        if torch.isnan(delta_height).any():
            pdb.set_trace()

        # len(concat_list)=4, each element is of [num_boxes, num_boxes]
        concat_list = [delta_x, delta_y, delta_width, delta_height]

        # get 0~nongt_dim (default 300) values at dim 1
        # sym [num_boxes, nongt_dim] --> [num_boxes, nongt_dim, 1]
        # position_matrix [num_boxes, nongt_dim, 4]
        for idx, sym in enumerate(concat_list):
            tmp = sym[:, :nongt_dim]
            concat_list[idx] = tmp.unsqueeze(2)

        position_matrix = torch.cat(concat_list, dim=2)
        return position_matrix

    # Usage: position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
    def extract_position_embedding(self, position_mat, feat_dim, wave_length=1000):
        # position_mat, [num_rois, nongt_dim, 4]
        # feat_range [0,1, ... ,7], full: fist input is shape, second is value
        # dim_mat=[1., 2.37137365, 5.62341309, 13.33521461, 31.62277603, 74.98941803, 177.82794189, 421.69650269]
        feat_range = torch.arange(0, feat_dim / 8).float().to(position_mat.device)
        dim_mat = torch.pow(torch.full((1,), wave_length).to(position_mat.device), (8. / feat_dim) * feat_range)

        position_mat = position_mat.unsqueeze(3)  # [num_rois, nongt_dim, 4, 1]
        div_mat = torch.div(position_mat, dim_mat)  # [num_rois, nongt_dim, 4, 8]
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)
        # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # embedding, [num_rois, nongt_dim, feat_dim]
        embedding = embedding.view(embedding.size(0), embedding.size(1), feat_dim)
        return embedding

# same as paper. apply attention at (linear_out + roi_feat)
class attention_module_multi_head_RN_cls_sa_art_sp(torch.nn.Module):
    # fc_new_1 [num_rois, 1024]
    # attention, [num_rois, feat_dim]
    # attention_module_multi_head, eq. 5,4,3,2, 6(last half)
    # get attention_1 [num_rois, 1024] to concat with the fc_new_1 in eq.6
    # fc_all_1 = fc_new_1 + attention_1
    """ Attetion module with vectorized version

        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:

        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
    """
    def __init__(self, emb_dim=64, fc_dim=16, feat_dim=1024,
                      cls_num=coco_num_classes, dim=(1024, 1024, 1024), group=16, index=1):
        super(attention_module_multi_head_RN_cls_sa_art_sp, self).__init__()
        assert dim[0] == dim[1]
        assert fc_dim == group

        self.cls_num = cls_num
        self.emb_dim = emb_dim # position_embedding.size(-1)
        self.fc_dim = fc_dim
        self.feat_dim = feat_dim
        self.dim = dim
        self.group = group
        self.index = index
        self.dim_group = (self.dim[0] // self.group, self.dim[1] // self.group, self.dim[2] // self.group) # (64,64,64)

        # eval('self.pair_pos_fc1_%d = torch.nn.Linear(self.emb_dim, self.fc_dim)'%self.index)
        self.pair_pos_fc1 = torch.nn.Linear(self.emb_dim, self.fc_dim) # WG
        self.query = torch.nn.Linear(self.feat_dim, self.dim[0]) # WQ, roi_feat is the fA in eq.4 [num_rois, feat_dim]
        self.key = torch.nn.Linear(self.feat_dim, self.dim[1]) # WK, nongt_roi_feat is the fA in eq.4 [nongt_dim, feat_dim]

        self.linear_out = torch.nn.Conv2d(self.fc_dim*self.feat_dim, self.dim[2], kernel_size=(1,1), groups=self.fc_dim)
        # groups param == Nr in the paper, conv contains the concat op in eq.6

        self.read_out = torch.nn.Linear(self.dim[2], self.cls_num)

        self.self_attention = torch.nn.Conv2d(self.dim[2], 1, kernel_size=(1, 1))

    def forward(self, bbox, roi_feat):
        nongt_dim = roi_feat.size(0)
        # nongt_roi_feat = roi_feat[:nongt_dim, :]
        # nongt_roi_feat = roi_feat.detach()
        nongt_roi_feat = roi_feat
        # bbox contains negative values
        position_matrix = self.extract_position_matrix(bbox, nongt_dim=nongt_dim)
        # print('position_matrix', position_matrix.max(), position_matrix.min())
        # position_matrix contains nan
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=self.emb_dim)

        # [num_rois, nongt_dim, emb_dim]-->[num_rois*nongt_dim, emb_dim]
        position_embedding_reshape = position_embedding.view(-1, position_embedding.size(2))

        # WG in eq.5
        # [num_rois*nongt_dim, emb_dim] position_embedding_reshape --> [num_rois*nongt_dim, fc_dim] position_feat_1
        # position_embedding_reshape contains nan
        position_feat_1 = self.pair_pos_fc1(position_embedding_reshape)
        if torch.isnan(position_feat_1).any():
            pdb.set_trace()
        position_feat_1_relu = F.relu(position_feat_1)
        if torch.isnan(position_feat_1_relu).any():
            pdb.set_trace()
        # position_feat_1_relu = F.relu(self.pair_pos_fc1(position_embedding_reshape))
        # aff_weight, [num_rois, nongt_dim, fc_dim]
        aff_weight = position_feat_1_relu.view(-1, nongt_dim, self.fc_dim)
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = aff_weight.permute(0, 2, 1)

        q_data = self.query(roi_feat)
        q_data_batch = q_data.view(-1, self.group, self.dim_group[0]) #[num_rois, group, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2) #[group, num_rois, dim_group[0]]

        k_data = self.key(nongt_roi_feat)
        k_data_batch = k_data.view(-1, self.group, self.dim_group[1])
        k_data_batch = k_data_batch.permute(1, 0, 2) # [group, nongt_dim, dim_group[0]]

        # v_data = nongt_roi_feat.detach() # version 1
        v_data = nongt_roi_feat # version 2

        aff = torch.bmm(q_data_batch, k_data_batch.permute(0, 2, 1))
        # aff_scale, [group, num_rois, nongt_dim]
        aff_scale = torch.mul(aff, 1./math.sqrt(float(self.dim_group[1])))
        aff_scale = aff_scale.permute(1, 0, 2) # input of the log function before wA

        # eq.5 e^(log(wG)+wA)=wG+e^(wA)
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
        aff_weight[aff_weight<1e-6] = 1e-6
        weighted_aff = torch.log(aff_weight) + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)
        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = aff_softmax.view(-1, aff_softmax.size(2))

        # output_t, [num_rois * fc_dim, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = output_t.view(-1, self.fc_dim*self.feat_dim, 1, 1)

        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.linear_out(output_t)
        # output [num_rois, dim[2]]
        # pdb.set_trace()
        linear_out = linear_out.squeeze(-1).squeeze(-1)
        linear_out = roi_feat + linear_out  # TODO: the article use residual to enhance the input feature.

        # self attention mechanism; competition
        linear_out = linear_out.unsqueeze(-1).unsqueeze(-1)
        linear_out_score = self.self_attention(linear_out)  # (N, 1, 1, 1)
        linear_out_score_sft = torch.softmax(linear_out_score, dim=0)  # (N, 1, 1, 1)
        linear_out = torch.mul(linear_out, linear_out_score_sft)
        linear_out = linear_out.squeeze(-1).squeeze(-1)

        # return linear_out.squeeze(-1).squeeze(-1)
        # pdb.set_trace()
        # output = self.read_out(linear_out.mean(dim=0, keepdim=True)) # not good
        output = self.read_out(linear_out.sum(dim=0, keepdim=True)) # all the previous experiments
        # print('output', output.max(), output.min())

        output = torch.sigmoid(output)  # output [b_s, num_rois, 1]
        # output = torch.softmax(output, 1)  # output [b_s, num_rois, 1]
        # pdb.set_trace()
        return output, linear_out_score_sft

        # return output.view(1, self.grid_N, self.grid_N)

    def extract_position_matrix(self, bbox, nongt_dim):
        '''
        relation network for object detection cvpr2018
        Extract position matrix
        Args:
            bbox: [num_boxes, 4]
        returns:
            position_matrix: [num_boxes, nongt_dim, 4]
        '''

        # xmin, ymin, xmax, ymax are [num_boxes, 1]
        xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=1)

        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        # delta_x [num_boxes, num_boxes]
        delta_x = center_x.repeat(1, xmin.size(0)) - torch.transpose(center_x, 1, 0).repeat(xmin.size(0), 1)
        delta_x = torch.div(delta_x, bbox_width)
        delta_x[delta_x.abs() < 1e-3] = 1e-3
        delta_x = torch.log(delta_x.abs())

        if torch.isnan(delta_x).any():
            pdb.set_trace()

        delta_y = center_y.repeat(1, xmin.size(0)) - torch.transpose(center_y, 1, 0).repeat(xmin.size(0), 1)
        delta_y = torch.div(delta_y, bbox_height)
        delta_y[delta_y.abs() < 1e-3] = 1e-3
        delta_y = torch.log(delta_y.abs())

        if torch.isnan(delta_y).any():
            pdb.set_trace()

        delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 0))
        delta_width[delta_width.abs() < 1e-3] = 1e-3
        delta_width = torch.log(delta_width)

        if torch.isnan(delta_width).any():
            pdb.set_trace()

        delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 0))
        delta_height[delta_height.abs() < 1e-3] = 1e-3
        delta_height = torch.log(delta_height)

        if torch.isnan(delta_height).any():
            pdb.set_trace()

        # len(concat_list)=4, each element is of [num_boxes, num_boxes]
        concat_list = [delta_x, delta_y, delta_width, delta_height]

        # get 0~nongt_dim (default 300) values at dim 1
        # sym [num_boxes, nongt_dim] --> [num_boxes, nongt_dim, 1]
        # position_matrix [num_boxes, nongt_dim, 4]
        for idx, sym in enumerate(concat_list):
            tmp = sym[:, :nongt_dim]
            concat_list[idx] = tmp.unsqueeze(2)

        position_matrix = torch.cat(concat_list, dim=2)
        return position_matrix

    # Usage: position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
    def extract_position_embedding(self, position_mat, feat_dim, wave_length=1000):
        # position_mat, [num_rois, nongt_dim, 4]
        # feat_range [0,1, ... ,7], full: fist input is shape, second is value
        # dim_mat=[1., 2.37137365, 5.62341309, 13.33521461, 31.62277603, 74.98941803, 177.82794189, 421.69650269]
        feat_range = torch.arange(0, feat_dim / 8).float().to(position_mat.device)
        dim_mat = torch.pow(torch.full((1,), wave_length).to(position_mat.device), (8. / feat_dim) * feat_range)

        position_mat = position_mat.unsqueeze(3)  # [num_rois, nongt_dim, 4, 1]
        div_mat = torch.div(position_mat, dim_mat)  # [num_rois, nongt_dim, 4, 8]
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)
        # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # embedding, [num_rois, nongt_dim, feat_dim]
        embedding = embedding.view(embedding.size(0), embedding.size(1), feat_dim)
        return embedding

class TwoMLPHead_my(torch.nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead_my, self).__init__()

        self.fc6 = torch.nn.Linear(in_channels, representation_size)
        self.fc7 = torch.nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))

        return x


# ***
class Wildcat_WK_hd_gs_compf_cls_att_A4_cw(torch.nn.Module):
    def __init__(self, n_classes, kmax=1, kmin=None, alpha=0.7, num_maps=4, fix_feature=False,
                 use_grid=False):
        super(Wildcat_WK_hd_gs_compf_cls_att_A4_cw, self).__init__()
        self.n_classes = n_classes
        self.use_grid = use_grid
        model = resnet50_dilate()

        ckpt_file = base_path + 'Weights/resnet50.pth'
        pretrained_dict = torch.load(ckpt_file)
        model.load_state_dict(pretrained_dict)

        pooling = torch.nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))

        # ---------------------------------------------
        self.features = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4)

        if fix_feature:
            for param in self.features.parameters():
                param.requires_grad = False
            # for name, parameter in self.features.named_parameters():
            #     if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            #         parameter.requires_grad_(False)
        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(num_features, n_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling
        # self.feature_refine = ConvLSTM(input_size=(14, 14),
        #              input_dim=n_classes,
        #              hidden_dim=[n_classes, n_classes],
        #              kernel_size=(3, 3),
        #              num_layers=2,
        #              batch_first=True,
        #              bias = True,
        #              return_all_layers = False)

        # ----------------------------------------
        self.to_img_size = torch.nn.Upsample(size=(input_h, input_w))
        self.to_attention_size = torch.nn.Upsample(size=(output_h, output_w))
        self.to_grid_size = torch.nn.Upsample(size=(7, 7))
        self.boxes_grid = gen_grid_boxes(in_h=input_h, in_w=input_w, N=7)  # 7 with features_fd
        self.grid_N = self.boxes_grid.size(1)

        self.relation_net = attention_module_multi_head_RN_cls(feat_dim=512, fc_dim=1, group=1, cls_num=n_classes,
                                                               dim=[512]*3)

        self.to_cw_feature_size = torch.nn.Upsample(size=(28, 28))
        # self.to_cw_feature_size = torch.nn.Upsample(size=(14, 14))
        self.to_output_size = torch.nn.Upsample(size=(output_h, output_w))

        # self.centerbias = CenterBias_A(n=n_gaussian, input_c=num_features) # gs_A_x
        self.centerbias = CenterBias_A(n=n_gaussian, input_c=n_classes*num_maps) #, in_h=28, in_w=28
        # self.centerbias = CenterBias_G(n=n_gaussian)

        self.gen_g_feature = torch.nn.Conv2d(n_classes * num_maps + n_gaussian, n_classes * num_maps, kernel_size=1)

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['layer1', 'layer2', 'layer3', 'layer4'],
            output_size=7,
            sampling_ratio=2)

        resolution = self.box_roi_pool.output_size[0]
        representation_size = 512
        out_channels = 256
        self.box_head = TwoMLPHead_my(
            out_channels * resolution ** 2,
            representation_size)

        in_channels_stage2 = 256
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, img, boxes, boxes_nums):  # img size 224
        # define forward process
        image_sizes = [im.size()[-2:] for im in img]
        boxes_list = [boxes[b_i, :boxes_nums[b_i], :] for b_i in range(len(boxes_nums))] #//self.cuda_num

        # if self.use_grid:
        features = OrderedDict()
        x = self.features[:5](img)
        features['layer1'] = x
        x = self.features[5](x)
        features['layer2'] = x
        x = self.features[6](x)
        features['layer3'] = x
        x = self.features[7](x)
        features['layer4'] = x

        features = self.fpn(features)

        processed_features = self.box_roi_pool(features, boxes_list, image_sizes)  # N, 256, 7, 7
        processed_features = self.box_head(processed_features)  # N, 1024

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in boxes_list]
        box_feature = processed_features.split(boxes_per_image, 0)

        # pdb.set_trace()
        # print('box_feature[0]', box_feature[0].max(), box_feature[0].min())

        if self.use_grid:
            box_feature_grid = torch.cat([self.to_grid_size(features['layer3']), self.to_grid_size(features['layer4'])],
                                         dim=1).permute(0, 2, 3, 1)
            box_feature_grid = box_feature_grid.view(x.size(0), -1, box_feature_grid.size(3))

            assert box_feature_grid.size(1) == self.grid_N

            # box_feature = torch.cat([box_feature_grid, box_feature], dim=1)
            box_feature = [torch.cat([box_feature_grid[b_i], box_feature[b_i]], dim=0) for b_i in range(len(boxes_nums))]
            # print('box_feature[0]', box_feature[0].max(), box_feature[0].min())

            self.boxes_grid = self.boxes_grid.to(img.device)
            boxes = torch.cat([self.boxes_grid.repeat(img.size(0), 1, 1), boxes], dim=1)
            boxes_nums = [n + self.grid_N for n in boxes_nums]
            boxes_list = [boxes[b_i, :boxes_nums[b_i], :] for b_i in range(len(boxes_nums))]


        # ori_logits = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        # print('ori_logits', ori_logits.size())
        x = self.classifier(x)  # (N, 1000, 7, 7) # previously, mask x, then pass to self.comp_pooling

        # ori_logits = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)


        gaussian = self.centerbias(x) #previous n_gaussian=8 use this settings; as all other n_gaussians
        if gaussian.size(0) != x.size(0):
            gaussian = gaussian.repeat(x.size(0), 1, 1, 1)
        # if gaussian.size(2) != x.size(2):
        #     gaussian = F.interpolate(gaussian, size=(x.size(2), x.size(3)))

        x = self.gen_g_feature(torch.cat([x, gaussian], dim=1))
        cw_maps = self.spatial_pooling.class_wise(x)  # (N, 1000, 7, 7)
        # cw_maps_rpt = torch.cat([cw_maps.unsqueeze(1), cw_maps.unsqueeze(1)], dim=1)
        # cw_maps_out = self.feature_refine(cw_maps_rpt)
        # cw_maps_refined = cw_maps_out[1][0][0]
        # pred_logits = self.spatial_pooling.spatial(cw_maps_refined)  # (N, 1000)

        # sft_scores = torch.sigmoid(pred_logits)  # the combined maps looks better...
        #
        # map = torch.mul(sft_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps), # TODO change map to hd_map
        #                 torch.sigmoid(cw_maps))
        # map = torch.div(map.sum(1, keepdim=True), sft_scores.sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1))

        # hard_sal_map = torch.zeros(img.size(0), 1, img.size(-2), img.size(-1))

        hard_scores = torch.zeros(img.size(0), self.n_classes)
        hard_scores = hard_scores.to(img.device)
        # hard_sal_map = torch.zeros(img.size(0), 1, self.grid_N, self.grid_N)
        # hard_sal_map = hard_sal_map.to(img.device)

        # print('boxes_nums', boxes_nums)
        #if np.sum(np.array(boxes_nums))>0:
            # box_scores = torch.zeros(img.size(0), np.array(boxes_nums).max()).to(img.device)

        for b_i in range(img.size(0)):
            if boxes_nums[b_i] > 0:
                # hard_scores[b_i, :] = self.relation_net(boxes[b_i, :boxes_nums[b_i], :], box_feature[b_i, :boxes_nums[b_i], :])
                hard_scores[b_i, :] = self.relation_net(boxes_list[b_i], box_feature[b_i])

        # hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps),  # TODO change map to hd_map
        #                cw_maps).sum(1, keepdim=True)
        # hard_sal_map = torch.sigmoid(hard_sal_map)

        # --add
        # hard_scores = torch.add(pred_logits, hard_scores)
        # hard_scores = torch.add(torch.sigmoid(pred_logits), hard_scores)

        # --multi
        # hard_scores = torch.mul(pred_logits, hard_scores)
        # hard_scores = torch.mul(torch.sigmoid(pred_logits), hard_scores)

        # --multi & add1
        # hard_scores = torch.add(pred_logits, torch.mul(pred_logits, hard_scores))
        # hard_scores = torch.add(torch.sigmoid(pred_logits), torch.mul(torch.sigmoid(pred_logits), hard_scores))

        # --multi & add2
        # hard_scores = torch.add(torch.mul(pred_logits, hard_scores), hard_scores)
        # hard_scores = torch.add(torch.mul(torch.sigmoid(pred_logits), hard_scores), hard_scores)

        # hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps_refined),
        #                          torch.sigmoid(cw_maps_refined)).sum(1, keepdim=True) ## 2
        hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps),  # TODO change map to hd_map
                        torch.sigmoid(cw_maps)).sum(1, keepdim=True) ## 1
        hard_sal_map = torch.div(hard_sal_map, hard_scores.sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)+1e-8)

        # hard_sal_map = self.to_cw_feature_size(hard_sal_map)
        # gs_map = torch.mul(hard_scores[:, -n_gaussian:].unsqueeze(-1).unsqueeze(-1).expand_as(gaussian),  # TODO change map to hd_map
        #                 gaussian).sum(1, keepdim=True)

        # hard_sal_map = gauss(hard_sal_map) # initially this looks better ...
        # torch.clamp(hard_sal_map, min=0.0, max=1.0)

        # hard_sal_map = torch.div(hard_sal_map,
        #                          hard_sal_map.max(-1, keepdim=True).values.max(-2, keepdim=True).values+1e-8)

        # hard_sal_map = torch.sigmoid(hard_sal_map) # _sameb3_nosig

        # hard_sal_map = gauss(hard_sal_map)

        # masked_cw_maps = torch.mul(cw_maps_refined, hard_sal_map)
        # masked_cw_maps = masked_cw_maps + cw_maps_refined  # TODO: comp_self_res
        # pred_comp_logits = self.spatial_pooling.spatial(masked_cw_maps) ## 2
        masked_cw_maps = torch.mul(cw_maps, hard_sal_map)
        masked_cw_maps = masked_cw_maps + cw_maps  # TODO: comp_self_res
        pred_comp_logits = self.spatial_pooling.spatial(masked_cw_maps) ## 1

        sal_map = self.to_attention_size(hard_sal_map)
        # sal_map = torch.sigmoid(sal_map)

        # return pred_logits, F.softmax(ori_logits, -1), torch.sigmoid(sal_map)
        # return pred_logits, pred_comp_logits, torch.clamp(sal_map, min=0.0, max=1.0)
        return pred_comp_logits, sal_map #, gaussian, gs_map
        # return pred_logits, pred_comp_logits, sal_map #, gaussian, gs_map
        # return pred_logits, pred_comp_logits, torch.sigmoid(sal_map)


# generate attention map for one image at one time
class GenAttentionMapFunction(torch.autograd.Function):
    """
    handling one image at one time
    att_scores: the soft attention score (box_num, 1)
    boxes: tensor (box_num, 4)
    input_size: tuple (w, h) or np array (2,), e.g a.shape[-2:]
    output_size: tuple (w, h) or np array (2,), e.g a.shape[-2:]
    """
    @staticmethod
    def forward(ctx, att_scores, boxes, input_size, output_size):
        ctx.save_for_backward(att_scores, boxes)
        ctx.input_size = input_size
        ctx.output_size = output_size

        possible_scales = []
        for s1, s2 in zip(output_size, input_size):
            approx_scale = float(s1) / s2
            scale = 2 ** torch.tensor(approx_scale).log2().round().item()
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]

        resized_boxes = boxes * possible_scales[0]

        att_maps = torch.zeros(att_scores.size(0), output_size[0], output_size[1],
                               dtype=att_scores.dtype, device=att_scores.device)
        for box_i in range(att_scores.size(0)):
            box = resized_boxes[box_i].int()

            att_maps[box_i, box[1]:box[3], box[0]:box[2]] = att_scores[box_i]

        final_map = att_maps.sum(0)
        # final_map = att_maps # I think this is more reasonable for calculating the att_scores' grads
        # print('final_map', final_map.size())
        return final_map


    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        att_scores, boxes = ctx.saved_tensors
        input_size = ctx.input_size
        output_size = ctx.output_size
        # print('att_scores', att_scores.size())
        # print('grad_output',grad_output.size())

        grad_att_scores = None
        # comment below block for _nob (no gradient backpropagation)
        if ctx.needs_input_grad[0]:
           grad_att_scores = grad_output.sum()
           grad_att_scores = grad_att_scores.expand_as(att_scores) # same grad
        #     # grad_att_scores = grad_output.sum() * att_scores
        #
        #     # possible_scales = []
        #     # for s1, s2 in zip(output_size, input_size):
        #     #     approx_scale = float(s1) / s2
        #     #     scale = 2 ** torch.tensor(approx_scale, dtype=grad_output.dtype, device=grad_output.device).log2().round().item()
        #     #     # scale = 2 ** approx_scale.clone().detach().log2().round().item()
        #     #     possible_scales.append(scale)
        #     # assert possible_scales[0] == possible_scales[1]
        #     #
        #     # resized_boxes = boxes * possible_scales[0]
        #     # # att_maps = torch.zeros_like(grad_output)
        #     # grad_att_scores = torch.zeros_like(att_scores)
        #     # grad_output_2 = grad_output.clone()
        #     # for box_i in range(att_scores.size(0)):
        #     #     # att_maps = torch.zeros_like(grad_output)
        #     #     # box = resized_boxes[box_i].int()
        #     #     # att_maps[box[1]:box[3], box[0]:box[2]] = 1.0
        #     #     # grad_att_scores[box_i] = torch.mul(att_maps, grad_output).sum()
        #     #     box = resized_boxes[box_i].int()
        #     #     tmp = grad_output_2[box[1]:box[3], box[0]:box[2]]
        #     #     grad_att_scores[box_i] = tmp.sum()
        #     #
        #     # # pdb.set_trace()
        #     # # grad_att_scores = torch.mul(att_maps, grad_output.unsqueeze(0).expand_as(att_maps))
        #     # # grad_att_scores = torch.sum(grad_att_scores, dim=[1,2])

        return grad_att_scores, None, None, None

def gen_attention_map(att_scores, boxes, input_size, output_size):
    # return GenAttentionMapFunction()(att_scores, boxes, input_size, output_size)
    return GenAttentionMapFunction.apply(att_scores, boxes, input_size, output_size)

class Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art(torch.nn.Module):
    def __init__(self, n_classes, kmax=1, kmin=None, alpha=0.7, num_maps=4, fix_feature=False,
                 use_grid=False):
        super(Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art, self).__init__()
        self.n_classes = n_classes
        self.use_grid = use_grid
        model = resnet50_dilate()

        ckpt_file = base_path + 'Weights/resnet50.pth'
        pretrained_dict = torch.load(ckpt_file)
        model.load_state_dict(pretrained_dict)

        pooling = torch.nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))

        # ---------------------------------------------
        self.features = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4)

        if fix_feature:
            for param in self.features.parameters():
                param.requires_grad = False
            # for name, parameter in self.features.named_parameters():
            #     if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            #         parameter.requires_grad_(False)
        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(num_features, n_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling
        # self.feature_refine = ConvLSTM(input_size=(14, 14),
        #              input_dim=n_classes,
        #              hidden_dim=[n_classes, n_classes],
        #              kernel_size=(3, 3),
        #              num_layers=2,
        #              batch_first=True,
        #              bias = True,
        #              return_all_layers = False)

        # ----------------------------------------
        self.to_img_size = torch.nn.Upsample(size=(input_h, input_w))
        self.to_attention_size = torch.nn.Upsample(size=(output_h, output_w))
        self.to_grid_size = torch.nn.Upsample(size=(7, 7))
        self.boxes_grid = gen_grid_boxes(in_h=input_h, in_w=input_w, N=7)  # 7 with features_fd
        self.grid_N = self.boxes_grid.size(1)

        self.relation_net = attention_module_multi_head_RN_cls_sa_art_sp(feat_dim=512, fc_dim=1, group=1, cls_num=n_classes,
                                                               dim=[512]*3)

        self.to_cw_feature_size = torch.nn.Upsample(size=(28, 28))
        # self.to_cw_feature_size = torch.nn.Upsample(size=(14, 14))
        self.to_output_size = torch.nn.Upsample(size=(output_h, output_w))

        # self.centerbias = CenterBias_A(n=n_gaussian, input_c=num_features) # gs_A_x
        self.centerbias = CenterBias_A(n=n_gaussian, input_c=n_classes*num_maps) #, in_h=28, in_w=28
        # self.centerbias = CenterBias_G(n=n_gaussian)

        self.gen_g_feature = torch.nn.Conv2d(n_classes * num_maps + n_gaussian, n_classes * num_maps, kernel_size=1)

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['layer1', 'layer2', 'layer3', 'layer4'],
            output_size=7,
            sampling_ratio=2)

        resolution = self.box_roi_pool.output_size[0]
        representation_size = 512
        out_channels = 256
        self.box_head = TwoMLPHead_my(
            out_channels * resolution ** 2,
            representation_size)

        in_channels_stage2 = 256
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, img, boxes, boxes_nums):  # img size 224
        # define forward process
        image_sizes = [im.size()[-2:] for im in img]
        boxes_list = [boxes[b_i, :boxes_nums[b_i], :] for b_i in range(len(boxes_nums))] #//self.cuda_num

        # if self.use_grid:
        features = OrderedDict()
        x = self.features[:5](img)
        features['layer1'] = x
        x = self.features[5](x)
        features['layer2'] = x
        x = self.features[6](x)
        features['layer3'] = x
        x = self.features[7](x)
        features['layer4'] = x

        features = self.fpn(features)

        processed_features = self.box_roi_pool(features, boxes_list, image_sizes)  # N, 256, 7, 7
        processed_features = self.box_head(processed_features)  # N, 1024

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in boxes_list]
        box_feature = processed_features.split(boxes_per_image, 0)

        # pdb.set_trace()
        # print('box_feature[0]', box_feature[0].max(), box_feature[0].min())

        if self.use_grid:
            box_feature_grid = torch.cat([self.to_grid_size(features['layer3']), self.to_grid_size(features['layer4'])],
                                         dim=1).permute(0, 2, 3, 1)
            box_feature_grid = box_feature_grid.view(x.size(0), -1, box_feature_grid.size(3))

            assert box_feature_grid.size(1) == self.grid_N

            # box_feature = torch.cat([box_feature_grid, box_feature], dim=1)
            box_feature = [torch.cat([box_feature_grid[b_i], box_feature[b_i]], dim=0) for b_i in range(len(boxes_nums))]
            # print('box_feature[0]', box_feature[0].max(), box_feature[0].min())

            self.boxes_grid = self.boxes_grid.to(img.device)
            boxes = torch.cat([self.boxes_grid.repeat(img.size(0), 1, 1), boxes], dim=1)
            boxes_nums = [n + self.grid_N for n in boxes_nums]
            boxes_list = [boxes[b_i, :boxes_nums[b_i], :] for b_i in range(len(boxes_nums))]


        # ori_logits = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        # print('ori_logits', ori_logits.size())
        x = self.classifier(x)  # (N, 1000, 7, 7) # previously, mask x, then pass to self.comp_pooling

        # ori_logits = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)


        gaussian = self.centerbias(x) #previous n_gaussian=8 use this settings; as all other n_gaussians
        if gaussian.size(0) != x.size(0):
            gaussian = gaussian.repeat(x.size(0), 1, 1, 1)
        if gaussian.size(2) != x.size(2):
            gaussian = F.interpolate(gaussian, size=(x.size(2), x.size(3)))

        x = self.gen_g_feature(torch.cat([x, gaussian], dim=1))
        cw_maps = self.spatial_pooling.class_wise(x)  # (N, 1000, 7, 7)
        # cw_maps_rpt = torch.cat([cw_maps.unsqueeze(1), cw_maps.unsqueeze(1)], dim=1)
        # cw_maps_out = self.feature_refine(cw_maps_rpt)
        # cw_maps_refined = cw_maps_out[1][0][0]
        # pred_logits = self.spatial_pooling.spatial(cw_maps_refined)  # (N, 1000)

        # sft_scores = torch.sigmoid(pred_logits)  # the combined maps looks better...
        #
        # map = torch.mul(sft_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps), # TODO change map to hd_map
        #                 torch.sigmoid(cw_maps))
        # map = torch.div(map.sum(1, keepdim=True), sft_scores.sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1))

        # hard_sal_map = torch.zeros(img.size(0), 1, img.size(-2), img.size(-1))

        hard_scores = torch.zeros(img.size(0), self.n_classes)
        hard_scores = hard_scores.to(img.device)
        # hard_sal_map = torch.zeros(img.size(0), 1, self.grid_N, self.grid_N)
        # hard_sal_map = hard_sal_map.to(img.device)

        att_scores = torch.zeros(img.size(0), 3)  # max & min & argmax 
        att_scores = att_scores.to(img.device)

        # print('boxes_nums', boxes_nums)
        #if np.sum(np.array(boxes_nums))>0:
            # box_scores = torch.zeros(img.size(0), np.array(boxes_nums).max()).to(img.device)

        for b_i in range(img.size(0)):
            if boxes_nums[b_i] > 0:
                # hard_scores[b_i, :] = self.relation_net(boxes[b_i, :boxes_nums[b_i], :], box_feature[b_i, :boxes_nums[b_i], :])
                hard_scores[b_i, :], linear_out_score_sft = \
                     self.relation_net(boxes_list[b_i], box_feature[b_i])
                att_scores[b_i, 0], att_scores[b_i, 1], att_scores[b_i, 2]= \
                    linear_out_score_sft.max(), linear_out_score_sft.min(), torch.argmax(linear_out_score_sft)

        # hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps),  # TODO change map to hd_map
        #                cw_maps).sum(1, keepdim=True)
        # hard_sal_map = torch.sigmoid(hard_sal_map)

        # --add
        # hard_scores = torch.add(pred_logits, hard_scores)
        # hard_scores = torch.add(torch.sigmoid(pred_logits), hard_scores)

        # --multi
        # hard_scores = torch.mul(pred_logits, hard_scores)
        # hard_scores = torch.mul(torch.sigmoid(pred_logits), hard_scores)

        # --multi & add1
        # hard_scores = torch.add(pred_logits, torch.mul(pred_logits, hard_scores))
        # hard_scores = torch.add(torch.sigmoid(pred_logits), torch.mul(torch.sigmoid(pred_logits), hard_scores))

        # --multi & add2
        # hard_scores = torch.add(torch.mul(pred_logits, hard_scores), hard_scores)
        # hard_scores = torch.add(torch.mul(torch.sigmoid(pred_logits), hard_scores), hard_scores)

        # hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps_refined),
        #                          torch.sigmoid(cw_maps_refined)).sum(1, keepdim=True) ## 2
        hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps),  # TODO change map to hd_map
                        torch.sigmoid(cw_maps)).sum(1, keepdim=True) ## 1
        hard_sal_map = torch.div(hard_sal_map, hard_scores.sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)+1e-8)

        # hard_sal_map = self.to_cw_feature_size(hard_sal_map)
        # gs_map = torch.mul(hard_scores[:, -n_gaussian:].unsqueeze(-1).unsqueeze(-1).expand_as(gaussian),  # TODO change map to hd_map
        #                 gaussian).sum(1, keepdim=True)

        # hard_sal_map = gauss(hard_sal_map) # initially this looks better ...
        # torch.clamp(hard_sal_map, min=0.0, max=1.0)

        # hard_sal_map = torch.div(hard_sal_map,
        #                          hard_sal_map.max(-1, keepdim=True).values.max(-2, keepdim=True).values+1e-8)

        # hard_sal_map = torch.sigmoid(hard_sal_map) # _sameb3_nosig

        # hard_sal_map = gauss(hard_sal_map)

        # masked_cw_maps = torch.mul(cw_maps_refined, hard_sal_map)
        # masked_cw_maps = masked_cw_maps + cw_maps_refined  # TODO: comp_self_res
        # pred_comp_logits = self.spatial_pooling.spatial(masked_cw_maps) ## 2
        masked_cw_maps = torch.mul(cw_maps, hard_sal_map)
        masked_cw_maps = masked_cw_maps + cw_maps  # TODO: comp_self_res
        pred_comp_logits = self.spatial_pooling.spatial(masked_cw_maps) ## 1

        sal_map = self.to_attention_size(hard_sal_map)
        # sal_map = torch.sigmoid(sal_map)

        # return pred_logits, F.softmax(ori_logits, -1), torch.sigmoid(sal_map)
        # return pred_logits, pred_comp_logits, torch.clamp(sal_map, min=0.0, max=1.0)
        return pred_comp_logits, sal_map, att_scores #, gaussian, gs_map
        # return pred_logits, pred_comp_logits, sal_map #, gaussian, gs_map
        # return pred_logits, pred_comp_logits, torch.sigmoid(sal_map)



class Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(torch.nn.Module):
    def __init__(self, n_classes, kmax=1, kmin=None, alpha=0.7, num_maps=4, fix_feature=False,
                 use_grid=False):
        super(Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp, self).__init__()
        self.n_classes = n_classes
        self.use_grid = use_grid
        model = resnet50_dilate()

        ckpt_file = base_path + 'Weights/resnet50.pth'
        pretrained_dict = torch.load(ckpt_file)
        model.load_state_dict(pretrained_dict)

        pooling = torch.nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))

        # ---------------------------------------------
        self.features = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4)

        if fix_feature:
            for param in self.features.parameters():
                param.requires_grad = False
            # for name, parameter in self.features.named_parameters():
            #     if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            #         parameter.requires_grad_(False)
        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(num_features, n_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling
        # self.feature_refine = ConvLSTM(input_size=(14, 14),
        #              input_dim=n_classes,
        #              hidden_dim=[n_classes, n_classes],
        #              kernel_size=(3, 3),
        #              num_layers=2,
        #              batch_first=True,
        #              bias = True,
        #              return_all_layers = False)

        # ----------------------------------------
        self.to_img_size = torch.nn.Upsample(size=(input_h, input_w))
        self.to_attention_size = torch.nn.Upsample(size=(output_h, output_w))
        self.to_grid_size = torch.nn.Upsample(size=(7, 7))
        self.boxes_grid = gen_grid_boxes(in_h=input_h, in_w=input_w, N=7)  # 7 with features_fd
        self.grid_N = self.boxes_grid.size(1)

        self.relation_net = attention_module_multi_head_RN_cls_sa_art_sp(feat_dim=512, fc_dim=1, group=1, cls_num=n_classes,
                                                               dim=[512]*3)

        self.to_cw_feature_size = torch.nn.Upsample(size=(28, 28))
        # self.to_cw_feature_size = torch.nn.Upsample(size=(14, 14))
        self.to_output_size = torch.nn.Upsample(size=(output_h, output_w))

        # self.centerbias = CenterBias_A(n=n_gaussian, input_c=num_features) # gs_A_x
        self.centerbias = CenterBias_A(n=n_gaussian, input_c=n_classes*num_maps) #, in_h=28, in_w=28
        # self.centerbias = CenterBias_G(n=n_gaussian)

        self.gen_g_feature = torch.nn.Conv2d(n_classes * num_maps + n_gaussian, n_classes * num_maps, kernel_size=1)

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['layer1', 'layer2', 'layer3', 'layer4'],
            output_size=7,
            sampling_ratio=2)

        resolution = self.box_roi_pool.output_size[0]
        representation_size = 512
        out_channels = 256
        self.box_head = TwoMLPHead_my(
            out_channels * resolution ** 2,
            representation_size)

        in_channels_stage2 = 256
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, img, boxes, boxes_nums):  # img size 224
        # define forward process
        image_sizes = [im.size()[-2:] for im in img]
        # print(img.size(), boxes.size(), len(boxes_nums))
        boxes_list = [boxes[b_i, :boxes_nums[b_i], :] for b_i in range(len(boxes_nums))] #//self.cuda_num

        # if self.use_grid:
        features = OrderedDict()
        x = self.features[:5](img)
        features['layer1'] = x
        x = self.features[5](x)
        features['layer2'] = x
        x = self.features[6](x)
        features['layer3'] = x
        x = self.features[7](x)
        features['layer4'] = x

        features = self.fpn(features)

        processed_features = self.box_roi_pool(features, boxes_list, image_sizes)  # N, 256, 7, 7
        processed_features = self.box_head(processed_features)  # N, 1024

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in boxes_list]
        box_feature = processed_features.split(boxes_per_image, 0)

        # pdb.set_trace()
        # print('box_feature[0]', box_feature[0].max(), box_feature[0].min())

        if self.use_grid:
            box_feature_grid = torch.cat([self.to_grid_size(features['layer3']), self.to_grid_size(features['layer4'])],
                                         dim=1).permute(0, 2, 3, 1)
            box_feature_grid = box_feature_grid.view(x.size(0), -1, box_feature_grid.size(3))

            assert box_feature_grid.size(1) == self.grid_N

            # box_feature = torch.cat([box_feature_grid, box_feature], dim=1)
            box_feature = [torch.cat([box_feature_grid[b_i], box_feature[b_i]], dim=0) for b_i in range(len(boxes_nums))]
            # print('box_feature[0]', box_feature[0].max(), box_feature[0].min())

            self.boxes_grid = self.boxes_grid.to(img.device)
            boxes = torch.cat([self.boxes_grid.repeat(img.size(0), 1, 1), boxes], dim=1)
            boxes_nums = [n + self.grid_N for n in boxes_nums]
            boxes_list = [boxes[b_i, :boxes_nums[b_i], :] for b_i in range(len(boxes_nums))]


        # ori_logits = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        # print('ori_logits', ori_logits.size())
        x = self.classifier(x)  # (N, 1000, 7, 7) # previously, mask x, then pass to self.comp_pooling

        # ori_logits = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)


        gaussian = self.centerbias(x) #previous n_gaussian=8 use this settings; as all other n_gaussians
        if gaussian.size(0) != x.size(0):
            gaussian = gaussian.repeat(x.size(0), 1, 1, 1)
        if gaussian.size(2) != x.size(2):
            gaussian = F.interpolate(gaussian, size=(x.size(2), x.size(3)))

        x = self.gen_g_feature(torch.cat([x, gaussian], dim=1))
        cw_maps = self.spatial_pooling.class_wise(x)  # (N, 1000, 7, 7)
        # cw_maps_rpt = torch.cat([cw_maps.unsqueeze(1), cw_maps.unsqueeze(1)], dim=1)
        # cw_maps_out = self.feature_refine(cw_maps_rpt)
        # cw_maps_refined = cw_maps_out[1][0][0]
        # pred_logits = self.spatial_pooling.spatial(cw_maps_refined)  # (N, 1000)

        # sft_scores = torch.sigmoid(pred_logits)  # the combined maps looks better...
        #
        # map = torch.mul(sft_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps), # TODO change map to hd_map
        #                 torch.sigmoid(cw_maps))
        # map = torch.div(map.sum(1, keepdim=True), sft_scores.sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1))

        # hard_sal_map = torch.zeros(img.size(0), 1, img.size(-2), img.size(-1))

        hard_scores = torch.zeros(img.size(0), self.n_classes)
        hard_scores = hard_scores.to(img.device)
        # hard_sal_map = torch.zeros(img.size(0), 1, self.grid_N, self.grid_N)
        # hard_sal_map = hard_sal_map.to(img.device)

        obj_att_maps = torch.zeros(img.size(0), 1, cw_maps.size(2), cw_maps.size(3), device=img.device)  # max & min & argmax
        # att_maps = att_maps.to(img.device)

        # print('boxes_nums', boxes_nums)
        #if np.sum(np.array(boxes_nums))>0:
            # box_scores = torch.zeros(img.size(0), np.array(boxes_nums).max()).to(img.device)

        possible_scales = []
        for s1, s2 in zip(x.size()[-2:], img.size()[-2:]):
            approx_scale = float(s1) / s2
            scale = 2 ** torch.tensor(approx_scale).log2().round().item()
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        # resized_boxes = boxes * possible_scales[0]
        for b_i in range(img.size(0)):
            if boxes_nums[b_i] > 0:
                # hard_scores[b_i, :] = self.relation_net(boxes[b_i, :boxes_nums[b_i], :], box_feature[b_i, :boxes_nums[b_i], :])
                hard_scores[b_i, :], att_scores = self.relation_net(boxes_list[b_i], box_feature[b_i])
                # att_scores = torch.ones_like(att_scores) # for no competition, set att_scores to be equal for each object
                # att_scores = torch.div(att_scores, att_scores.size(0))

                obj_att_maps[b_i, 0, :, :] = gen_attention_map(att_scores.squeeze(-1).squeeze(-1).squeeze(-1), boxes_list[b_i],
                                                                img.size()[-2:], cw_maps.size()[-2:])
                # tmp_att_maps = gen_attention_map(att_scores.squeeze(), boxes_list[b_i], img.size()[-2:], cw_maps.size()[-2:])
                # obj_att_maps[b_i, 0, :, :] = tmp_att_maps.sum(0)
            else:
                # att_scores = torch.tensor([0], device=img.device)
                att_scores = torch.tensor([[[[0]]]], device=img.device)

        # hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps),  # TODO change map to hd_map
        #                cw_maps).sum(1, keepdim=True)
        # hard_sal_map = torch.sigmoid(hard_sal_map)

        # --add
        # hard_scores = torch.add(pred_logits, hard_scores)
        # hard_scores = torch.add(torch.sigmoid(pred_logits), hard_scores)

        # --multi
        # hard_scores = torch.mul(pred_logits, hard_scores)
        # hard_scores = torch.mul(torch.sigmoid(pred_logits), hard_scores)

        # --multi & add1
        # hard_scores = torch.add(pred_logits, torch.mul(pred_logits, hard_scores))
        # hard_scores = torch.add(torch.sigmoid(pred_logits), torch.mul(torch.sigmoid(pred_logits), hard_scores))

        # --multi & add2
        # hard_scores = torch.add(torch.mul(pred_logits, hard_scores), hard_scores)
        # hard_scores = torch.add(torch.mul(torch.sigmoid(pred_logits), hard_scores), hard_scores)

        # hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps_refined),
        #                          torch.sigmoid(cw_maps_refined)).sum(1, keepdim=True) ## 2
        hard_sal_map = torch.mul(hard_scores.unsqueeze(-1).unsqueeze(-1).expand_as(cw_maps),  # TODO change map to hd_map
                        torch.sigmoid(cw_maps)).sum(1, keepdim=True) ## 1
        hard_sal_map = torch.div(hard_sal_map, hard_scores.sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)+1e-8)

        # hard_sal_map = torch.mul(hard_sal_map, obj_att_maps)

        # tmp_sal_map = torch.cat([hard_sal_map, obj_att_maps], dim=1) # _cat
        # hard_sal_map = torch.sigmoid(self.fuse_map(tmp_sal_map)) # _cat

        tmp_sal_map = torch.mul(hard_sal_map, obj_att_maps) # _mres
        hard_sal_map = hard_sal_map + tmp_sal_map # _mres

        # hard_sal_map = 0.5 * (hard_sal_map+obj_att_maps) # _avg

        # hard_sal_map = self.to_cw_feature_size(hard_sal_map)
        # gs_map = torch.mul(hard_scores[:, -n_gaussian:].unsqueeze(-1).unsqueeze(-1).expand_as(gaussian),  # TODO change map to hd_map
        #                 gaussian).sum(1, keepdim=True)

        # hard_sal_map = gauss(hard_sal_map) # initially this looks better ...
        # torch.clamp(hard_sal_map, min=0.0, max=1.0)

        # hard_sal_map = torch.div(hard_sal_map,
        #                          hard_sal_map.max(-1, keepdim=True).values.max(-2, keepdim=True).values+1e-8)

        # hard_sal_map = torch.sigmoid(hard_sal_map) # _sameb3_nosig

        # hard_sal_map = gauss(hard_sal_map)

        # masked_cw_maps = torch.mul(cw_maps_refined, hard_sal_map)
        # masked_cw_maps = masked_cw_maps + cw_maps_refined  # TODO: comp_self_res
        # pred_comp_logits = self.spatial_pooling.spatial(masked_cw_maps) ## 2
        masked_cw_maps = torch.mul(cw_maps, hard_sal_map)
        masked_cw_maps = masked_cw_maps + cw_maps  # TODO: comp_self_res
        pred_comp_logits = self.spatial_pooling.spatial(masked_cw_maps) ## 1

        sal_map = self.to_attention_size(hard_sal_map) # hard_sal_map is already the multiplication of the two
        # sal_map = torch.sigmoid(sal_map)

        # return pred_logits, F.softmax(ori_logits, -1), torch.sigmoid(sal_map)
        # return pred_logits, pred_comp_logits, torch.clamp(sal_map, min=0.0, max=1.0)
        return pred_comp_logits, sal_map, self.to_attention_size(obj_att_maps), att_scores #, gaussian, gs_map
        # return pred_logits, pred_comp_logits, sal_map #, gaussian, gs_map
        # return pred_logits, pred_comp_logits, torch.sigmoid(sal_map)

