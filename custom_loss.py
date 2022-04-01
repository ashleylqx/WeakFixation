
import torch
import torch.nn.functional as F


# *** default ***
class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = torch.softmax(x, dim=-1)
        ind = torch.ones_like(x, requires_grad=True)
        ind = torch.div(ind , ind.sum(-1, keepdim=True))

        pos_mask = torch.nn.ReLU()((x - x.mean(-1, keepdim=True).expand_as(x)))
        pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (x.var(-1, keepdim=True)+1e-8)) * 1e3)

        # saliency value of region
        pos_x = torch.mul(pos_mask, x).sum(-1, keepdim=True)
        neg_x = torch.mul((1.-pos_mask), x).sum(-1, keepdim=True)
        p = torch.cat([pos_x, neg_x], dim=-1)

        # pixel percentage of region
        pos_ind = torch.mul(pos_mask, ind).sum(-1, keepdim=True)
        neg_ind = torch.mul((1. - pos_mask), ind).sum(-1, keepdim=True)
        ratio_w = torch.cat([pos_ind, neg_ind], dim=-1)

        b = F.softmax(ratio_w, dim=-1) * F.log_softmax(p, dim=-1)
        b = -1.0 * b.sum(dim=-1)
        return b.mean()




