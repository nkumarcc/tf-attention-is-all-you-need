import pdb
import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, ignore_index, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pdb.set_trace()
        pred = pred.log_softmax(dim=self.dim)
        non_pad_mask = target.ne(self.ignore_index)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim).masked_select(non_pad_mask))