from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class DeepSupervisionL1(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(DeepSupervisionL1, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, srs, hr):
        total_loss = 0
        length = len(srs)
        for sr in srs:
            total_loss += F.l1_loss(sr, hr, size_average=self.size_average, reduce=self.reduce)

        return total_loss / length