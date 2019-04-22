# -*- coding: utf-8 -*-
# @Time         : 2019-04-22 10:39
# @Author       : Jayce Wong
# @ProjectName  : EDSR-PyTorch
# @FileName     : deepsupervisionl2.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce


import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionL2(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(DeepSupervisionL2, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, srs, hr):
        total_loss = 0
        length = len(srs)
        loss_exits = []
        for sr in srs:
            loss_exits.append(F.mse_loss(sr, hr, size_average=self.size_average,
                                         reduce=self.reduce))
            total_loss += loss_exits[-1]

        loss_exits.insert(0, total_loss / length)

        return loss_exits
