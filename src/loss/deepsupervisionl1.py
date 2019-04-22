import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionL1(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(DeepSupervisionL1, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, srs, hr):
        total_loss = 0
        length = len(srs)

        loss_exits = []
        for sr in srs:
            loss_exits.append(F.l1_loss(sr, hr, size_average=self.size_average,
                                        reduce=self.reduce))
            total_loss += loss_exits[-1]

        loss_exits.insert(0, total_loss / length)
        return loss_exits
