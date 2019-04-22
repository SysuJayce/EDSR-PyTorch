# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return MERDN(args)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            *[nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class MERDN(nn.Module):
    def __init__(self, args):
        super(MERDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64), }[args.RDNconfig]

        self.n_exits = args.n_exits
        if self.n_exits > self.D:
            self.n_exits = self.D
        assert self.D % self.n_exits == 0, 'RDBs must be divisible by n_exits.'

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize,
                                 padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
                                 stride=1)

        # Redidual dense blocks and dense feature fusion
        self.body = nn.ModuleList()
        for i in range(self.n_exits):
            temp_body = nn.ModuleList()
            for _ in range(self.D // self.n_exits):
                temp_body.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
            self.body.append(temp_body)

        # Global Feature Fusion
        m_body_last = []
        for i in range(self.n_exits):
            temp_body_last = nn.Sequential(*[
                nn.Conv2d((self.D // self.n_exits) * (i + 1) * G0, G0, 1,
                          padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)])
            m_body_last.append(temp_body_last)

        # define tail module: Up-sampling net
        m_tail = []
        for i in range(self.n_exits):
            if r == 2 or r == 3:
                temp_tail = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2,
                              stride=1), nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2,
                              stride=1)])
            elif r == 4:
                temp_tail = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2,
                              stride=1), nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2,
                              stride=1), nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2,
                              stride=1)])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")
            m_tail.append(temp_tail)

        self.body_last = nn.ModuleList()
        for temp_body_last in m_body_last:
            self.body_last.append(nn.Sequential(*temp_body_last))

        self.tail = nn.ModuleList()
        for temp_tail in m_tail:
            self.tail.append(nn.Sequential(*temp_tail))

        self.idx_exit = None

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        output = []
        body_out = x
        RDBs_out = []
        for i in range(self.n_exits):
            for j in range(self.D // self.n_exits):
                body_out = self.body[i][j](body_out)
                RDBs_out.append(body_out)
            gff_out = self.body_last[i](torch.cat(RDBs_out, 1))
            gff_out += f__1
            out = self.tail[i](gff_out)
            output.append(out)

        return output
