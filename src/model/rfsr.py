# -*- coding: utf-8 -*-
# @Time         : 2019-03-04 16:36
# @Author       : Jayce Wong
# @ProjectName  : EDSR-PyTorch
# @FileName     : rfsr.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

""" Fractal Model - per sample drop path """
import torch.nn as nn
from model import common


def make_model(args):
    return RFSR(args)


# noinspection PyBroadException
class RFSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        """ FractalNet
        Args:
            - data_shape: (C, H, W, n_classes). e.g. (3, 32, 32, 10) - CIFAR 10.
            - n_columns: the number of columns
            - init_channels: the number of out channels in the first block
            - p_ldrop: local drop prob
            - dropout_probs: dropout probs (list)
            - gdrop_ratio: global droppath ratio
            - gap: pooling type for last block
            - init: initializer type
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - consist_gdrop
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [common.ResFractalBlock(args.n_columns, n_feats, n_feats)
                  for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                  conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except:
                    if name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(
                            name))
