import torch.nn as nn

from model import common

url = {"r20f64": ""}


def make_model(args, parent=False):
    return MEVDSR(args)


class MEVDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MEVDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.url = url["r{}f{}".format(n_resblocks, n_feats)]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.idx_exit = None
        self.n_exits = args.n_exits
        if self.n_exits > n_resblocks - 2:
            self.n_exits = n_resblocks - 2
        assert (n_resblocks - 2) % self.n_exits == 0, '(n_resblocks - 2) must be divisible by n_exits.'

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv,
                in_channels,
                out_channels,
                kernel_size,
                bias=True,
                bn=False,
                act=act,
            )

        # define body module
        # m_body = []
        # m_body.append(basic_block(args.n_colors, n_feats, nn.ReLU(True)))
        # for _ in range(n_resblocks - 2):
        #     m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        # m_body.append(basic_block(n_feats, args.n_colors, None))
        #
        # self.body = nn.Sequential(*m_body)
        m_body_first = [basic_block(args.n_colors, n_feats, nn.ReLU(True))]
        m_body_mid = []
        for i in range(self.n_exits):
            temp_body = [basic_block(n_feats, n_feats, nn.ReLU(True)) for _ in
                         range((n_resblocks - 2) // self.n_exits)]
            m_body_mid.append(temp_body)
        m_body_last = [basic_block(n_feats, args.n_colors, None)]
        self.body_first = nn.Sequential(*m_body_first)
        self.body_mid = nn.ModuleList()
        for temp_body in m_body_mid:
            self.body_mid.append(nn.Sequential(*temp_body))
        self.body_last = nn.Sequential(*m_body_last)

    def forward(self, x):
        x = self.sub_mean(x)
        res = self.body_first(x)
        output = []
        body_out = res
        for i in range(self.n_exits):
            body_out = self.body_mid[i](body_out)
            res_out = self.body_last(body_out)
            res_out += x
            res_out = self.add_mean(res_out)
            output.append(res_out)
        # res = self.body(x)
        # res += x
        # x = self.add_mean(res)

        # if self.training:
        #     return output
        # else:
        #     return output[-1]

        return output

    def set_exit(self, idx_exit):
        self.idx_exit = idx_exit
