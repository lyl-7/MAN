from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def make_model(args, parent=False):
    return MAN(args)

"""
Gumbel Softmax Sampler
Requires 2D input [batchsize, number of categories]
Does not support sinlge binary category. Use two dimensions with softmax instead.
"""
class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(-1)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = torch.log(logits) + Variable(gumble_samples_tensor)
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)
        return soft_samples

    def forward(self, logits, temp=1):
        return self.gumbel_softmax_sample(logits, temperature=temp)


# Mixer-order Channel Attention (MOCA)
class MOCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MOCA, self).__init__()
        # global average pooling: feature --> point
        self.num_gates = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channel, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Conv2d(16, self.num_gates, kernel_size=1)
        self.gs = GumbleSoftmax()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # channel downscale
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),  # channel upscale
            nn.Sigmoid()
        )

    def forward(self, x):  # Equation(13) x:Xg,b y:Rg,b(Xg,b)
        batch_size, C, H, W = x.shape
        w =self.avg_pool(x)
        w = F.relu(self.fc1(w))
        w = self.fc2(w)
        # Sample from Gumble Module
        gates = self.gs(w, temp=1) #(batch_size,num_gates)
        gates = gates.view(batch_size,self.num_gates)
        _, max_value_indexes = gates.data.max(1, keepdim=True)
        zeros= torch.zeros_like(gates)
        gates = zeros.scatter(1,max_value_indexes,1)
        gates = gates.transpose(0,1)
        gates = gates.contiguous().view(batch_size*self.num_gates,1)

        #y1 = self.avg_pool(x)
        #y1 = y1.view(batch_size,C)

        y2 = torch.var(x.view(batch_size, C, H * W), 2)  # (batch_size,C)
        y2 = torch.sqrt(y2)

        x1 = x.view(batch_size, C, H * W)
        mean = torch.mean(x1, 2, True)  # (batch_size,C,1)
        diffs = x1 - mean  # (batch_size,C,H*W)
        std = torch.std(x1, 2, False, True)  # (batch_size,C,1)
        zscores = diffs / std  # (batch_size,C,H*W)
        skew = torch.mean(torch.pow(zscores, 3.0), 2)  # (batch_size,C,1)
        kur = torch.mean(torch.pow(zscores, 4.0), 2)  # (batch_size,C,1)

        y3 = skew

        y4 = kur

        y = torch.cat((y2, y3, y4), dim=0) #(batch_size*4,C)
        y = y*gates
        zeros =torch.zeros(batch_size,C).cuda()
        index = [i for i in range(batch_size)]
        batch_index = []
        for i in range(self.num_gates):
            batch_index = batch_index + index
        batch_index=torch.tensor(batch_index).cuda()#4*batch_size
        com = zeros.index_add_(0, batch_index, y)

        y = com.view(batch_size, C, 1, 1)
        y = self.conv_du(y.cuda())  # (batch_size,C,1,1)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(MOCA(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

#Mixer-order channel Attention Network (MAN)
class MAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MAN, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            RCAB(
                conv, n_feats, kernel_size, reduction, bias=True, bn=False,act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

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
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

