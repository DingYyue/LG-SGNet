import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    # weight.size ->  (out_channels, input_channels/groups, kernel_size[0], kernel_size[1])
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        # 使用kaiming正态分布初始化卷积层参数，通过创建随机矩阵显式创建权重，则应进行设置mode=‘fan_out’
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(TemporalAttention, self).__init__()
        self.conv_ta = nn.Conv2d(in_channels, 1, kernel_size=(5, 1), padding=(2, 0))
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=-1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        # N, C, T, V -> N, C, T, 1
        att = self.sigmoid(self.conv_ta(avg_out))   # N, 1, T, 1
        x = x * att + x
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class TemporalSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(TemporalSelfAttention, self).__init__()

        # Temporal Self-Attention Module
        self.out_channels = out_channels
        self.stride = stride
        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1))
        self.tan = nn.Tanh()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        conv_init(self.conv_q)
        conv_init(self.conv_k)
        conv_init(self.conv_v)
        conv_init(self.conv_1)
        bn_init(self.bn1, 1)
        bn_init(self.bn2, 1)
        # bn_init(self.bn3, 1e-6)

    def forward(self, x):
        # TSA分支
        x_tsa = x
        v = self.conv_v(x_tsa)  # N*M, C, T, V
        v = v.permute(0, 3, 2, 1).contiguous()  # N*M, V, T, C
        q = (self.conv_q(x_tsa)).permute(0, 3, 2, 1).contiguous()  # N*M, C, T, V ->  N*M, V, T, C
        k = (self.conv_k(x_tsa)).permute(0, 3, 2, 1).contiguous()  # N*M, C, T, V ->  N*M, V, T, C
        att_t = (torch.einsum('nvsc,nvtc->nvst', q, k)) / math.sqrt(self.out_channels)  # N*M, V, T, T
        att_t = self.tan(att_t)  # N, V, T, T
        out_sa = torch.einsum('nvst,nvtc->nvsc', att_t, v)  # N, V, T, C
        out_sa = out_sa.permute(0, 3, 2, 1).contiguous()  # N*M, C, T, V
        if self.stride == 1:
            out_tsa = self.bn1(out_sa)  # N*M, C, T, V
        else:
            out_tsa = self.bn2(self.conv_1(out_sa))  # N*M, C, T/2, V
        return out_tsa


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        self.temporal_attention = TemporalAttention(in_channels)
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        # 模型除了两条带有dilation的分支外，还有一个Maxpool 和 普通的1*1卷积，所以+2
        branch_channels = out_channels // self.num_branches
        # 计算每个分支的平均维度，这样所有分支结果进行concat之后维度与out_channels一致
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])
        
        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels) 
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))
        
        self.branches.append(nn.Sequential(
                # nn.Conv2d(
                #     in_channels,
                #     branch_channels,
                #     kernel_size=1,
                #     padding=0),
                # nn.BatchNorm2d(branch_channels),
                # nn.ReLU(inplace=True),
                TemporalSelfAttention(
                    branch_channels,
                    branch_channels,
                    stride=stride)
        ))
        self.branches.append(nn.Sequential(
                # nn.Conv2d(
                #     in_channels,
                #     branch_channels,
                #     kernel_size=1,
                #     padding=0),
                # nn.BatchNorm2d(branch_channels),
                # nn.ReLU(inplace=True),
                TemporalSelfAttention(
                    branch_channels,
                    branch_channels,
                    stride=stride)
        ))  

        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.bn1 = nn.BatchNorm2d(branch_channels)
        self.bn2 = nn.BatchNorm2d(branch_channels)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # self.af = CBAMLayer(out_channels)

        # initialize
        self.apply(weights_init)
        bn_init(self.bn1, 1)
        bn_init(self.bn2, 1)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        x = self.temporal_attention(x)
        branch_outs = []
        x1 = self.branches[0][0](x)
        x1 = self.branches[0][1](x1)
        x1 = self.branches[0][2](x1)
        x1_1 = self.branches[0][3](x1)
        x1_2 = self.branches[4](x1)
        x1 = self.bn1(x1_1 + x1_2 * self.a)
        x2 = self.branches[1][0](x)
        x2 = self.branches[1][1](x2)
        x2 = self.branches[1][2](x2)
        x2_1 = self.branches[1][3](x2)
        x2_2 = self.branches[5](x2)
        x2 = self.bn2(x2_1 + x2_2 * self.b)
        x3 = self.branches[2](x)
        x4 = self.branches[3](x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        # 横向concat连接
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
            # //表示 除 之后向下取整，不大于商的最大整数
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # 1*1卷积改变维度，相当于T*N*C -> T*N*C/r
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv7 = nn.Conv2d(1, 1, kernel_size=1)
        # self.conv6 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
                # isinstance用来判断m与后面一个参数的类型（也就是是卷积还是bn操作）是否相同，再考虑进行初始化
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1, beta=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # x1, x2: N*C*T*V -> N*C/r*T*V -> N*C/r*V  先求T帧的平均值，再降低维度减少计算成本
        # x5 = torch.mean(x, dim=2, keepdim=True)
        # # x5 -> N*C*1*V   卷积层要求4D
        # q = self.conv5(x5)
        # # N*C/r*1*V
        # q = torch.mean(q, dim=2)或 q = q.flatten(2)
        # # N*C/r*V
        # k = self.conv6(x5)
        # # N*C/r*1*V
        # k = torch.mean(k, dim=2)或 k = k.flatten(2)
        # # N*C/r*V
        # norm_fact = 1 / math.sqrt(x5.shape[1])
        # att = torch.bmm(q.transpose(1, 2), k) * norm_fact
        # B = torch.softmax(att, dim=-1)

        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # 表示M()函数计算Xi与Xj之间的通道特定关系 x1: N*C*V*1 - N*C*1*V -> N*C/r*V*V
        x_attention_source = self.conv5(x)  # (N,rel_channels,T,V)
        N, C, T, V = x_attention_source.size()
        att_output_of_each_channels = []
        for i in range(x_attention_source.size()[1]):
            channel = x_attention_source[:, i]  # (N,T,V)
            channel = torch.mean(channel, 1, keepdim=True).unsqueeze(1)  # (N,T,V) -> (N,1,V) -> (N,1,1,V)
            channel = channel.transpose(2, 3)  # (N,1,V,1)
            channel = torch.repeat_interleave(channel, 25, 3)
            channel = channel.reshape(N, 1, V, 25)
            channel_q = self.conv6(channel).permute(0, 2, 3, 1).contiguous().flatten(2)  # (N,V,25,1) -> (N, V, 25)
            channel_k = self.conv7(channel).permute(0, 2, 3, 1).contiguous().flatten(2)  # (N,V,25,1) -> (N, V, 25)
            att = torch.matmul(channel_q, channel_k.transpose(-1, -2))  # N,V,V
            att_scores = att / math.sqrt(25.0)
            att_res = torch.softmax(att_scores, dim=-1)   # N, V, V
            att_output_of_each_channels.append(att_res)

        res = self.tanh(torch.stack(att_output_of_each_channels, 1))  # N C V V

        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0) + (self.conv4(res) if res is not None else 0) * beta       # N,C,V,V
        # conv4的作用：M（）函数计算相关性之后，通过1*1卷积将Q维度改为C_out，从而能与输入特征X的新维度C_out对应
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        # 爱因斯坦求和约定，对前 -> 后消除的字母进行求和，即对v进行求和
        return x1
        # 得到的Q矩阵与conv3卷积后输入特征相乘，得到输出特征

class unit_tcn(nn.Module):
    # 该普通时间卷积是为了给残差连接进行时间卷积
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
            # 将PA添加到参数列表中，送入优化器随着训练一起学习更新；后面from_numpy等 是将数据类型修改为float32的数组A，改为张量类型
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            # requires_grad是参不参与误差反向传播, 要不要计算梯度
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha, self.beta)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        # 自定义的异常处理
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25   一开始是 3  channels

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
            # 先将x的维度改为N,T,V,C,然后根据索引调换位置 -> N,C,T,V,在最后的索引位置上添加一个 M 维度
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

# if __name__ == '__main__':
#     a = torch.randn(1,3,64,25)
#     A = torch.ones(25)
#     ctr = Model(60)
#     ctr(a)
