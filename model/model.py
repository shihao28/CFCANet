import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.ops import FeaturePyramidNetwork
from typing import Union, List, Dict, Any, cast, Type, Callable, Optional, Tuple
from torch import Tensor
from collections import namedtuple, OrderedDict
import warnings
import re
import math

from .module import *


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
                relu=True, bn=True, bias=False, isInvolution=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Separable conv
        # if kernel_size==3:
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, dilation, in_planes, bias=False),
        #         nn.Conv2d(in_planes, out_planes, 1, 1, bias=bias)
        #     )
        # else:
        #     self.conv = nn.Conv2d(
        #         in_planes, out_planes, kernel_size=kernel_size,stride=stride, 
        #         padding=padding, dilation=dilation, groups=groups, bias=bias
        #         )
        if isInvolution:
            self.conv = Involution(in_planes, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        # self.relu = nn.SiLU(inplace=True) if relu else None 
        # self.SE = SE(16, out_planes)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        # x = self.SE(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MultiHeadedAttentionMini(nn.Module):
    
    def __init__(
        self, in_model_dimension, out_model_dimension, number_of_heads,
        dropout_probability, log_attention_weights):
        super().__init__()
        assert out_model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(out_model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = nn.Linear(in_model_dimension, out_model_dimension * 3)
        self.out_projection_net = nn.Linear(out_model_dimension, out_model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.softmax = nn.Softmax(dim=-1)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

    def attention(self, query, key, value, mask):
        scores = query.matmul(key.transpose(2, 3)) / self.head_dimension**0.5  # (B, H, S, S)

        # Optionally mask tokens whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        attention_weights = self.softmax(scores)

        attention_weights = self.attention_dropout(attention_weights)

        intermediate_token_representations = attention_weights.matmul(value)  # (B, H, S, D)

        return intermediate_token_representations, attention_weights

    def forward(self, x, mask):
        B, S, D = x.size()

        # Step 1: Input linear projection
        x = self.qkv_nets(x)  # (B, S, 3*D) where D=out_model_dimension
        x = x.reshape(B, S, self.number_of_heads, 3*self.head_dimension).transpose(1, 2)
        query, key, value = x.chunk(3, -1) # (B, H, S, D)

        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        # Potentially, for visualization purposes, log the attention weights, turn off during training though!
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        reshaped = intermediate_token_representations.transpose(1, 2).reshape(B, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations


class SpatialAttention(nn.Module):
    def __init__(self, x_C_in, x_C_out):
        super(SpatialAttention, self).__init__()
        self.proj_in = nn.Conv2d(x_C_in, x_C_out, 1)
        self.proj_out = nn.Linear(x_C_out, x_C_out)
    
    def forward(self, x, part_emb):
        x = self.proj_in(x)
        x = nn.Flatten(2)(x).transpose(1, 2)
        attention_mask = nn.Softmax(1)(x.bmm(part_emb))  # nn.Sigmoid(), nn.Softmax(1)
        x = x + x * attention_mask
        x = self.proj_out(x).transpose(1, 2)
        return x


class SpatialAttentionv1(nn.Module):
    def __init__(self, x_C_in, part_emb_C_in, C_out):
        super(SpatialAttentionv1, self).__init__()
        self.x_proj_in = nn.Conv2d(x_C_in, C_out, 1)
        self.part_emb_proj_in = nn.Conv1d(part_emb_C_in, C_out, 1)
        self.proj_out = nn.Linear(C_out, C_out)
    
    def forward(self, x, part_emb):
        x = self.x_proj_in(x)
        part_emb = self.part_emb_proj_in(part_emb)
        x = nn.Flatten(2)(x).transpose(1, 2)
        attention_mask = nn.Softmax(1)(x.bmm(part_emb))  # sigmoid
        x = x + x * attention_mask
        x = self.proj_out(x).transpose(1, 2)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, C_in, r):
        super(ChannelAttention, self).__init__()
        self.se = nn.Sequential(
            nn.Conv2d(C_in, C_in//r, 1),
            nn.ReLU(),
            nn.Conv2d(C_in//r, C_in),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x_c = nn.AdaptiveAvgPool2d(1)(x)
        x_c = self.se(x_c)
        x = x + x * x_c
        return x


class SE(nn.Module):
    def __init__(self, C_in, r):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_in, C_in//r, 1),
            nn.ReLU(),
            nn.Conv2d(C_in//r, C_in, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x_c = self.se(x)
        x = x * x_c
        return x


class BAM(nn.Module):
    def __init__(self, r, C):
        super().__init__()
        self.r = r
        self.C = C
        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.bn1 = nn.BatchNorm1d(self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)

        self.conv1 = nn.Conv2d(self.C, self.C // self.r, kernel_size=1,
                                stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(self.C // self.r)
        self.conv2 = nn.Conv2d(self.C // self.r, self.C // self.r, kernel_size=3,
                                stride=1, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm2d(self.C // self.r)
        self.conv3 = nn.Conv2d(self.C // self.r, self.C // self.r, kernel_size=3,
                                stride=1, padding=4, dilation=4)
        self.bn4 = nn.BatchNorm2d(self.C // self.r)
        self.conv4 = nn.Conv2d(self.C // self.r, 1, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_c = self.globalavgpool(x)
        x_c = x_c.view((batch_size, -1))
        x_c = self.dense1(x_c)
        x_c = self.bn1(x_c)
        x_c = nn.ReLU(inplace=True)(x_c)
        x_c = self.dense2(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))

        x_s = self.conv1(x)
        x_s = self.bn2(x_s)
        x_s = nn.ReLU(inplace=True)(x_s)
        x_s = self.conv2(x_s)
        x_s = self.bn3(x_s)
        x_s = nn.ReLU(inplace=True)(x_s)
        x_s = self.conv3(x_s)
        x_s = self.bn4(x_s)
        x_s = nn.ReLU(inplace=True)(x_s)
        x_s = self.conv4(x_s)
        x_cs = nn.Sigmoid()(x_c + x_s)
        x_cs = x * x_cs
        x = x + x_cs
        return x


class CBAM(nn.Module):
    def __init__(self, r, C):
        super().__init__()
        self.r = r
        self.C = C
        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.globalmaxpool = nn.AdaptiveMaxPool2d((1,1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size=7,
            stride=1, padding=3)
        # self.conv1 = Involution()
        self.bn1 = nn.BatchNorm2d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_c_avg = self.globalavgpool(x)
        x_c_avg = x_c_avg.view((batch_size, -1))
        x_c_avg = self.dense1(x_c_avg)
        x_c_avg = nn.ReLU(inplace=True)(x_c_avg)
        x_c_avg = self.dense2(x_c_avg)
        x_c_max = self.globalmaxpool(x)
        x_c_max = x_c_max.view((batch_size, -1))
        x_c_max = self.dense1(x_c_max)
        x_c_max = nn.ReLU(inplace=True)(x_c_max)
        x_c_max = self.dense2(x_c_max)
        x_c = x_c_avg + x_c_max
        x_c = nn.Sigmoid()(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))
        x_c = x * x_c

        x_s_avg = x_c.mean(axis=1, keepdim=True)
        x_s_max = x_c.max(axis=1, keepdim=True)[0]
        x_s = torch.cat((x_s_avg, x_s_max), axis=1)
        x_s = self.conv1(x_s)
        x_s = self.bn1(x_s)
        x_s = nn.Sigmoid()(x_s)
        # two lines below are modified
        x_cs = x_c * x_s
        x = x + x_cs
        return x


class CrossScaleAttention(nn.Module):
    def __init__(self, channel):
        super(CrossScaleAttention, self).__init__()
        self.project_out = nn.Linear(512, 2048)

    def forward(self, x3, x4, x5):
        x3 = x3.transpose(0, 1)
        x4 = x4.transpose(0, 1)
        x5 = x5.transpose(0, 1)

        x5 = x5 + x5.bmm(nn.Softmax(1)(x5.transpose(1, 2).bmm(x4)))
        x5 = x5 + x5.bmm(nn.Softmax(1)(x5.transpose(1, 2).bmm(x3)))
        x5 = self.project_out(x5)
        return x5


class SPP(nn.Module):
    # Modified from common.py in yolov5 repo
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1),
            nn.BatchNorm2d(c_),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_*(len(k) + 1), c2, 1, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        # https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    # https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class MultiScaleSE(nn.Module):
    def __init__(self, C, r, d, g):
        super(MultiScaleSE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, padding=d[0], dilation=d[0], groups=g),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, padding=d[1], dilation=d[1], groups=g),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, padding=d[2], dilation=d[2], groups=g),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C//r, 1),
            nn.ReLU(),
            nn.Conv2d(C//r, C, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        x_conv3 = self.conv3(x)
        x_sum = x_conv1 + x_conv2 + x_conv3
        x_se = self.se(x_sum)
        x_conv1 = x_conv1 * x_se
        x_conv2 = x_conv2 * x_se
        x_conv3 = x_conv3 * x_se
        x = x_conv1 + x_conv2 + x_conv3

        return x


class MultiScaleSEv1(nn.Module):
    def __init__(self, C, r, d, g):
        super(MultiScaleSEv1, self).__init__()
        self.d_len = len(d)
        self.g = g
        # one branch can be max-pooling -> Upsample
        # try dilation 1, 3, 5, 7
        self.conv1 = nn.Sequential(
            nn.Conv2d(C, C//2 if self.d_len == 2 else C//4, 3, 1, padding=d[0], dilation=d[0], groups=g),
            # nn.BatchNorm2d(C),
            # nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(C, C//2 if self.d_len == 2 else C//4, 3, 1, padding=d[1], dilation=d[1], groups=g),
            # nn.BatchNorm2d(C),
            # nn.ReLU()
        )
        if self.d_len == 4:
            self.conv3 = nn.Sequential(
                nn.Conv2d(C, C//4, 3, 1, padding=d[2], dilation=d[2], groups=g),
                # nn.BatchNorm2d(C),
                # nn.ReLU()
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(C, C//4, 3, 1, padding=d[3], dilation=d[3], groups=g),
                # nn.AdaptiveAvgPool2d((1, 1)),  # Can change to conv
                # nn.Conv2d(C, C//4, 1, 1),
                # nn.BatchNorm2d(C//4),
                # nn.ReLU()
            )
        self.combine = nn.Sequential(
            # nn.Conv2d(C, C, 1, 1, groups=1),
            nn.Conv2d(C, C, 1, 1, groups=1 if g == 1 else 2 if self.d_len == 2 else 4),  # Can try conv3
            # nn.BatchNorm2d(C),
            # nn.ReLU()
        )
        # can try mha-like channel attention
        # reduce_layer_planes = max(C * 4 // 8, 64)
        # r = C // reduce_layer_planes
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # eca, padding='circular', use kernel 7
            nn.Conv2d(C, C//r, 1),
            nn.ReLU(),
            nn.Conv2d(C//r, C, 1),
            nn.Sigmoid()
        )
        # self.eca = ECA(C, 7)
    
    def forward(self, x):
        # x = x + self.se(x)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        if self.d_len == 2:
            x_combine = torch.cat([x_conv1, x_conv2], 1)
            x_combine = channel_shuffle(x_combine, 1 if self.g == 1 else 4)
        else:
            x_conv3 = self.conv3(x)
            x_conv4 = self.conv4(x)
            # x_conv4 = F.interpolate(self.conv4(x), size=x.size()[2:], mode='bilinear', align_corners=True) 
            x_combine = torch.cat([x_conv1, x_conv2, x_conv3, x_conv4], 1)
            # x_combine = torch.cat([
            #     x_conv1, x_conv1 + x_conv2,
            #     x_conv1 + x_conv2 + x_conv3,
            #     x_conv1 + x_conv2 + x_conv3 + x_conv4], 1)
            x_combine = channel_shuffle(x_combine, 1 if self.g == 1 else 4)
        x_combine = self.combine(x_combine)
        x_se = self.se(x_combine)
        x = x_combine * x_se
        # x = x_combine + x_combine * x_se
        # B, C, H, W = x_combine.size()
        # x_att = nn.Softmax(-1)(x_combine.flatten(2).bmm(x_combine.flatten(2).transpose(1, 2)))
        # x = x_att.bmm(x_combine.flatten(2)).reshape(B, C, H, W)

        return x


class MultiScaleSEv2(nn.Module):
    def __init__(
        self, C_in_low, C_in_high, C_out, r, d, g, msconv_low, dropout=0.,
        drop_block_rate=0., block_size=3, gamma_scale=1, drop_path_rate=0.,
        stride_low=True, is_inception=False):
        super(MultiScaleSEv2, self).__init__()
        self.d_len = len(d)
        # self.g = g
        self.msconv_low = msconv_low
        stride = 2 if stride_low else 1
        # to set msconv_low=False
        # # apply sigmoid on multi-scale low-level feature maps, like FEB in cross-part CNN
        if msconv_low:
            self.conv1_low = nn.Sequential(
                nn.Conv2d(C_in_low, C_out//2, 3, stride, padding=0 if is_inception else d[0], dilation=d[0], groups=g),
                # nn.BatchNorm2d(C_out//2),
                # nn.ReLU()
            )
            if self.d_len >= 2:
                self.conv2_low = nn.Sequential(
                    nn.Conv2d(C_in_low, C_out//2, 3, stride, padding=d[0] if is_inception else d[1], dilation=d[1], groups=g),
                    # nn.BatchNorm2d(C_out//2),
                    # nn.ReLU()
                )
            if self.d_len >= 3:
                self.conv3_low = nn.Sequential(
                    nn.Conv2d(C_in_low, C_out//2, 3, stride, padding=d[1] if is_inception else d[2], dilation=d[2], groups=g),
                    # nn.BatchNorm2d(C_out//2),
                    # nn.ReLU()
                )
            if self.d_len >= 4:
                self.conv4_low = nn.Sequential(
                    nn.Conv2d(C_in_low, C_out//2, 3, stride, padding=d[2] if is_inception else d[3], dilation=d[3], groups=g),
                    # nn.BatchNorm2d(C_out//2),
                    # nn.ReLU()
                )
            # self.bn_low = nn.BatchNorm2d(C_out//2)
            # self.conv_depthwise = nn.Conv2d(C_out//2, C_out//2, 2, 2, groups=C_out//2)
            # self.conv4_low = nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Conv2d(C_in_low, C_out//2, 1, 1),
            #     # nn.MaxPool2d(2, 2),
            #     # nn.BatchNorm2d(C_out//2),
            #     # nn.ReLU()
            #     )
        else:
            self.conv1_low = nn.Conv2d(C_in_low, C_out//2, 3, stride, padding=0 if is_inception else 1, groups=1)

        self.conv1_high = nn.Sequential(
            nn.Conv2d(C_in_high, C_out//2, 3, 1, padding=d[0], dilation=d[0], groups=g),
            # nn.BatchNorm2d(C_out//2),
            # nn.ReLU()
        )
        if self.d_len >= 2:
            self.conv2_high = nn.Sequential(
                nn.Conv2d(C_in_high, C_out//2, 3, 1, padding=d[1], dilation=d[1], groups=g),
                # nn.BatchNorm2d(C_out//2),
                # nn.ReLU()
            )
        if self.d_len >= 3:
            self.conv3_high = nn.Sequential(
                nn.Conv2d(C_in_high, C_out//2, 3, 1, padding=d[2], dilation=d[2], groups=g),
                # nn.BatchNorm2d(C_out//2),
                # nn.ReLU()
            )
        if self.d_len >= 4:
            self.conv4_high = nn.Sequential(
                nn.Conv2d(C_in_high, C_out//2, 3, 1, padding=d[3], dilation=d[3], groups=g),
                # nn.BatchNorm2d(C_out//2),
                # nn.ReLU()
            )
        # self.bn_high = nn.BatchNorm2d(C_out//2)
        self.combine = nn.Sequential(
            # Use depthwise separable conv
            nn.Conv2d(C_out, C_out, 1, 1, groups=1),
            # nn.BatchNorm2d(C_out),
            # nn.ReLU(),
            # nn.Conv2d(C_out, C_out, 3, 1, 1, groups=C_out),
            # nn.Conv2d(C, C, 1, 1, groups=1 if g == 1 else 2 if self.d_len == 2 else 4),  # Can try conv3
            nn.BatchNorm2d(C_out),
            nn.ReLU(),
        )
        # can try mha-like channel attention
        # reduce_layer_planes = max(C * 4 // 8, 64)
        # r = C // reduce_layer_planes
        # self.se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),  # eca, padding='circular', use kernel 7
        #     nn.Conv2d(C_out, C_out//r, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(C_out//r, C_out, 1),
        #     nn.Sigmoid()
        # )
        # # self.eca = ECA(C, 7)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        self.dropblock = None
        if drop_block_rate > 0:
            self.dropblock = DropBlock2d(drop_block_rate, block_size, gamma_scale)
        self.droppath = None
        if drop_path_rate > 0:
            self.droppath = DropPath(drop_path_rate)
    
    def forward(self, x_low, x_high):
        x_high_ori = x_high.clone()
        if self.msconv_low:
            x_low_tmp = self.conv1_low(x_low)
            if self.d_len >= 2:
                x_low_tmp = x_low_tmp + self.conv2_low(x_low)
            if self.d_len >= 3:
                x_low_tmp = x_low_tmp + self.conv3_low(x_low)
            if self.d_len >= 4:
                x_low_tmp = x_low_tmp + self.conv4_low(x_low)
            x_low = x_low_tmp
                # F.interpolate(self.conv4_low(x_low), size=x_high.size()[2:], mode='bilinear', align_corners=True)
            # x_low = nn.ReLU()(self.bn_low(x_low))
            # x_low = nn.AvgPool2d(2)(x_low)
            # x_low = self.conv_depthwise(x_low)
            
        else:
            x_low = self.conv1_low(x_low)
        x_high_tmp = self.conv1_high(x_high)
        if self.d_len >= 2:
            x_high_tmp = x_high_tmp + self.conv2_high(x_high)
        if self.d_len >= 3:
            x_high_tmp = x_high_tmp + self.conv3_high(x_high)
        if self.d_len >= 4:
            x_high_tmp = x_high_tmp + self.conv4_high(x_high)
        x_high = x_high_tmp
        # x_high = nn.ReLU()(self.bn_high(x_high))
        x = torch.cat([x_low, x_high], 1)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.dropblock is not None:
            x = self.dropblock(x)
        if self.droppath is not None:
            x = self.droppath(x)
        x = self.combine(x)
        # x = x + x * self.se(x)
        # B, C, H, W = x.size()
        # x_flatten = nn.Flatten(2)(x)
        # attention_weight = nn.Softmax(-1)(x_flatten.bmm(x_flatten.transpose(1, 2)))
        # x = attention_weight.bmm(x_flatten)
        # x = x.view(B, C, H, W)
        # x = x_high_ori*nn.Sigmoid()(x)

        return x


class MultiScaleSEv3(nn.Module):
    def __init__(self, C_in_high, C_out, r, d, g):
        super(MultiScaleSEv3, self).__init__()
        # # apply sigmoid on multi-scale low-level feature maps, like FEB in cross-part CNN
        self.conv1_low = nn.Sequential(
            nn.Conv2d(C_in_high//2, C_out, 3, 2, padding=d[0], dilation=d[0], groups=g),
            # nn.BatchNorm2d(C_out//2),
            # nn.ReLU()
        )
        self.conv2_low = nn.Sequential(
            nn.Conv2d(C_in_high//2, C_out, 3, 2, padding=d[1], dilation=d[1], groups=g),
            # nn.BatchNorm2d(C_out//2),
            # nn.ReLU()
        )
        self.conv3_low = nn.Sequential(
            nn.Conv2d(C_in_high//2, C_out, 3, 2, padding=d[2], dilation=d[2], groups=g),
            # nn.BatchNorm2d(C_out//2),
            # nn.ReLU()
        )

        # self.combine = nn.Sequential(
        #     nn.Conv2d(C_out, C_out, 1, 1, groups=1),
        #     # nn.Conv2d(C, C, 1, 1, groups=1 if g == 1 else 2 if self.d_len == 2 else 4),  # Can try conv3
        #     # nn.BatchNorm2d(C_out),
        #     # nn.ReLU()
        # )
        # can try mha-like channel attention
        # reduce_layer_planes = max(C * 4 // 8, 64)
        # r = C // reduce_layer_planes
        # self.se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),  # eca, padding='circular', use kernel 7
        #     nn.Conv2d(C_out, C_out//r, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(C_out//r, C_out, 1),
        #     nn.Sigmoid()
        # )
        # self.eca = ECA(C, 7)
    
    def forward(self, x_low, x_high):
        x_low = self.conv1_low(x_low) + self.conv2_low(x_low) + self.conv3_low(x_low)
        x = x_high + x_high * nn.Sigmoid()(x_low)

        return x


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False,
            padding_mode='circular') 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y


class GlobalAtt(nn.Module):
    def __init__(self, size_in, C_in, size_out, C_out, r, g):
        super(GlobalAtt, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_in//r, 1, groups=g)
        self.unfold = torch.nn.Unfold(
            kernel_size=size_in//size_out, stride=size_in//size_out)
        self.mha_norm = nn.LayerNorm(C_in//r*(size_in//size_out)**2)
        self.mha = nn.MultiheadAttention(
                embed_dim=C_in//r*(size_in//size_out)**2, num_heads=4, dropout=0)
        self.dropout = nn.Dropout(0)
        self.ffn = nn.Sequential(
            nn.LayerNorm(C_in//r*(size_in//size_out)**2),
            nn.Linear(C_in//r*(size_in//size_out)**2, C_in//r*(size_in//size_out)**2),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(C_in//r*(size_in//size_out)**2, C_in//r*(size_in//size_out)**2),
            nn.Dropout(0),
        )
        self.fold = torch.nn.Fold(size_out, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.unfold(x).transpose(1, 2)
        x_norm = self.mha_norm(x)
        x = x + self.mha(x_norm, x_norm, x_norm)[0]
        x = self.dropout(x)
        x = x + self.ffn(x)
        x = self.fold(x.transpose(1, 2))
        return x


class GlobalAttv1(nn.Module):
    def __init__(self):
        super(GlobalAttv1, self).__init__()
        self.linear_proj1 = nn.Linear(256, 512)
        self.linear_proj2 = nn.Linear(512, 512)
        self.linear_proj3 = nn.Linear(1024, 512)
        self.linear_proj4 = nn.Linear(2048, 512)
        self.linear_proj = nn.Linear(512, 2048)
        

    def forward(self, x2, x3, x4, x5):
        x2 = nn.Unfold(1)(torch.nn.functional.interpolate(x2, size=(14, 14), mode='bilinear'))
        x3 = nn.Unfold(1)(torch.nn.functional.interpolate(x3, size=(14, 14), mode='bilinear'))
        x4 = nn.Unfold(1)(torch.nn.functional.interpolate(x4, size=(14, 14), mode='bilinear'))
        x5 = nn.Unfold(1)(x5)

        x2 = self.linear_proj1(x2.transpose(1, 2))
        x3 = self.linear_proj2(x3.transpose(1, 2))
        x4 = self.linear_proj3(x4.transpose(1, 2))
        x5 = self.linear_proj4(x5.transpose(1, 2))

        x = x5 + torch.bmm(torch.bmm(x2, x5.transpose(1, 2)), x5) +\
            torch.bmm(torch.bmm(x3, x5.transpose(1, 2)), x5) +\
                torch.bmm(torch.bmm(x4, x5.transpose(1, 2)), x5)
        x = self.linear_proj(x)
        x = nn.Fold(14, 1)(x.transpose(1, 2))

        return x


class LRAU(nn.Module):
    def __init__(self, k, n, C_in):
        super(LRAU, self).__init__()
        self.k = k  # number of attention masks
        self.n = n  # number of classes
        self.C_in = C_in
        self.w_s = nn.Conv2d(k*n, k*n, 1, 2)
        self.w_cs = nn.Conv2d(k*n, k, 1)
        self.w_ch = nn.Conv2d(C_in, k, 1)
        self.w_uh = nn.Conv2d(C_in, n, 1)
    
    def forward(self, h, last_s=None):
        # h is current feature maps
        # last_s is previous attention state
        B, C, H, W = h.size()
        if last_s is None:
            last_s = torch.zeros((B, self.k*self.n, H*2, W*2), device=h.device)
        last_s_prime = self.w_s(last_s)
        cs = self.w_cs(last_s_prime)
        ch = self.w_ch(h)
        cl = nn.Sigmoid()(cs + ch).unsqueeze(2)
        ul = self.w_uh(h).unsqueeze(1)
        cul = (cl * ul).view(B, self.n*self.k, H, W)
        sl = cul + last_s_prime
        
        return sl


class Combine(nn.Module):
    def __init__(self, C_in, C_out, spatial_size):
        super(Combine, self).__init__()
        self.combine = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1),
            nn.BatchNorm2d(C_out),
            nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(spatial_size)

    def forward(self, x3, x4, x5):
        x3 = self.avg_pool(x3)
        x4 = self.avg_pool(x4)
        x = torch.cat([x3, x4, x5], 1)
        x = self.combine(x)

        return x


class SimpleFPA(nn.Module):
    def __init__(self, in_planes, out_planes):
        """
        https://github.com/PRIS-CV/AP-CNN_Pytorch-master/blob/master/model/resnet50.py
        Feature Pyramid Attention
        :type channels: int
        """
        super(SimpleFPA, self).__init__()

        self.channels_cond = in_planes
        # Master branch
        self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

        # Global pooling branch
        self.conv_gpb = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)

        out = x_master + x_gpb

        return out


class PyramidFeatures(nn.Module):
    """https://github.com/PRIS-CV/AP-CNN_Pytorch-master/blob/master/model/resnet50.py"""
    """Feature pyramid module with top-down feature pathway"""
    def __init__(self, B2_size, B3_size, B4_size, B5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = SimpleFPA(B5_size, feature_size)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(B4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(B3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P2_1 = nn.Conv2d(B2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        B2, B3, B4, B5 = inputs

        P5_x = self.P5_1(B5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(B4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(B3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = F.interpolate(P3_x, scale_factor=2)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(B2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        return [P2_x, P3_x, P4_x, P5_x]


class Involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(channels // reduction_ratio, kernel_size**2 * self.groups, 1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)

        return out


def drop_block_2d(
        x, drop_prob: float = 0.1, block_size: int = 7,  gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(
        x: torch.Tensor, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    block_mask = torch.empty_like(x).bernoulli_(gamma)
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.empty_like(x).normal_()
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-6)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ 
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/drop.py
    DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False,
                 inplace=False,
                 batchwise=False,
                 fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/drop.py
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class CGAttention(nn.Module):
    def __init__(self, C_in_low, C_in_high, C_out, num_heads=8, g=16):
        super(CGAttention, self).__init__()
        # MHSA
        self.C_in_low = C_in_low
        self.C_in_high = C_in_high
        self.C_out = C_out
        self.num_heads = num_heads
        self.head_dim = C_out // num_heads

        self.conv_q = nn.Sequential(
            nn.Conv2d(C_in_low, C_out, kernel_size=3, stride=2, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(C_out)
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(C_in_high, C_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=g),
            nn.BatchNorm2d(C_out)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(C_in_high, C_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=g),
            nn.BatchNorm2d(C_out)
        )
        self.proj = nn.Conv2d(C_out, C_out, 1)

        # FFN
        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_out, C_out*4, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(C_out*4),
        )
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_out*4, C_out*4, kernel_size=3, stride=1, padding=1, groups=C_out*4),
            nn.GELU(),
            nn.BatchNorm2d(C_out*4)
        )
        self.ffn3 = nn.Sequential(
            nn.Conv2d(C_out*4, C_out, 1, 1),
            nn.BatchNorm2d(C_out),
        )
    
    def forward(self, x_low, x_high):
        B = x_low.size(0)
        # MHSA
        q = self.conv_q(x_low).reshape(B, self.C_out, -1).transpose(1, 2)
        k = self.conv_k(x_high).reshape(B, self.C_out, -1).transpose(1, 2)
        v = self.conv_v(x_high).reshape(B, self.C_out, -1).transpose(1, 2)
        q = q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention_weights = nn.Softmax(-1)(torch.matmul(q, k) * self.head_dim**-0.5)
        x = torch.matmul(attention_weights, v).permute(0, 2, 1, 3)
        spatial_size = int(x.size(1)**0.5)
        x = x.reshape(B, self.C_out, spatial_size, spatial_size)
        x = self.proj(x)

        # FFN
        x = self.ffn1(x)
        x = self.ffn2(x)
        x = self.ffn3(x)

        return x


""" VGG16 """
class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        use_my_model = False
    ) -> None:
        super(VGG, self).__init__()
        self.use_my_model = use_my_model
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._initialize_weights()
        
        if use_my_model:
            self.ms2 = MultiScaleSEv2(
                128, 256, 256, 16, (1,2,3,4), 32, msconv_low=True,
                drop_block_rate=0., block_size=7, gamma_scale=0.0625,
                drop_path_rate=0.)
            self.ms3 = MultiScaleSEv2(
                256, 512, 512, 16, (1,2,3,4), 32, msconv_low=True,
                drop_block_rate=0., block_size=5, gamma_scale=0.25,
                drop_path_rate=0.)
            self.ms4 = MultiScaleSEv2(
                512, 512, 512, 16, (1,2,3,4), 32, msconv_low=True,
                drop_block_rate=0., block_size=3, gamma_scale=1,
                drop_path_rate=0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.features(x)

        x1 = self.features[:5](x)
        x2 = self.features[5:10](x1)
        x3 = self.features[10:17](x2)
        x4 = self.features[17:24](x3)
        x5 = self.features[24:](x4)

        if self.use_my_model:
            x3 = self.ms2(x2, x3)
            x4 = self.ms3(x3, x4)
            x5 = self.ms4(x4, x5)
        
        x_final = self.avgpool(x5)
        x_final = torch.flatten(x_final, 1)
        logits = self.classifier(x_final)
        return x_final, logits

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # model.load_state_dict(state_dict)

        # 1. filter out unnecessary keys
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and not k.startswith('classifier'):
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


""" Inceptionv3 """
InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs

def inception_v3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> "Inception3":
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        kwargs['init_weights'] = False  # we are loading weights from a pretrained model
        model = Inception3(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                              progress=progress)

        pretrained_dict = dict()
        model_dict = model.state_dict()
        for k, v in state_dict.items():
            if k in model_dict and not k.startswith('classifier') and not k.startswith('fc') and "Aux" not in k:
                pretrained_dict[k] = v
        model_dict.update(pretrained_dict) 
        
        model.load_state_dict(model_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None
        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None
    ) -> None:
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux_v3
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # Groups for convolution is 16 instead of 32 for Inceptionv3
        # bcuz 288 is not divisible by 32
        self.ms2 = MultiScaleSEv2(
            192, 288, 288, 16, (1,2,3,4), 16, msconv_low=True,
            drop_block_rate=0., block_size=7, gamma_scale=0.0625,
            drop_path_rate=0., is_inception=True)
        self.ms3 = MultiScaleSEv2(
            288, 768, 768, 16, (1,2,3,4), 16, msconv_low=True,
            drop_block_rate=0., block_size=5, gamma_scale=0.25,
            drop_path_rate=0., is_inception=True)
        self.ms4 = MultiScaleSEv2(
            768, 2048, 2048, 16, (1,2,3,4), 16, msconv_low=True,
            drop_block_rate=0., block_size=3, gamma_scale=1,
            drop_path_rate=0., is_inception=True)
        
    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x1 = self.maxpool1(x)
        
        # N x 64 x 73 x 73
        x2 = self.Conv2d_3b_1x1(x1)
        # N x 80 x 73 x 73
        x2 = self.Conv2d_4a_3x3(x2)
        
        # N x 192 x 71 x 71
        x3 = self.maxpool2(x2)
        # N x 192 x 35 x 35
        x3 = self.Mixed_5b(x3)
        # N x 256 x 35 x 35
        x3 = self.Mixed_5c(x3)
        # N x 288 x 35 x 35
        x3 = self.Mixed_5d(x3)
        
        # N x 288 x 35 x 35
        x4 = self.Mixed_6a(x3)
        # N x 768 x 17 x 17
        x4 = self.Mixed_6b(x4)
        # N x 768 x 17 x 17
        x4 = self.Mixed_6c(x4)
        # N x 768 x 17 x 17
        x4 = self.Mixed_6d(x4)
        # N x 768 x 17 x 17
        x4 = self.Mixed_6e(x4)
        
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        
        # N x 768 x 17 x 17
        x5 = self.Mixed_7a(x4)
        # N x 1280 x 8 x 8
        x5 = self.Mixed_7b(x5)
        # N x 2048 x 8 x 8
        x5 = self.Mixed_7c(x5)
        
        x3 = self.ms2(x2, x3)
        x4 = self.ms3(x3, x4)
        x5 = self.ms4(x4, x5)

        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x5)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux_v3(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux_v3, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


""" ResNet """
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) # nn.ReLU(inplace=True) nn.SiLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_block_rate: float = 0., block_size: int = 5, gamma_scale: float = 0.25,
        drop_path_rate: float=0.,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True) # nn.ReLU(inplace=True) nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # Refer link below on how SE is added into ResNet50
        # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py
        # self.se = SE(r = 16, C = planes * 4)

        # Refer link below for Official site from paper on how CBAM in included in ResNet50
        # https://github.com/Jongchan/attention-module/blob/master/MODELS/model_resnet.py
        # self.cbam = CBAM(r = 16, C = planes * 4)

        # Refer link below for Official site from paper on how ECA in included in ResNet50
        # https://github.com/BangguWu/ECANet/blob/master/models/eca_resnet.py
        # self.eca = eca_layer(channel = planes * 4, k_size = 3)

        # self.se = MultiScaleSE(C=planes * self.expansion, r=16, d=(1,2,3), g=32)

        # Set a kernel to 0
        self.drop_block = None
        if drop_block_rate > 0:
            self.drop_block = DropBlock2d(drop_block_rate, block_size=block_size, gamma_scale=gamma_scale)  
            # self.drop_block = DropBlock2d(drop_block_rate, block_size=5, gamma_scale=0.25)  # block 3
            # self.drop_block = DropBlock2d(drop_block_rate, block_size=3, gamma_scale=1.00)  # block 4      
        
        # Set a sample to 0
        self.drop_path = None
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.drop_block is not None:
            out = self.drop_block(out)

        # out = self.se(out)
        # out = self.eca(out)

        if self.drop_path is not None:
            out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out


class MSBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        r=4, d=[1, 2, 3, 4], g=4
    ) -> None:
        super(MSBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True) # nn.ReLU(inplace=True) nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.msse = MultiScaleSEv1(width, r, d, g)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.msse(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Original
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu =  nn.ReLU(inplace=True) # nn.ReLU(inplace=True) nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Original
        self.layer1 = self._make_layer(block, 64, layers[0], drop_block_rate=0, drop_path_rate=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       drop_block_rate=0, drop_path_rate=0)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       drop_block_rate=0, block_size=5, gamma_scale=0.25,
                                       drop_path_rate=0)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       drop_block_rate=0, block_size=3, gamma_scale=1.00,
                                       drop_path_rate=0)
        
        # Mine
        # self.layer1 = self._make_layer_muliscale(block, 64, layers[0])
        # self.layer2 = self._make_layer_muliscale(block, 128, layers[1], stride=2,
        #                                         dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer_muliscale(block, 256, layers[2], stride=2,
        #                                         dilate=replace_stride_with_dilation[1],
        #                                         previous_block_channel=256)
        # self.layer4 = self._make_layer_muliscale(block, 512, layers[3], stride=2, # 
        #                                         dilate=replace_stride_with_dilation[2],
        #                                         previous_block_channel=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.dropout = nn.Dropout(0.1)
        # self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
        # fpn_channel_out = 256
        # self.fpn = FeaturePyramidNetwork([512, 1024, 2048], fpn_channel_out)
        # self.fpn = PyramidFeatures(256, 512, 1024, 2048)

        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.cls3 = nn.Sequential(
        #     nn.Flatten(),            
        #     nn.BatchNorm1d(fpn_channel_out),
        #     # nn.Linear(fpn_channel_out, 256),
        #     # nn.BatchNorm1d(256),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(fpn_channel_out, num_classes)
        # )
        # self.cls4 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.BatchNorm1d(fpn_channel_out),
        #     # nn.Linear(fpn_channel_out, 256),
        #     # nn.BatchNorm1d(256),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(fpn_channel_out, num_classes)
        # )
        # self.cls5 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.BatchNorm1d(fpn_channel_out),
        #     # nn.Linear(fpn_channel_out, 256),
        #     # nn.BatchNorm1d(256),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(fpn_channel_out, num_classes)
        # )
        # self.cls_concat = nn.Sequential(
        #     nn.Flatten(),
        #     nn.BatchNorm1d(fpn_channel_out*3),
        #     # nn.Linear(fpn_channel_out*3, 256),
        #     # nn.BatchNorm1d(256),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(fpn_channel_out*3, num_classes)
        # )

        # self.apcnncrop = APCNNCropMine(224, 224)

        # self.lstm = nn.LSTM(fpn_channel_out, fpn_channel_out, batch_first=True, dropout=0, bidirectional=False)
        # self.cls_lstm = nn.Sequential(
        #     nn.Flatten(),
        #     nn.BatchNorm1d(fpn_channel_out),
        #     # nn.Linear(fpn_channel_out*3, 256),
        #     # nn.BatchNorm1d(256),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(fpn_channel_out, num_classes)
        # )
        
        # placeholder for the gradients
        # self.gradients = None
        
        # Try lstm, fpn, mha
        # self.mhsa3 = MultiHeadedAttentionMini(
        #     in_model_dimension=512, out_model_dimension=512,
        #     number_of_heads=8, dropout_probability=0, log_attention_weights=False
        # )
        # self.mhsa4 = MultiHeadedAttentionMini(
        #     in_model_dimension=1024, out_model_dimension=1024,
        #     number_of_heads=8, dropout_probability=0, log_attention_weights=False
        # )
        # self.mhsa5 = MultiHeadedAttentionMini(
        #     in_model_dimension=2048, out_model_dimension=2048,
        #     number_of_heads=8, dropout_probability=0, log_attention_weights=False
        # )

        # # self.lstm2 = nn.LSTM(256, 512, batch_first=True, dropout=0.2, bidirectional=False)
        # self.lstm3 = nn.LSTM(512, 512, batch_first=True, dropout=0.2, bidirectional=False)
        # self.lstm4 = nn.LSTM(1024, 512, batch_first=True, dropout=0.2, bidirectional=False)
        # self.lstm5 = nn.LSTM(2048, 512, batch_first=True, dropout=0.2, bidirectional=False)
        # # self.gru3 = nn.GRU(512, 512, batch_first=True, dropout=0.2, bidirectional=False)
        # # self.gru4 = nn.GRU(1024, 512, batch_first=True, dropout=0.2, bidirectional=False)
        # # self.gru5 = nn.GRU(2048, 512, batch_first=True, dropout=0.2, bidirectional=False)
        # self.conv1d = nn.Sequential(
        #     nn.Conv1d(3, 1, 1, 1, padding=0),  # try kernel size 3
        #     # nn.BatchNorm1d(1),
        #     nn.ReLU(inplace=True)
        # )
        # # self.att = SpatialAttentionv1(2048, 512, 2048)
        # self.att = SpatialAttention(2048, 512)
        # # self.channel_att = ChannelAttention(2048, 16)
        # # self.att = CrossScaleAttention(512)
        # self.fc = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.BatchNorm1d(512),
        #     nn.Flatten(),
        #     nn.Linear(512, num_classes)
        # )
        # self.mhsa = nn.MultiheadAttention()
        
        # placeholder for the gradients
        # self.gradients = None

        # self.linear_proj2 = nn.Sequential(
        #     nn.Conv2d(512, 512, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(14)
        #     )
        # self.linear_proj3 = nn.Sequential(
        #     nn.Conv2d(1024, 512, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(14)
        #     )
        # self.linear_proj4 = nn.Sequential(
        #     nn.Conv2d(2048, 512, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        #     )
        # self.convlstm = ConvLSTM(
        #     input_dim=512, hidden_dim=512//8, kernel_size=(3, 3),
        #     num_layers=1, batch_first=True, return_all_layers=False)
        # self.final_proj = nn.Sequential(
        #     nn.Conv2d(512//8, 2048, 1, 1),
        # )
        # self.fc = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.BatchNorm2d(2048),
        #     nn.Flatten(),
        #     nn.Linear(2048, num_classes)
        # )

        # self.se1 = MultiScaleSE(C=256, r=16, d=(1, 2, 3), g=32)
        # self.se2 = MultiScaleSE(C=512, r=16, d=(1, 2, 3), g=32)
        # self.se3 = MultiScaleSE(C=1024, r=16, d=(1, 2, 3), g=32)
        # self.se4 = MultiScaleSE(C=2048, r=16, d=(1, 2, 3), g=32)

        # self.se1 = MultiScaleSEv1(C=256, r=4, d=(1, 2, 3, 4), g=4)
        # self.se2 = MultiScaleSEv1(C=512, r=8, d=(1, 2, 3, 4), g=8)
        # self.se3 = MultiScaleSEv1(C=1024, r=16, d=(1, 2, 3, 4), g=16)
        # self.se4 = MultiScaleSEv1(C=2048, r=32, d=(1, 2, 3, 4), g=32)

        # self.ms2 = CGAttention(256, 512, 4)
        # self.ms1 = MultiScaleSEv2(
        #     64, 256, 256, 16, (1,2,3), 32, msconv_low=True,
        #     drop_block_rate=0., block_size=7, gamma_scale=0.0625,
        #     drop_path_rate=0., stride_low=False)
        self.ms2 = MultiScaleSEv2(
            256, 512, 512, 16, (1,2,3), 32, msconv_low=True,
            drop_block_rate=0., block_size=7, gamma_scale=0.0625,
            drop_path_rate=0.)
        self.ms3 = MultiScaleSEv2(
            512, 1024, 1024, 16, (1,2,3), 32, msconv_low=True,
            drop_block_rate=0., block_size=5, gamma_scale=0.25,
            drop_path_rate=0.)
        self.ms4 = MultiScaleSEv2(
            1024, 2048, 2048, 16, (1,2,3), 32, msconv_low=True,
            drop_block_rate=0., block_size=3, gamma_scale=1,
            drop_path_rate=0.)
        # self.ms2 = MultiScaleSEv2(
        #     256, 512, 192, 16, (1, 2, 3), 16, msconv_low=True,
        #     drop_block_rate=0., block_size=7, gamma_scale=0.0625,
        #     drop_path_rate=0.)
        # self.ms3 = MultiScaleSEv2(
        #     192, 1024, 384, 16, (1, 2, 3), 16, msconv_low=True,
        #     drop_block_rate=0., block_size=5, gamma_scale=0.25,
        #     drop_path_rate=0.)
        # self.ms4 = CGAttention(384, 2048, 768, 16)
        # self.ms2 = MultiScaleSEv2(256, 256, 256, 16, (1, 2, 3), 4, msconv_low=True)
        # self.ms3 = MultiScaleSEv2(256, 256, 256, 16, (1, 2, 3), 4, msconv_low=True)
        # self.ms4 = MultiScaleSEv2(256, 256, 256, 16, (1, 2, 3), 4, msconv_low=True)
        # self.se = SE(2048, 16)
        # Try lstm here
        # self.combine = Combine(512*3, 512*3, 7)
        # self.agg = nn.Sequential(
        #     nn.Conv2d(1024, 1024, 1),
        #     nn.BatchNorm2d(1024),
        #     # nn.ReLU()
        # )
        # self.se = SE(1024, 16)

        # self.ms2 = MultiScaleSEv3(512, 512, 16, (1, 2, 3), 32)
        # self.ms3 = MultiScaleSEv3(1024, 1024, 16, (1, 2, 3), 32)
        # self.ms4 = MultiScaleSEv3(2048, 2048, 16, (1, 2, 3), 32)

        # self.bn = nn.BatchNorm2d(2048)
        # self.fc3 = nn.Linear(512, num_classes)
        # self.fc4 = nn.Linear(1024, num_classes)

        # self.globalatt1 = GlobalAtt(112, 256, 14, 512, 32, 1)
        # self.globalatt2 = GlobalAtt(56, 512, 14, 512, 16, 1)
        # self.globalatt3 = GlobalAtt(28, 1024, 14, 512, 8, 1)
        # self.globalatt4 = GlobalAtt(14, 2048, 14, 512, 4, 1)
        # self.combine = nn.Sequential(
        #     nn.Conv2d(2048, 2048, 1),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU())

        # lrau1 = LRAU(k=2, n=num_classes, C_in=256)
        # self.lrau2 = LRAU(k=2, n=num_classes, C_in=512)
        # self.lrau3 = LRAU(k=2, n=num_classes, C_in=1024)
        # self.lrau4 = LRAU(k=2, n=num_classes, C_in=2048)
        # self.lrau_fc = nn.Sequential(
        #     nn.AdaptiveMaxPool2d(1),
        #     nn.Flatten(1),
        #     nn.Linear(2*num_classes, num_classes)
        # )

        # self.globalatt = GlobalAttv1()

        # self.bam = BAM(16, 2048)
        # self.cbam = CBAM(16, 2048)

        # self.spatial_weight = nn.Parameter(torch.zeros((1, 14, 14)))
        # nn.init.kaiming_normal(self.spatial_weight)

        # self.involution = Involution(2048, 3, 1)

        # self.conv_final = nn.Conv2d(
        #     2, 1, kernel_size=5,
        #     stride=1, padding=2)
        # self.bn_final = nn.BatchNorm2d(1)

    # https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    # hook for the gradients of the activations
    # def activations_hook(self, grad):
    #     self.gradients = grad
    
    # method for the gradient extraction
    # def get_activations_gradient(self):
    #     return self.gradients

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,
                    drop_block_rate: float = 0., block_size: int = 5, gamma_scale: float = 0.25,
                    drop_path_rate: float=0.) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            drop_block_rate, block_size, gamma_scale,
                            drop_path_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                drop_block_rate=drop_block_rate, block_size=block_size, gamma_scale=gamma_scale,
                                drop_path_rate=drop_path_rate))

        # layers.append(MSBottleneck(
        #     self.inplanes, planes, groups=self.groups,
        #     base_width=self.base_width, dilation=self.dilation,
        #     norm_layer=norm_layer, r=2, d=[1, 2, 3, 4], g=1
        # ))
        return nn.Sequential(*layers)

    def _make_layer_muliscale(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, previous_block_channel=0) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes+previous_block_channel, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes+previous_block_channel, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, viewpoint_emb) -> Tensor:
        # See note [TorchScript super()]
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.layer1(x1)
        # x2 = x2 + self.se1(x2)
        x3 = self.layer2(x2)
        # x3 = x3 + self.se2(x3)
        # x3 = self.ms2(x2, x3)
        x4 = self.layer3(x3)
        # x4 = x4 + self.se3(x4)
        # x4 = self.ms3(x3, x4)
        x5 = self.layer4(x4)
        # x5 = x5 + self.se4(x5)
        # x5 = self.ms4(x4, x5)

        # x5 = x5 - x5 * nn.Sigmoid()(viewpoint_emb.mean(1, keepdim=True))

        # _make_layer_muliscale
        # x2 = self.layer1(x1)
        # x3 = self.layer2(x2)
        # x2 = nn.AvgPool2d(2, 2)(x2)
        # x4 = self.layer3(torch.cat([x3, x2], 1))
        # x3 = nn.AvgPool2d(2, 2)(x3)
        # x5 = self.layer4(torch.cat([x4, x3], 1))
        
        # if round == 0:
            # register the hook for Grad-CAM
            # h = x5.register_hook(self.activations_hook)

            # x_final = self.avgpool(x5_se) 
            # x_final = torch.flatten(x_final, 1)
            # logits = self.fc(x_final)

            # WS-DAN
            # Generate Attention Map
            # batch_size = x5.size(0)
            # EPSILON = 1e-12
            # M = 32
            # attention_maps = x5[:, :M, ...]
            # if self.training:
            #     # Randomly choose one of attention maps Ak
            #     attention_map = []
            #     for i in range(batch_size):
            #         attention_weights = torch.sqrt(attention_maps[i].abs().sum(dim=(1, 2)).detach() + EPSILON)
            #         attention_weights = F.normalize(attention_weights, p=1, dim=0)
            #         k_index = np.random.choice(M, 2, p=attention_weights.cpu().numpy())
            #         attention_map.append(attention_maps[i, k_index, ...])
            #     attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
            # else:
            #     # Object Localization Am = mean(Ak)
            #     attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)
            # return attention_map, logits

            # return logits

        # localization and second pass
        # x = AOLM(x, x5)
        # x1 = self.conv1(x)
        # x1 = self.bn1(x1)
        # x1 = self.relu(x1)
        # x1 = self.maxpool(x1)
        # x2 = self.layer1(x1)
        # x3 = self.layer2(x2)
        # x4 = self.layer3(x3)
        # x5 = self.layer4(x4)

        # FPN
        # x_fpn = dict(
        #     layer3=x3, layer4=x4, layer5=x5
        # )
        # x3, x4, x5 = self.fpn(x_fpn).values()
        # x2, x3, x4, x5 = self.fpn([x2, x3, x4, x5])
        # x3_pool = self.pool(x3)
        # x4_pool = self.pool(x4)
        # x5_pool = self.pool(x5)
        # x3_logits = self.cls3(x3_pool)
        # x4_logits = self.cls4(x4_pool)
        # x5_logits = self.cls5(x5_pool)
        # xcat_logits = self.cls_concat(torch.cat([x3_pool, x4_pool, x5_pool], 1))

        # APCNN crop
        # x2_crop, crop_all = self.apcnncrop(x1, x2, x3, x4, x5)
        
        # lstm_logits = []
        # for crop in crop_all:
        #     hidden_state, _ = self.lstm(crop.unsqueeze(0))[1]
        #     lstm_logits.append(hidden_state.squeeze())
        # lstm_logits = self.cls_lstm(torch.stack(lstm_logits, 0))
        
        # x_mhsa3 = roi3 + self.mhsa3(roi3, None)
        # x_mhsa4 = roi4 + self.mhsa4(roi4, None)
        # x_mhsa5 = roi5 + self.mhsa5(roi5, None)
        # x_fpn = dict(
        #     layer3=x_mhsa3, layer4=x_mhsa4, layer5=x_mhsa5
        # )
        # x_mhsa3, x_mhsa4, x_mhsa5 = self.fpn(x_fpn).values()
        # x_final = torch.cat([x_mhsa3, x_mhsa4, x_mhsa5], 1)

        # use layer2-4 as logits
        # x_final3 = self.avgpool(x3)
        # x_final3 = torch.flatten(x_final3, 1)
        # logits3 = self.fc3(x_final3)
        # x_final4 = self.avgpool(x4)
        # x_final4 = torch.flatten(x_final4, 1)
        # logits4 = self.fc4(x_final4)

        # x2 = self.globalatt1(x2)
        # x3 = self.globalatt2(x3)
        # x4 = self.globalatt3(x4)
        # x5 = self.globalatt4(x5)
        # x5 = torch.cat([x2, x3, x4, x4], 1)
        # x5 = self.combine(x5)

        # x5 = self.globalatt(x2, x3, x4, x5)
        # x2 = self.ms1(x1, x2)
        x3 = self.ms2(x2, x3)
        x4 = self.ms3(x3, x4)
        x5 = self.ms4(x4, x5)
        # x3 = torch.nn.functional.interpolate(
        #     x3, size=x5.size()[2:], mode='bilinear'
        # )
        # x4 = torch.nn.functional.interpolate(
        #     x4, size=x5.size()[2:], mode='bilinear'
        # )
        # x5 = x5 + x5*nn.Sigmoid()(x3) + x5*nn.Sigmoid()(x4) + x5*nn.Sigmoid()(x5)
        # x5 = self.agg(x5)
        # x5 = x5 + self.se(x5)
        # x = x + self.se(x)
        # x = self.combine(x3, x4, x5)
        # x5 = self.bam(x5)
        # x5 = self.cbam(x5)
        # x5 = self.spatial_weight * x5

        # x = self.lrau2(x3)
        # x = self.lrau3(x4, x)
        # x = self.lrau4(x5, x)
        # logits_lrau = self.lrau_fc(x)

        # x5 = self.involution(x5)

        # x5_avg = x5.mean(axis=1, keepdim=True)
        # x5_max = x5.max(axis=1, keepdim=True)[0]
        # x5_combine = torch.cat([x5_avg, x5_max], axis=1)
        # x5_combine = self.conv_final(x5_combine)
        # x5_combine = self.bn_final(x5_combine)
        # x5 = x5 * nn.Sigmoid()(x5_combine)

        x_final = self.avgpool(x5)
        # According to ConvNeXt, can add normalization after gap
        # x_final = self.bn(x_final)
        x_final = torch.flatten(x_final, 1)
        # x_final = torch.cat([x_final, viewpoint_emb], 1)
        # x_final = self.dropout(x_final)
        logits = self.fc(x_final)
        
        # # rois_coord2 = getROIS(x2.size(2), 3, 3)
        # # roi2 = RoiPooling(x2, rois_coord2).squeeze()
        # rois_coord3 = getROIS(x3.size(2), 3, 3)
        # roi3 = RoiPooling(x3, rois_coord3).squeeze()
        # rois_coord4 = getROIS(x4.size(2), 3, 3)
        # roi4 = RoiPooling(x4, rois_coord4).squeeze()
        # rois_coord5 = getROIS(x5.size(2), 3, 3)
        # roi5 = RoiPooling(x5, rois_coord5).squeeze()
        
        # # B = roi3.size(0)
        # # _, (hidden2, _) = self.lstm2(roi2)
        # _, (hidden3, _) = self.lstm3(roi3)
        # _, (hidden4, _) = self.lstm4(roi4)
        # _, (hidden5, _) = self.lstm5(roi5)
        # # _, hidden3 = self.gru3(roi3)
        # # _, hidden4 = self.gru4(roi4)
        # # _, hidden5 = self.gru5(roi5)
        # x_final = torch.cat([hidden3, hidden4, hidden5], 0).transpose(0, 1)
        # # x_final = torch.cat([
        # #     hidden3.transpose(0, 1).reshape(B, 1, -1),
        # #     hidden4.transpose(0, 1).reshape(B, 1, -1), 
        # #     hidden5.transpose(0, 1).reshape(B, 1, -1)], 1)
        # x_final = self.conv1d(x_final).transpose(1, 2)
        # # x_final = (hidden3 + hidden4 + hidden5).permute(1, 2, 0)
        # # x5 = self.channel_att(x5)
        # x_final = self.att(x5, x_final)       
        # # x_final = self.att(hidden3, hidden4, hidden5).transpose(1, 2)
        # logits = self.fc(x_final)

        # x3_proj = self.linear_proj2(x3)
        # x4_proj = self.linear_proj3(x4)
        # x5_proj = self.linear_proj4(x5)
        # x_combined = torch.stack([x3_proj, x4_proj, x5_proj], 1)
        # _, states = self.convlstm(x_combined)
        # hidden_state, _ = states[0]
        # x_final = self.fc(self.final_proj(hidden_state))

        return x_final, logits

    def forward(self, x: Tensor, viewpoint_emb=None) -> Tensor:
        return self._forward_impl(x, viewpoint_emb)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # model.load_state_dict(state_dict)

        # 1. filter out unnecessary keys
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and 'fc' not in k:
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


""" DenseNet """
class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
        self.ms2 = MultiScaleSEv2(
            128, 256, 256, 16, (1,2,3,4), 32, msconv_low=True,
            drop_block_rate=0., block_size=7, gamma_scale=0.0625,
            drop_path_rate=0.)
        self.ms3 = MultiScaleSEv2(
            256, 640, 640, 16, (1,2,3,4), 32, msconv_low=True,
            drop_block_rate=0., block_size=5, gamma_scale=0.25,
            drop_path_rate=0.)
        self.ms4 = MultiScaleSEv2(
            640, 1664, 1664, 16, (1,2,3,4), 32, msconv_low=True,
            drop_block_rate=0., block_size=3, gamma_scale=1,
            drop_path_rate=0., stride_low=False)

    def forward(self, x: Tensor) -> Tensor:
        # features = self.features(x)
        # out = F.relu(features, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = torch.flatten(out, 1)
        # out = self.classifier(out)

        x1 = self.features.conv0(x)
        x1 = self.features.norm0(x1)
        x1 = self.features.relu0(x1)
        x1 = self.features.pool0(x1)

        x2 = self.features.denseblock1(x1)
        x2 = self.features.transition1(x2)

        x3 = self.features.denseblock2(x2)
        x3 = self.features.transition2(x3)

        x4 = self.features.denseblock3(x3)
        x4 = self.features.transition3(x4)

        x5 = self.features.denseblock4(x4)
        x5 = self.features.norm5(x5)
        x5 = F.relu(x5, inplace=True)

        x3 = self.ms2(x2, x3)
        x4 = self.ms3(x3, x4)
        x5 = self.ms4(x4, x5)

        x_final = F.adaptive_avg_pool2d(x5, (1, 1))
        x_final = torch.flatten(x_final, 1)
        logits = self.classifier(x_final)
        return x_final, logits


def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    # model.load_state_dict(state_dict)

    # 1. filter out unnecessary keys
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and not k.startswith('classifier') and not k.startswith('fc'): # and not k.startswith('conv1') and not k.startswith('bn1'):
            pretrained_dict[k] = v
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


"""
MobileNetv3
Taken from https://github.com/d-li14/mobilenetv3.pytorch
"""
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1., use_my_model=False, **kwargs):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']
        self.use_my_model = use_my_model

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

        if use_my_model:
            self.ms2 = MultiScaleSEv2(
                24, 40, 40, 16, (1,2,3), 1, msconv_low=True,
                drop_block_rate=0., block_size=7, gamma_scale=0.0625,
                drop_path_rate=0.)
            self.ms3 = MultiScaleSEv2(
                40, 112, 112, 16, (1,2,3), 1, msconv_low=True,
                drop_block_rate=0., block_size=5, gamma_scale=0.25,
                drop_path_rate=0.)
            self.ms4 = MultiScaleSEv2(
                112, 160, 160, 16, (1,2,3), 1, msconv_low=True,
                drop_block_rate=0., block_size=3, gamma_scale=1,
                drop_path_rate=0.)

    def forward(self, x):
        x2 = self.features[:4](x)
        x3 = self.features[4:7](x2)
        x4 = self.features[7:13](x3)
        x5 = self.features[13:](x4)

        if self.use_my_model:
            x3 = self.ms2(x2, x3)
            x4 = self.ms3(x3, x4)
            x5 = self.ms4(x4, x5)

        x5 = self.conv(x5)
        x5 = self.avgpool(x5)
        x5 = x5.view(x5.size(0), -1)

        if self.use_my_model:
            x_final = self.classifier[:-1](x5)
            logits = self.classifier[-1](x_final)
            return x_final, logits
        else:
            logits = self.classifier(x5)
            return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    model = MobileNetV3(cfgs, mode='large', **kwargs)
    if kwargs['pretrained']:
        state_dict = torch.load('mobilenetv3-large-1cd25616.pth', map_location='cpu')

        # 1. filter out unnecessary keys
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and 'classifier' not in k:
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)
