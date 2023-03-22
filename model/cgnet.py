import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast, Type, Callable, Optional, Tuple
from torch import Tensor
import copy
import math
from collections import namedtuple, OrderedDict
import torch.nn.functional as F
import warnings
from torch.nn.modules import padding
from einops.layers.torch import Rearrange
from torch.nn.modules.activation import ReLU, Sigmoid, Softmax
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.rnn import LSTM
from torchvision import models
import re
import torch.utils.checkpoint as cp
import numpy as np

# from convlstm import ConvLSTM
# from involution_naive import Involution
# from locally_connected import LocallyConnected2d
# from pooling import SpatialPyramidPooling, TemporalPyramidPooling
# from ml_decoder import MLDecoder


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

# https://github.com/gordicaleksa/pytorch-original-transformer/blob/main/The%20Annotated%20Transformer%20%2B%2B.ipynb
class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)


def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


class MultiHeadedAttention(nn.Module):
    """
        This module already exists in PyTorch. The reason I implemented it here from scratch is that
        PyTorch implementation is super complicated as they made it as generic/robust as possible whereas
        on the other hand I only want to support a limited use-case.

        Also this is arguable the most important architectural component in the Transformer model.

        Additional note:
        This is conceptually super easy stuff. It's just that matrix implementation makes things a bit less intuitive.
        If you take your time and go through the code and figure out all of the dimensions + write stuff down on paper
        you'll understand everything. Also do check out this amazing blog for conceptual understanding:

        https://jalammar.github.io/illustrated-transformer/

        Optimization notes:

        qkv_nets could be replaced by Parameter(torch.empty(3 * model_dimension, model_dimension)) and one more matrix
        for bias, which would make the implementation a bit more optimized. For the sake of easier understanding though,
        I'm doing it like this - using 3 "feed forward nets" (without activation/identity hence the quotation marks).
        Conceptually both implementations are the same.

        PyTorch's query/key/value are of different shape namely (max token sequence length, batch size, model dimension)
        whereas I'm using (batch size, max token sequence length, model dimension) because it's easier to understand
        and consistent with computer vision apps (batch dimension is always first followed by the number of channels (C)
        and image's spatial dimensions height (H) and width (W) -> (B, C, H, W).

        This has an important optimization implication, they can reshape their matrix into (B*NH, S/T, HD)
        (where B - batch size, S/T - max src/trg sequence length, NH - number of heads, HD - head dimension)
        in a single step and I can only get to (B, NH, S/T, HD) in single step
        (I could call contiguous() followed by view but that's expensive as it would incur additional matrix copy)

    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value, mask):
        # Step 1: Scaled dot-product attention, Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        # Notation: B - batch size, S/T max src/trg token-sequence length, NH - number of heads, HD - head dimension
        # query/key/value shape = (B, NH, S/T, HD), scores shape = (B, NH, S, S), (B, NH, T, T) or (B, NH, T, S)
        # scores have different shapes as MHA is used in 3 contexts, self attention for src/trg and source attending MHA
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Step 2: Optionally mask tokens whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        # Step 3: Calculate the attention weights - how much should we attend to surrounding token representations
        attention_weights = self.softmax(scores)

        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S), value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Apply attention - compare query with key and use that to combine values (see the function for details)
        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        # Potentially, for visualization purposes, log the attention weights, turn off during training though!
        # I had memory problems when I leave this on by default
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        # Step 3: Reshape from (B, NH, S/T, HD) over (B, S/T, NH, HD) (via transpose) into (B, S/T, NHxHD) which is
        # the same shape as in the beginning of this forward function i.e. input to MHA (multi-head attention) module
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations


class MultiHeadedAttention_XCiT(nn.Module):
    """
        https://github.com/facebookresearch/xcit/blob/master/xcit.py
    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

        self.temperature = nn.Parameter(torch.ones(number_of_heads, 1, 1))

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Transpose into (B, NH, HD, S/T)
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)

        # Step 3: Normalize q and k
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        # Step 4: Compute attention to produce output (B, NH, HD, HD)
        attn = (query @ key.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)
        x = (attn @ value).permute(0, 3, 1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)
        x = self.out_projection_net(x)
        # x = self.proj_drop(x)

        return x


class MultiHeadedAttention_Channel(nn.Module):
    """
        https://github.com/facebookresearch/xcit/blob/master/xcit.py
    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        # self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        # self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        # self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        # self.log_attention_weights = log_attention_weights  # should we log attention weights
        # self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

        # self.temperature = nn.Parameter(torch.ones(number_of_heads, 1, 1))

        # self.linear_proj_conv = nn.Conv2d(in_channels = number_of_heads, out_channels = 1, 
        #                             kernel_size = 1)                                 
        # self.out_proj = nn.Linear(self.head_dimension, model_dimension)
        self.out_proj = nn.Linear(self.head_dimension, 1)
        # self.out_proj = nn.Conv1d(self.head_dimension, 1, )
        
    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Transpose into (B, NH, HD, S/T)
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)

        # Step 3: Normalize q and k
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        # Step 4: Compute attention 
        attn = (query @ key.transpose(-2, -1)) # (B, NH, HD, HD)
        # attn = attn.mean(axis = 1) # (B, HD, HD), can try conv1x1
        attn = attn.squeeze() # (B, HD, HD)
        # attn = self.linear_proj_conv(attn)
        # attn = attn.sum(axis = -1) # (B, HD)
        # attn = self.linear_proj_conv(attn)
        x = self.out_proj(attn) # (B, HD, 1)
        # x = self.out_proj(attn) # (B, NH*HD)
        
        return x


class MultiHeadedAttention_ChannelTest(nn.Module):
    """
        https://github.com/facebookresearch/xcit/blob/master/xcit.py
    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        # self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        # self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        # self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        # self.log_attention_weights = log_attention_weights  # should we log attention weights
        # self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

        # self.temperature = nn.Parameter(torch.ones(number_of_heads, 1, 1))

        # self.linear_proj_conv = nn.Conv2d(in_channels = number_of_heads, out_channels = 1, 
        #                             kernel_size = 1)                                 
        # self.out_proj = nn.Linear(self.head_dimension, model_dimension)
        # self.out_proj = nn.Linear(self.head_dimension, 1)
        # self.out_proj = nn.Conv1d(self.head_dimension, 1, )
        self.out_proj = nn.Linear(model_dimension, model_dimension)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        x = query

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                            for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Transpose into (B, NH, HD, S/T)
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)

        # Step 3: Normalize q and k
        # query = torch.nn.functional.normalize(query, dim=-1)
        # key = torch.nn.functional.normalize(key, dim=-1)

        # Step 4: Compute attention 
        attn = (query @ key.transpose(-2, -1)) # (B, NH, HD, HD)
        # attn = torch.nn.functional.normalize(attn, dim=-1)
        # attn = attn.mean(axis = 1) # (B, HD, HD), can try conv1x1
        # attn = attn.squeeze() # (B, HD, HD)
        attn = nn.Sigmoid()(attn)
        x = attn.matmul(value).permute(0, 3, 1, 2).reshape(batch_size, -1, self.head_dimension * self.number_of_heads)
        x = self.out_proj(x)
        # attn = self.liar_proj_conv(attn)
        # attn = attn.sum(axis = -1) # (B, HD)
        # attn = self.linear_proj_conv(attn)
        # x = self.out_proj(attn) # (B, HD, 1)
        # x = self.out_proj(attn) # (B, NH*HD)
        
        return x


class MultiHeadedAttention_Conv1d(nn.Module):

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Conv1d(model_dimension, model_dimension, 3, 1, padding=1), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value, mask):
        # Step 1: Scaled dot-product attention, Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        # Notation: B - batch size, S/T max src/trg token-sequence length, NH - number of heads, HD - head dimension
        # query/key/value shape = (B, NH, S/T, HD), scores shape = (B, NH, S, S), (B, NH, T, T) or (B, NH, T, S)
        # scores have different shapes as MHA is used in 3 contexts, self attention for src/trg and source attending MHA
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Step 2: Optionally mask tokens whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        # Step 3: Calculate the attention weights - how much should we attend to surrounding token representations
        attention_weights = self.softmax(scores)

        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S), value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x.transpose(1,2)).transpose(1,2).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Apply attention - compare query with key and use that to combine values (see the function for details)
        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        # Potentially, for visualization purposes, log the attention weights, turn off during training though!
        # I had memory problems when I leave this on by default
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        # Step 3: Reshape from (B, NH, S/T, HD) over (B, S/T, NH, HD) (via transpose) into (B, S/T, NHxHD) which is
        # the same shape as in the beginning of this forward function i.e. input to MHA (multi-head attention) module
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations


class GFMA(nn.Module):
    def __init__(self, isChannel = False):
        super().__init__()
        self.isChannel = isChannel
        if self.isChannel:
            self.lstm = nn.LSTM(input_size = 56 * 56, hidden_size = 56 * 56, num_layers = 1, 
                                batch_first = True, bidirectional = False)
        else:
            self.lstm = nn.LSTM(input_size = 64, hidden_size = 64, num_layers = 1,
                                batch_first = True, bidirectional = False) # for channel
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_att = x.reshape((B, C, H * W))
        if not self.isChannel:
            x_att = x_att.permute(0, 2, 1)
        x_att, (hidden, cell) = self.lstm(x_att)
        if not self.isChannel:
            x_att = x_att.permute(0, 2, 1)
        x_att = x_att.reshape((B, C, H, W))
        x_att = nn.Sigmoid()(x_att)
        x = x + x * x_att
        return x


class FSRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlstm1 = ConvLSTM(input_dim=2048, hidden_dim=2048, kernel_size=(3, 3),
                                    num_layers=1, batch_first=True, bias=True,
                                    return_all_layers=True)
        self.convlstm2 = ConvLSTM(input_dim=2048, hidden_dim=1024, kernel_size=(3, 3),
                                    num_layers=1, batch_first=True, bias=True,
                                    return_all_layers=True)
        self.convlstm3 = ConvLSTM(input_dim=1024, hidden_dim=1024, kernel_size=(3, 3),
                                    num_layers=1, batch_first=True, bias=True,
                                    return_all_layers=True)
    
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = x.repeat(1, 3, 1, 1, 1)
        hidden1, cell1 = self.convlstm1(x)
        hidden2, cell2 = self.convlstm2(hidden1[0])
        hidden3, cell3 = self.convlstm3(hidden2[0])
        x_out = torch.cat((hidden3[0][:, -2], hidden3[0][:, -1]), 1)
        return x_out


class SE(nn.Module):
    def __init__(self, r, C):
        super().__init__()
        self.r = r
        self.C = C
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_c = self.globalavgpool(x)
        x_c = x_c.view((batch_size, -1))
        x_c = self.dense1(x_c)
        x_c = nn.ReLU(inplace=True)(x_c)
        x_c = self.dense2(x_c)
        x_c = nn.Sigmoid()(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))
        x = x * x_c
        return x


# class BAM(nn.Module):
#     def __init__(self, r, C):
#         super().__init__()
#         self.r = r
#         self.C = C
#         self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.dense1 = nn.Linear(self.C, self.C // self.r)
#         self.dense2 = nn.Linear(self.C // self.r, self.C)
#         self.bn1 = nn.BatchNorm2d(self.C)
#         self.conv1 = nn.Conv2d(self.C, self.C // self.r, kernel_size = 1,
#                                 stride = 1, padding = 0)
#         self.conv2 = nn.Conv2d(self.C // self.r, self.C // self.r, kernel_size = 3,
#                                 stride = 1, padding = 4, dilation = 4)
#         self.conv3 = nn.Conv2d(self.C // self.r, self.C // self.r, kernel_size = 3,
#                                 stride = 1, padding = 4, dilation = 4)
#         self.conv4 = nn.Conv2d(self.C // self.r, 1, kernel_size = 1, stride = 1)
#         self.bn2 = nn.BatchNorm2d(1)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x_c = self.globalavgpool(x)
#         x_c = x_c.view((batch_size, -1))
#         x_c = self.dense1(x_c)
#         x_c = self.dense2(x_c)
#         x_c = x_c.view((batch_size, self.C, 1, 1))
#         x_c = self.bn1(x_c)
#         x_s = self.conv1(x)
#         x_s = self.conv2(x_s)
#         x_s = self.conv3(x_s)
#         x_s = self.conv4(x_s)
#         x_s = self.bn2(x_s)
#         x_cs = nn.Sigmoid()(x_c + x_s)
#         x_cs = x * x_cs
#         x = x + x_cs
#         return x


# class CBAM(nn.Module):
#     def __init__(self, r, C):
#         super().__init__()
#         self.r = r
#         self.C = C
#         self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.globalmaxpool = nn.AdaptiveMaxPool2d((1,1))
#         self.dense1 = nn.Linear(self.C, self.C // self.r)
#         self.dense2 = nn.Linear(self.C // self.r, self.C)
#         self.conv1 = nn.Conv2d(2, 1, kernel_size = 7,
#                                 stride = 1, padding = 3)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x_c_avg = self.globalavgpool(x)
#         x_c_avg = x_c_avg.view((batch_size, -1))
#         x_c_avg = self.dense1(x_c_avg)
#         x_c_avg = self.dense2(x_c_avg)
#         x_c_max = self.globalmaxpool(x)
#         x_c_max = x_c_max.view((batch_size, -1))
#         x_c_max = self.dense1(x_c_max)
#         x_c_max = self.dense2(x_c_max)
#         x_c = x_c_avg + x_c_max
#         x_c = nn.Sigmoid()(x_c)
#         x_c = x_c.view((batch_size, self.C, 1, 1))
#         x = x * x_c
#         x_s_avg = x.mean(axis = 1, keepdim = True)
#         x_s_max = x.max(axis = 1, keepdim = True)[0]
#         x_s = torch.cat((x_s_avg, x_s_max), axis = 1)
#         x_s = self.conv1(x_s)
#         x_s = nn.Sigmoid()(x_s)
#         x_s = x * x_s
#         x = x + x_s
#         return x


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
        x = x * x_c

        x_s_avg = x.mean(axis=1, keepdim=True)
        x_s_max = x.max(axis=1, keepdim=True)[0]
        x_s = torch.cat((x_s_avg, x_s_max), axis=1)
        x_s = self.conv1(x_s)
        x_s = self.bn1(x_s)
        x_s = nn.Sigmoid()(x_s)
        x_s = x * x_s
        # x = x + x_s
        return x_s


class eca_layer(nn.Module):
    # https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SE_Mine(nn.Module):
    def __init__(self, r, C):
        super().__init__()
        self.r = r
        self.C = C
        # self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.involution = Involution(channels = self.C, kernel_size = 1, stride = 1)
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)
        # self.dense2 = nn.Linear((self.C - 5) // 5 + 1, self.C)
        
        # self.conv1 = nn.Conv2d(self.C, self.C, 1)
        # self.q = nn.Linear(self.C, self.C)
        # self.k = nn.Linear(self.C, self.C)
        # self.linear_proj = nn.Linear(self.C, 1)

        # self.temperature = nn.Parameter(torch.rand(1))

        # self.conv1 = nn.Conv1d(1, 1, 5, 5)
        # self.lstm1 = LSTM(self.C, self.C // 16, batch_first = True, bidirectional = False)

    def forward(self, x):
        batch_size = x.shape[0]
        # x1 = self.conv1(x)

        x1 = x.view(batch_size, self.C, -1)
        x1 = torch.nn.functional.normalize(x1, dim = 1)
        x2 = x1.permute(0, 2, 1)
        x_c = x1 @ x2
        # x_c = torch.nn.functional.normalize(x_c, dim = 1)
        # x_c = x_c * self.temperature
        # x_c = torch.abs(x_c)
        # x_c = nn.Softmax(dim=-1)(x_c) # torch.nn.functional.normalize(x_c, dim = 1) # try softmax
        # x_c = self.linear_proj(x_c) # use max/ mean/Linear Proj/ conv1d
        # x_c = x_c.mean(-1)
        x_c, _ = x_c.max(-1)
        x_c = (nn.Sigmoid()(x_c).view(batch_size, self.C, 1, 1) * x).mean((2, 3))

        # query, key projection
        # x1 = x1.permute(0, 2, 1)
        # q = self.q(x1)
        # q = torch.nn.functional.normalize(q, dim = 1)
        # k = self.k(x1)
        # k = torch.nn.functional.normalize(k, dim = 1)
        # q = q.permute(0, 2, 1)
        # x_c = q @ k
        # x_c = x_c * self.temperature
        # # x_c = x_c.mean(-1)
        # x_c, _ = x_c.max(-1)

        # x_c = self.involution(x)
        # x_c = self.globalavgpool(x_c)
        # x_c = x_c.view((batch_size, -1))
        # x_c = x_c.unsqueeze(1)
        x_c = self.dense1(x_c)
        # x_c = self.conv1(x_c)
        # x_c, (hidden, cell) = self.lstm1(x_c)
        # hidden = hidden.view(batch_size, -1)
        # hidden = hidden.mean(axis = 0)
        x_c = nn.ReLU(inplace = True)(x_c)
        x_c = self.dense2(x_c)
        x_c = nn.Sigmoid()(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))
        x = x * x_c
        # x = x + x_c
        return x


class SE_Mine1(nn.Module):
    def __init__(self, r, C, S): # r=reduction ratio, C=channel number, S=sequence length
        super().__init__()
        self.r = r
        self.C = C
        self.S = S
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)
        
        # self.conv1 = nn.Conv2d(self.C, self.C, 1)
        self.q = nn.Linear(self.S, self.S)
        self.k = nn.Linear(self.S, self.S)
        # self.linear_proj = nn.Linear(self.C, 1)

        # self.temperature = nn.Parameter(torch.ones(1, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        # x1 = self.conv1(x)

        x1 = x.view(batch_size, self.C, -1)
        # x1 = torch.nn.functional.normalize(x1, dim = 1)
        # x2 = x1.permute(0, 2, 1)
        # x_c = x1 @ x2
        # x_c = self.linear_proj(x_c) # use max/ mean/Linear Proj
        # x_c = x_c.mean(-1)
        # x_c, _ = x_c.max(-1)

        # query, key projection
        # x1 = x1.permute(0, 2, 1)
        q = self.q(x1)
        q = torch.nn.functional.normalize(q, dim = -1)
        k = self.k(x1)
        k = torch.nn.functional.normalize(k, dim = -1)
        k = k.permute(0, 2, 1)
        x_c = q @ k
        # x_c = x_c * self.temperature
        # # x_c = x_c.mean(-1)
        x_c, _ = x_c.max(-1)

        # x_c = self.globalavgpool(x)
        x_c = x_c.view((batch_size, -1))
        x_c = self.dense1(x_c)
        x_c = nn.ReLU()(x_c, inplace = True)
        x_c = self.dense2(x_c)
        x_c = nn.Sigmoid()(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))
        x = x * x_c
        return x


class seq2seq_att(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear_proj = nn.Linear(2048, 2048)

    def forward(self, x_enc, x_dec):
        B_enc, C_enc, H_enc, W_enc = x_enc.size()
        enc_seq_len = H_enc * W_enc
        x_enc = x_enc.permute(0, 2, 3, 1)
        x_enc = x_enc.view(B_enc, enc_seq_len, C_enc)

        B_dec, C_dec, H_dec, W_dec = x_dec.size()
        dec_seq_len = H_dec * W_dec
        x_dec = x_dec.permute(0, 2, 3, 1)
        x_dec = x_dec.view(B_dec, dec_seq_len, C_dec)

        x_att = x_dec @ x_enc.transpose(1, 2)
        x_att = nn.Softmax(dim=-1)(x_att)
        
        x_out = []
        for i in range(dec_seq_len):
            x_out.append(torch.sum(torch.unsqueeze(x_att[:, i], -1) * x_enc, dim=1))
        x_out = torch.stack(x_out, 1)

        # x_out = x_out.transpose(1, 2)
        # x_out = x_out.view(B_dec, C_dec, H_dec, W_dec)

        x_out = torch.cat([x_dec, x_out], 1) 

        return x_out


class InvolutionSE(nn.Module):
    def __init__(self, size, C, r = 16, channel_per_group = 1, avg = False):
        super(InvolutionSE, self).__init__()
        self.size = size # feature map size, assume H=W
        self.C = C
        self.r = r
        self.channel_per_group = channel_per_group
        self.avg = avg

        assert C % channel_per_group == 0
        self.groups = self.C // self.channel_per_group
        
         # can add bn and relu after locallyconnected layer
        self.LC = nn.ModuleList([nn.Sequential(LocallyConnected2d(in_channels =  1 if self.avg else channel_per_group, 
                                                    out_channels = 1, # can change out_channels
                                                    output_size = size, kernel_size = 1, 
                                                    stride = 1, bias = True),
                                                nn.ReLU(inplace = True))        
                                                    for i in range(self.groups)])

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dense1 = nn.Linear(self.C, self.C // r)
        self.dense2 = nn.Linear(self.C // r, self. C)

    def forward(self, x):
        B, C, H, W = x.size()
        x_split = torch.split(x, self.channel_per_group, dim = 1)
        x_weighted = []
        for x_tmp, layer in zip(x_split, self.LC):
            if self.avg:
                x_weighted_tmp = x_tmp * layer(x_tmp.mean(dim=1, keepdim=True))
            else:
                x_weighted_tmp = x_tmp * layer(x_tmp)
            x_weighted.append(x_weighted_tmp)
        x_weighted = torch.cat(x_weighted, dim = 1)
        x_c = self.gap(x_weighted)
        x_c = self.dense1(x_c.squeeze())
        x_c = nn.ReLU(inplace = True)(x_c)
        x_c = self.dense2(x_c)
        x_c = nn.Sigmoid()(x_c)
        x_c = x_c.view(B, C, 1, 1)
        x = x * x_c
        return x


class CorBasedPool(nn.Module):
    def __init__(self):
        super(CorBasedPool, self).__init__()
        self.dense = nn.Linear(49,1)
    
    def forward(self, x):
        x_ori = torch.clone(x)
        B, C, H, W = x.size()
        x = x.view(B, C, H * W)
        cor = x.transpose(1, 2) @ x
        weight = self.dense(cor)
        weight = nn.Softmax(dim=-1)(torch.squeeze(weight))
        weight = weight.view(B, 1, H, W)
        return torch.sum(x_ori * weight, dim=(2,3))


class att_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 1, (256, 2, 2), padding = 1)

    def forward(self, x):
        x = x.unsqueeze(axis=1)
        x = self.conv1(x)
        x = x.squeeze()
        x = nn.ReLU(inplace=True)(x)
        return x


class Perceiver_like(nn.Module):
    def __init__(self, q_dim, q_seq_len, kv_dim, kv_seq_len, number_of_heads, 
                patch_size, reduce_by_conv, concat):
        super(Perceiver_like, self).__init__()
        self.reduce_q = True if q_seq_len > kv_seq_len else False
        self.reduce_by_conv = reduce_by_conv
        self.number_of_heads = number_of_heads
        model_dimension = kv_dim if self.reduce_q else q_dim
        self.head_dimension = model_dimension // number_of_heads
        self.concat = concat
        self.attention_dropout = nn.Dropout(0, inplace = True)

        if self.reduce_q:
            if reduce_by_conv:
                self.conv_q = nn.Conv2d(q_dim, model_dimension, 3, 2, padding = 1)
                self.to_patch_embedding_q = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                        p1 = 1, p2 = 1)
                self.pos_enc_q = PositionalEncoding(model_dimension = model_dimension, 
                                                    dropout_probability = 0, expected_max_sequence_length = q_seq_len // 4)
                self.q_proj = nn.Linear(model_dimension, model_dimension)
            else:
                self.to_patch_embedding_q = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                        p1 = patch_size, p2 = patch_size)
                self.pos_enc_q = PositionalEncoding(model_dimension = q_dim * patch_size ** 2, 
                                                    dropout_probability = 0, expected_max_sequence_length = q_seq_len)
                self.q_proj = nn.Linear(q_dim * patch_size ** 2, model_dimension)
        else:
            self.to_patch_embedding_q = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                    p1 = 1, p2 = 1)
            self.pos_enc_q = PositionalEncoding(model_dimension = q_dim, 
                                                dropout_probability = 0, expected_max_sequence_length = q_seq_len)
            self.q_proj = nn.Linear(q_dim, model_dimension)

        if not self.reduce_q:
            if reduce_by_conv:
                self.conv_kv = nn.Conv2d(kv_dim, model_dimension, 3, 2, padding = 1)
                self.to_patch_embedding_kv = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                         p1 = 1, p2 = 1)
                self.pos_enc_kv = PositionalEncoding(model_dimension = model_dimension,
                                                        dropout_probability = 0, expected_max_sequence_length = kv_seq_len // 4)
                self.kv_proj = nn.Linear(model_dimension, model_dimension * 2)
            else:
                self.to_patch_embedding_kv = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                        p1 = patch_size, p2 = patch_size)
                self.pos_enc_kv = PositionalEncoding(model_dimension = kv_dim * patch_size ** 2,
                                                        dropout_probability = 0, expected_max_sequence_length = kv_seq_len)
                self.kv_proj = nn.Linear(kv_dim * patch_size ** 2, model_dimension * 2)
        else:
            self.to_patch_embedding_kv = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                    p1 = 1, p2 = 1)
            self.pos_enc_kv = PositionalEncoding(model_dimension = kv_dim,
                                                    dropout_probability = 0, expected_max_sequence_length = kv_seq_len)
            self.kv_proj = nn.Linear(kv_dim, model_dimension * 2)

        self.out_proj = nn.Linear(model_dimension, model_dimension)

        if self.concat:
            self.conv1 = nn.Sequential(nn.Conv2d(model_dimension * 2, model_dimension, 1, 1),
                                        nn.BatchNorm2d(model_dimension),
                                        nn.ReLU(inplace = True))

    def forward(self, q, kv):
        B = q.size()[0]
        x = torch.clone(kv if self.reduce_q else q) 

        if self.reduce_q and self.reduce_by_conv:
            q = self.conv_q(q)
        q = self.to_patch_embedding_q(q)
        q = self.pos_enc_q(q)
        q = self.q_proj(q)
        q = q.view(B, q.size()[1], self.number_of_heads, self.head_dimension).transpose(1, 2)
        
        if not self.reduce_q and self.reduce_by_conv:
            kv = self.conv_kv(kv)
        kv = self.to_patch_embedding_kv(kv)
        kv = self.pos_enc_kv(kv)
        kv = self.kv_proj(kv)
        kv = kv.view(B, kv.size()[1], self.number_of_heads, self.head_dimension * 2).transpose(1, 2)
        k, v = torch.split(kv, kv.size()[-1] // 2, -1)

        attention_weight = nn.Softmax(dim = -1)(q @ k.transpose(2, 3) / self.head_dimension ** 0.5)
        attention_weight = self.attention_dropout(attention_weight)
        attention_out = attention_weight @ v
        attention_out = attention_out.transpose(1, 2).reshape(B, -1, self.number_of_heads * self.head_dimension)
        attention_out = self.out_proj(attention_out)
        attention_out = attention_out.transpose(1, 2).reshape(x.size())

        if self.concat:
            x = torch.cat([x, attention_out], 1)
            x = self.conv1(x)
        return x if self.concat else attention_out


class Perceiver_HT(nn.Module):
    def __init__(self, q_dim, q_seq_len, kv_dim, kv_seq_len, number_of_heads):
        super(Perceiver_HT, self).__init__()
        self.number_of_heads = number_of_heads
        self.model_dimension = q_dim if q_dim > kv_dim else kv_dim
        self.head_dimension = self.model_dimension // number_of_heads

        self.to_patch_embedding_q = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                            p1 = 1, p2 = 1)
        self.pos_enc_q = PositionalEncoding(model_dimension = q_dim, 
                                            dropout_probability = 0, expected_max_sequence_length = q_seq_len)
        self.q_proj = nn.Linear(q_dim, self.model_dimension)

        self.max_pool_kv = nn.MaxPool2d(8, 8)
        self.to_patch_embedding_kv = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                            p1 = 1, p2 = 1)
        self.pos_enc_kv = PositionalEncoding(model_dimension = kv_dim, 
                                            dropout_probability = 0, expected_max_sequence_length = kv_seq_len)
        self.kv_proj = nn.Linear(kv_dim, self.model_dimension * 2)
        
        self.attention_dropout = nn.Dropout(0, inplace = True)

        self.out_proj = nn.Linear(self.model_dimension, self.model_dimension)

    def forward(self, q, kv):
        B = q.size()[0]

        q = self.to_patch_embedding_q(q)
        q = self.pos_enc_q(q)
        q = self.q_proj(q)
        q = q.view(B, q.size()[1], self.number_of_heads, self.head_dimension).transpose(1, 2)
        
        kv = self.max_pool_kv(kv)
        kv = self.to_patch_embedding_kv(kv)
        kv = self.pos_enc_kv(kv)
        kv = self.kv_proj(kv)
        kv = kv.view(B, kv.size()[1], self.number_of_heads, self.head_dimension * 2).transpose(1, 2)
        k, v = torch.split(kv, kv.size()[-1] // 2, -1)

        attention_weight = nn.Softmax(dim = -1)(q @ k.transpose(2, 3) / self.head_dimension ** 0.5)
        attention_weight = self.attention_dropout(attention_weight)
        attention_out = attention_weight @ v
        attention_out = attention_out.transpose(1, 2).reshape(B, -1, self.number_of_heads * self.head_dimension)
        attention_out = self.out_proj(attention_out)

        return attention_out.mean(axis = 1)


class MHA_SE(nn.Module):
    def __init__(self, seq_len, number_of_heads, patch_size, r, C):
        super().__init__()
        self.number_of_heads = number_of_heads
        self.r = r
        self.C = C
        
        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                            p1 = patch_size, p2 = patch_size)
        # self.pos_enc = PositionalEncoding(model_dimension = C * patch_size ** 2, 
        #                                     dropout_probability = 0, expected_max_sequence_length = seq_len)
        self.proj = nn.Linear(C* patch_size ** 2, C * patch_size ** 2 * 2)
        self.attention_dropout = nn.Dropout(0, inplace = True)

        self.gap = nn.AdaptiveMaxPool2d((1, 1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)
       
    def forward(self, x):
        B, C, H, W = x.shape

        x_att = self.to_patch_embedding(x)
        # x_att = self.pos_enc(x_att)
        x_att = self.proj(x_att)
        x_att = x_att.view(B, x_att.size(1), self.number_of_heads, x_att.size(-1) // self.number_of_heads)
        x_att = x_att.transpose(1, 2)
        q, k = torch.split(x_att, x_att.size(-1) // 2, dim = -1)
        att_weight = q @ k.transpose(2, 3)
        x_att = nn.Sigmoid()(att_weight.max(-1)[0].mean(1).reshape(B, 1, H, W)) * x
        
        x_att = self.gap(x_att)
        x_att = x_att.view((B, -1))
        x_att = self.dense1(x_att)
        x_att = nn.ReLU(inplace=True)(x_att)
        x_att = self.dense2(x_att)
        x_att = nn.Sigmoid()(x_att)

        return x + x * x_att.view(B, C, 1, 1) # x * x_att.view(B, C, 1, 1)


class RecurrentMHA(nn.Module):
    def __init__(self, model_dimension, number_of_heads):
        super(RecurrentMHA, self).__init__()
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads
        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                            p1 = 1, p2 = 1)
        self.q = nn.Parameter(torch.randn(1, 49, 2048)).repeat(64)
        self.out_proj = nn.Linear(model_dimension, model_dimension)
    
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q.view(B, self.q.size(1), self.number_of_heads, self.head_dimension).tranpose(1, 2)
        x = self.to_patch_embedding(x)
        x = PositionalEncoding(model_dimension = x.size(-1), dropout_probability = 0,
                                expected_max_sequence_length = x.size(1))(x)
        x = x.view(B, x.size(1), self.number_of_heads, self.head_dimension).tranpose(1, 2)
        att_weight = q @ x.transpose(2, 3)
        q = att_weight * x
        q = q.tranpose(1, 2).view(B, 49, -1)
        self.q = self.out_proj(q)

        return self.q


class PartAttention(nn.Module):
    def __init__(self, C_in, C_out):
        super(PartAttention, self).__init__()
        self.x_cls = nn.Conv2d(C_in, 1, 1, 1)
        self.C_out = C_out
        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                            p1 = 1, p2 = 1)
    
    def forward(self, x):
        x_ori = x
        x_cls = self.x_cls(x)
        x_cls = self.to_patch_embedding(x_cls)
        x = self.to_patch_embedding(x)
        att_weight = x_cls.transpose(1, 2) @ x
        idx = torch.argsort(att_weight, -1, descending = True).squeeze()
        x_new = []
        for i, idx_tmp in enumerate(idx):
            x_new.append(x_ori[i, idx_tmp[:self.C_out]])
        x_new = torch.stack(x_new, dim = 0)
        return x_new


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


class BasicConv_v1(nn.Module):
    def __init__(self, in_planes, out_planes, use_pool):
        super(BasicConv_v1, self).__init__()
        self.out_channels = out_planes
        self.use_pool = use_pool
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Sequential(
                                    nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_planes),
                                    nn.ReLU(inplace=True)
        )
        if self.use_pool:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv3(x)
        if self.use_pool:
            x = self.pool(x)
        return x


class Cor(nn.Module):
    def __init__(self):
        super(Cor, self).__init__()
    
    def forward(self, x2, x3, x4, x5):
        B, C, H, W = x2.shape
        x2 = x2.view(B, C, -1).transpose(1,2)
        x3 = x3.view(B, C, -1).transpose(1,2)
        x4 = x4.view(B, C, -1).transpose(1,2)
        x5 = x5.view(B, C, -1).transpose(1,2)

        x2 = torch.nn.functional.normalize(x2, dim = -1)
        x3 = torch.nn.functional.normalize(x3, dim = -1)
        x4 = torch.nn.functional.normalize(x4, dim = -1)
        x5 = torch.nn.functional.normalize(x5, dim = -1)
        
        cor2 = torch.matmul(x5, x2.transpose(1, 2))
        cor3 = torch.matmul(x5, x3.transpose(1, 2))
        cor4 = torch.matmul(x5, x4.transpose(1, 2))
        
        cor2 = torch.diagonal(cor2, dim1 = 1, dim2 = 2)
        cor3 = torch.diagonal(cor3, dim1 = 1, dim2 = 2)
        cor4 = torch.diagonal(cor4, dim1 = 1, dim2 = 2)

        (x5[0,0] * x2[0,0]).sum()

        return None


class Mlp(nn.Module):
    def __init__(self, C):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(C, C * 2)
        self.fc2 = nn.Linear(C * 2, C * 2)
        self.act_fn = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(0)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ExternalAttention(nn.Module):
    # Base code obtained from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/ExternalAttention.py
    # Base code is amended according to https://arxiv.org/pdf/2105.02358.pdf
    def __init__(self, d_model,S=64,att_head=8):
        super().__init__()
        self.conv = nn.Conv2d(d_model, d_model, 3, 1, 1,)
        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1)
        self.d_model = d_model
        self.att_head = att_head
        self.att_dimension = d_model // att_head
        # self.linear_proj = nn.Linear(d_model, d_model)
        self.mk = nn.Linear(self.att_dimension, S, bias = False)
        self.mv = nn.Linear(S, self.att_dimension, bias = False)
        self.softmax = nn.Softmax(dim = 1)
        self.init_weights()
        self.out_proj = nn.Linear(d_model, d_model)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        B, C, H, W = queries.size()
        queries = self.conv(queries)
        queries = self.to_patch_embedding(queries)
        # queries = self.linear_proj(queries)
        queries = queries.view(B, H * W, self.att_head, self.att_dimension).transpose(1, 2)
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        out = out.transpose(1, 2).reshape(B, H * W, -1)
        out = self.out_proj(out)
        out = out.transpose(1,2).view(B, C, H, W)

        return out


class MHSE(nn.Module):
    def __init__(self, r, C, H, W, S):
        super().__init__()
        self.r = r
        self.C = C
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.swp = nn.Parameter(torch.normal(0, 0.005, size=(1, H, W)))
        self.conv1 = nn.Conv2d(C, C //r, 1, 1, groups = S)
        self.conv2 = nn.Conv2d(C//r, C, 1, 1, groups = S)
        self.conv_all = nn.Conv2d(C, C, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        # x_c = self.gap(x)
        x_c = torch.sum(x * self.swp, dim=(2,3), keepdim=True)
        x_c = self.conv1(x_c)
        x_c = nn.ReLU(inplace=True)(x_c)
        x_c = self.conv2(x_c)
        x_c = self.conv_all(x_c)
        x_c = nn.Sigmoid()(x_c)
        x = x * x_c
        return x


class SELC(nn.Module):
    def __init__(self, r, C, output_size, series, stride=1, kernel_size=3):
        super(SELC, self).__init__()
        self.r = r
        self.C = C
        # self.swp = nn.Parameter(torch.zeros((1, 1, output_size, output_size)))
        
        # self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.globalmaxpool = nn.AdaptiveMaxPool2d(1)
        # self.dynamic_pool = nn.Linear(output_size ** 2, 1)
        # self.dense1 = nn.Linear(in_features=self.C, out_features=self.C//self.r, bias=True)
        # self.dense2 = nn.Linear(in_features=self.C//self.r, out_features=self.C, bias=True)

        self.se = nn.Sequential(
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(self.C, self.C // self.r, 1, 1),
                                # nn.BatchNorm2d(self.C // self.r),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.C // self.r, self.C, 1, 1),
                                # nn.BatchNorm2d(self.C),
                                # nn.Conv2d(self.C, self.C, 1, 1),
                                nn.Sigmoid()
                                )
        
        # self.se_max = nn.Sequential(
        #                         nn.AdaptiveMaxPool2d((1, 1)),
        #                         nn.Conv2d(self.C, self.C // self.r, 1, 1),
        #                         # nn.BatchNorm2d(self.C // self.r),
        #                         nn.ReLU(inplace=True),
        #                         nn.Conv2d(self.C // self.r, self.C, 1, 1),
        #                         # nn.BatchNorm2d(self.C),
        #                         # nn.Conv2d(self.C, self.C, 1, 1),
        #                         nn.Sigmoid()
        #                         )
        
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gmp = nn.AdaptiveMaxPool2d(1)
        # self.squeeze = nn.Conv2d(self.C, self.C // self.r, 1)
        # self.excite = nn.Conv2d(self.C // self.r, self.C, 1)
        # self.combine = nn.Conv2d(self.C * 2, self.C, 1)

        self.lc = nn.Sequential(
                                nn.Conv2d(C, 1, 1),
                                # nn.BatchNorm2d(1),
                                nn.ReLU(inplace=True),
                                LocallyConnected2d(1, 1, output_size, kernel_size, stride, 
                                                    True, (kernel_size - 1)//2),
                                # BatchNorm2d(1),
                                # nn.ReLU(inplace=True),
                                # nn.Conv2d(C // r, 1, 1, 1),
                                # BatchNorm2d(self.C),
                                # Involution(C, 3, 1),
                                # nn.ReLU(inplace=True),
                                # nn.Conv2d(C, 1, 1)
                                )
        
        # self.conv1 = nn.Sequential(
        #                             nn.Conv2d(C, 1, 1),
        #                             nn.BatchNorm2d(1),
        #                             nn.ReLU(inplace=True)
        #                             )
        # self.lc = nn.ModuleList([LocallyConnected2d(1, 1, output_size, 3, 1) for i in range(self.r)])
        # self.out_proj = nn.Conv2d(self.r, 1, 1)
        # self.bn = nn.BatchNorm2d(1)
        # self.lc1x1 = LocallyConnected2d(1, 1, output_size, 1, 1, True, 0)
        # self.lc3x3 = LocallyConnected2d(1, 1, output_size, 3, 1, True, 1)
        # self.conv1x1 = nn.Conv2d(2, 1, 1)
        # self.bn = nn.BatchNorm2d(1)

        # self.downsample = None
        # if stride == 2:
        #     self.downsample = nn.Conv2d(C, C, 3, 2, 1)

        # self.combine = nn.Conv2d(C * 2, C, 1, 1)
        
        self.series = series
    
    def forward(self, x):
        B, C, H, W = x.size()
        # if self.downsample is not None:
        #     x = self.downsample(x)
        
        # x_se = torch.sum(x * self.swp, (2, 3), keepdim = True)
        # x_se = self.globalavgpool(x)
        # x_se = x_se.view(batch_size, -1)
        # x_se = nn.Sigmoid()(self.dense2(nn.ReLU(inplace=True)(self.dense1(x_se))))
        # x_se = x_se.view(batch_size, self.C, 1, 1)
        # x_avg = self.globalavgpool(x)
        # x_max = self.globalavgpool(x)
        # x_se = x_avg + x_max
        x_se = self.se(x)
        # x_se_max = self.se_max(x)
        # x_se = torch.cat([x_se, x_se_max], 1)
        # x_se = nn.Sigmoid()(self.combine(x_se))

        
        # x_cbam_gap = self.excite(nn.ReLU(inplace=True)(self.squeeze(self.gap(x))))
        # x_cbam_gmp = self.excite(nn.ReLU(inplace=True)(self.squeeze(self.gmp(x))))
        # x_cbam = nn.Sigmoid()(x_cbam_gap + x_cbam_gmp)

        # x_cbam_gap = nn.ReLU(inplace=True)(self.excite(nn.ReLU(inplace=True)(self.squeeze(self.gap(x)))))
        # x_cbam_gmp = nn.ReLU(inplace=True)(self.excite(nn.ReLU(inplace=True)(self.squeeze(self.gmp(x)))))
        # x_cbam = nn.Sigmoid()(x_cbam_gap + x_cbam_gmp)
        # x_cbam = nn.Sigmoid()(self.combine(torch.cat([x_cbam_gmp, x_cbam_gap], 1)))
        # x_se = x_cbam

        # x_se = self.dynamic_pool(x.view(B, C, -1)).squeeze()
        # x_se = nn.ReLU(inplace=True)(self.dense1(x_se))
        # x_se = nn.Sigmoid()(self.dense2(x_se))
        # x_se = x_se.view(B, C, 1, 1)

        if self.series:
            x_se = x * x_se
            x_lc = self.lc(x_se)
            x_out = nn.Sigmoid()(x_lc)
        else:
            # x_mean = x.mean(1, True)
            # x_max = x.max(1, True)[0]
            # x_lc = torch.cat([x_mean, x_max], 1)
            x_lc = self.lc(x)
            # x_lc = self.lc(torch.max(x, 1, True)[0])
            x_lc = nn.Sigmoid()(x_lc)
            
            # x_out = nn.Sigmoid()(x_se + x_lc)
            
            # x_lc = self.conv1(x)
            # x_lc_new = []
            # for i, layer in enumerate(self.lc):
            #     x_lc_new.append(layer(x_lc[:, i:i+1]))
            # x_lc = torch.cat(x_lc_new, 1)
            # x_lc = self.out_proj(x_lc)
            # x_lc = nn.Sigmoid()(self.bn(x_lc))

            # x_lc = self.conv1(x)
            # x_lc_conv1x1 = self.lc1x1(x_lc)
            # x_lc_conv3x3 = self.lc3x3(x_lc)
            # x_lc = nn.ReLU(inplace=True)(torch.cat([x_lc_conv1x1, x_lc_conv3x3], 1))
            # x_lc = nn.Sigmoid()(self.bn(self.conv1x1(x_lc)))
            # x_lc = nn.Sigmoid()(nn.ReLU(inplace=True)(self.bn(x_lc_conv1x1 + x_lc_conv3x3)))

            x_out = x * x_se + x * x_lc

            # x_se = x * x_se
            # x_lc = x * x_lc
            # x_out = torch.cat([x_se, x_lc], 1)
            # x_out = self.combine(x_out)
        # x_out = x * x_out
        x_out = x + x_out # uncomment this if using selc like BAM
        # x_out = x + x * x_se
        return x_out


class dual_att(nn.Module):
    def __init__(self, r, C, number_of_heads, patch_size):
        super(dual_att, self).__init__()
        self.r = r
        self.C = C
        self.se = nn.Sequential(
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(self.C, self.C // self.r, 1, 1),
                                # nn.BatchNorm2d(self.C // self.r),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.C // self.r, self.C, 1, 1),
                                # nn.BatchNorm2d(self.C),
                                # nn.Conv2d(self.C, self.C, 1, 1),
                                nn.Sigmoid()
                                )
        
        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                            p1 = patch_size, p2 = patch_size)
        self.mha = MultiHeadedAttention(C, number_of_heads, 0, False)
        # self.combine = nn.Sequential(
        #                             nn.Conv2d(C * 2, C, 1, 1),
        #                             # nn.BatchNorm2d(C),
        #                             # nn.ReLU(inplace=True)
        #                             )
        

    def forward(self, x):
        B, C, H, W = x.size()

        # x_se = x * self.se(x)
        # x_mha = self.to_patch_embedding(x)
        # x_mha = self.mha(x_mha, x_mha, x_mha, None)
        # x_mha = x_mha.transpose(1,2).view(B, C, H, W)
        # # x = x_se + x_mha
        # x = torch.cat([x_se, x_mha], 1)
        # x = self.combine(x)

        x_mha = self.to_patch_embedding(x)
        x_mha = self.mha(x_mha, x_mha, x_mha, None)
        x_mha = x_mha.transpose(1,2).view(B, C, H, W)
        x = x_mha + x_mha * self.se(x_mha)

        return x


class CA(nn.Module):
    def __init__(self, C):
        super(CA, self).__init__()
        self.eca = nn.Sequential(
                                nn.Conv1d(1, 1, 3, 1, 1),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        cor = x.bmm(x.transpose(1, 2)) # (B, C, C)
        cor_avg = cor.mean(dim=-1, keepdim=True) # (B, C, 1)
        cor = cor.bmm(cor_avg) # (B, C, 1)
        cor = cor.transpose(1, 2) # (B, 1, C)
        x_c = self.eca(cor).transpose(1, 2) # (B, C, 1)
        x_c = x * x_c
        return x_c.view(B, C, H, W)


class MultiConvKernel(nn.Module):
    def __init__(self, C, use_conv5):
        super(MultiConvKernel, self).__init__()
        self.C = C
        self.use_conv5 = use_conv5
        if use_conv5:
            out_channel = (C//4, C//2, C//4)
        else:
            out_channel = (C//2, C//2)
        self.conv1 = nn.Sequential(
                                    nn.Conv2d(C, out_channel[0], 1, 1, 0),
                                    nn.BatchNorm2d(out_channel[0]),
                                    nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
                                    nn.Conv2d(C, out_channel[1], 3, 1, 1),
                                    nn.BatchNorm2d(out_channel[1]),
                                    nn.ReLU(inplace=True)
        )
        if use_conv5:
            self.conv5 = nn.Sequential(
                                    nn.Conv2d(C, out_channel[2], 5, 1, 2),
                                    nn.BatchNorm2d(out_channel[2]),
                                    nn.ReLU(inplace=True)
        )
        self.combine = nn.Sequential(
                                    nn.Conv2d(C, C, 1),
                                    nn.BatchNorm2d(C),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        B, C, H, W = x.size()
        x_conv1 = self.conv1(x)
        x_conv3 = self.conv3(x)
        if self.use_conv5:
            x_conv5 = self.conv5(x)
            x = torch.cat([x_conv1, x_conv3, x_conv5], 1)
        else:
            x = torch.cat([x_conv1, x_conv3], 1)
        x = self.combine(x)
        return x


class MHA_like_CA(nn.Module):
    def __init__(self):
        super(MHA_like_CA, self).__init__()
    
    def forward(self, x):
        B, C, H, W = x.size()
        x = nn.Flatten(2)(x)
        att_weight = x.bmm(x.transpose(1, 2))
        att_weight = nn.Softmax(dim=-1)(att_weight)
        x = att_weight.bmm(x)
        x = x.view(B, C, H, W)
        return x


class SE_avgmax(nn.Module):
    def __init__(self, r, C):
        super().__init__()
        self.r = r
        self.C = C
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.globalmaxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_avg = self.globalavgpool(x)
        x_max = self.globalmaxpool(x)
        x_c = x_avg + x_max
        x_c = x_c.view((batch_size, -1))
        x_c = self.dense1(x_c)
        x_c = nn.ReLU(inplace = True)(x_c)
        x_c = self.dense2(x_c)
        x_c = nn.Sigmoid()(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))
        x = x + x * x_c
        return x


class AggregatedAttentionModule(nn.Module):
    # https://link.springer.com/content/pdf/10.1007/s12652-021-03599-7.pdf
    def __init__(self, C):
        super(AggregatedAttentionModule, self).__init__()
        self.eca_gmp = nn.AdaptiveMaxPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, 5, 1, padding=(5-1)//2)

        self.sa_conv = nn.Conv2d(1, 1, 7, 1, (7-1)//2)


    def forward(self,x):
        B, C, H, W = x.size()
        x_c = x * nn.Sigmoid()(self.eca_conv(self.eca_gmp(x).squeeze(-1).transpose(1, 2))).view(B, C, 1, 1)
        x_s = x * nn.Sigmoid()(self.sa_conv(x.mean(1, True) + x.max(1, True)[0]))
        x = x * x_c * x_s
        return x


class VanillaSA(nn.Module):
    def __init__(self, C):
        super(VanillaSA, self).__init__()
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C, C)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1).transpose(1, 2)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        att_weight = nn.Softmax(dim=-1)(q.bmm(k.transpose(1, 2)))
        x = att_weight.bmm(v).transpose(1, 2).view(B, C, H, W)
        return x


class VanillaSA1(nn.Module):
    def __init__(self):
        super(VanillaSA1, self).__init__()
        self.linear = nn.Conv1d(2048, 2048, 1)

    def forward(self, q, k, v):
        B, C, H, W = q.size()
        q = q.view(B, C, -1).transpose(1, 2)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1).transpose(1, 2)
        att_weight = nn.Softmax(dim=-1)(q.bmm(k) / C ** 0.5)      
        # x = (v + att_weight.bmm(v)).transpose(1, 2).view(B, C, -1)
        # x = self.linear(x)
        x = att_weight.bmm(v).transpose(1, 2).view(B, C, -1)
        x = k + self.linear(x)
        return x


class SpinalNet_ResNet(nn.Module):
    """
    https://github.com/dipuk0506/SpinalNet/blob/master/Transfer%20Learning/Transfer_Learning_Stanford_Cars.py
    """
    def __init__(self, layer_width, Num_class):
        super(SpinalNet_ResNet, self).__init__()
        self.half_in_size = 2048 // 2

        self.fc_spinal_layer1 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(self.half_in_size, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(self.half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(self.half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(self.half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(layer_width*4, Num_class),)
        
    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:self.half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,self.half_in_size:2*self.half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:self.half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,self.half_in_size:2*self.half_in_size], x3], dim=1))
        
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        
        x = self.fc_out(x)
        return x


class ShuffleChannel(nn.Module):
    def __init__(self, groups):
        super(ShuffleChannel, self).__init__()
        self.groups = groups

    def forward(self, x):
        # Shuffle channel
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups,
                    channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class GCN(nn.Module):
    """
    https://arxiv.org/pdf/2202.03822.pdf
    https://github.com/chou141253/FGVC-PIM/blob/master/models/ResNet50.py
    Graph Convolutional Network
    """
    def __init__(self, 
                 num_joints: int, 
                 in_features: int, 
                 num_classes: int,
                 use_global_token: bool = False):
        super(GCN, self).__init__()
        self.num_joints = num_joints
        self.in_features = in_features
        self.num_classes = num_classes

        A = torch.eye(num_joints)/200 + 1/2000
        if use_global_token:
            A[0] += 1/200
            A[:, 0] += 1/200

        self.adj = nn.Parameter(A)
        self.conv1 = nn.Conv1d(in_features, in_features, 1)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Conv1d(in_features, num_classes, 1)

    def forward(self, x):
        """
        x size: [B, C, N]
        """
        x = self.conv1(x)
        x = torch.matmul(x, self.adj)
        x = self.bn1(x)
        x = self.relu(x)
        
        # x = self.dropout(x)
        x = self.gap(x)
        x = self.classifier(x).squeeze()

        return x


class SimpleChannelMHA(nn.Module):
    def __init__(self, C):
        super(SimpleChannelMHA, self).__init__()
        self.proj = nn.Linear(C, C * 3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_gap = self.gap(x).squeeze()
        x_proj = self.proj(x_gap)
        q, k ,v = torch.chunk(x_proj, 3, 1)
        att_weight = nn.Softmax(-1)(q.unsqueeze(2).bmm(k.unsqueeze(1)))
        out = att_weight.bmm(v.unsqueeze(2)).unsqueeze(-1)
        return x + out


class SelectFeatures(nn.Module):
    """
    https://arxiv.org/pdf/2202.03822.pdf
    https://github.com/chou141253/FGVC-PIM/blob/master/models/ResNet50.py
    """
    def __init__(self, num_features, num_classes, num_select):
        super(SelectFeatures, self).__init__()
        self.cls = nn.Conv2d(num_features, num_classes, 1)
        self.num_select = num_select
        self.gcn = GCN(num_select, num_features, num_classes)
    
    def forward(self, x):
        logits = self.cls(x)
        B, C, H, W = logits.size()
        logits = logits.view(B, -1, H * W).transpose(1, 2)
        x = x.view(B, -1, H * W).transpose(1, 2)
        probs = nn.Softmax(-1)(logits)
        selected_features = []
        selected_confs = []
        selected_logits = {"selected":[], "not_selected":[]}
        for bi in range(B):
            max_ids, _ = torch.max(probs[bi], dim=-1)
            confs, ranks = torch.sort(max_ids, descending=True)
            sf = x[bi][ranks[:self.num_select]]
            nf = x[bi][ranks[self.num_select:]]  # calculate
            selected_features.append(sf) # [num_selected, C]
            selected_confs.append(confs) # [num_selected]
            selected_logits["selected"].append(logits[bi][ranks[:self.num_select]])
            selected_logits["not_selected"].append(logits[bi][ranks[self.num_select:]])
        
        selected_features = torch.stack(selected_features, 0) # B, S, C
        selected_features = selected_features.transpose(1, 2).contiguous()
        logits = self.gcn(selected_features)

        return logits


class GCN_SA(nn.Module):
    def __init__(self, num_joints, in_features):
        super(GCN_SA, self).__init__()
        self.num_joints = num_joints
        self.conv1 = nn.Conv1d(in_features, in_features, 1)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        
        A = torch.eye(num_joints)/200 + 1/2000
        self.adj = nn.Parameter(A)

        self.ff = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

    def forward(self, x):
        B, C, H, W = x.size()
        # Get salient spatial position
        x_flat = x.view(B, C, -1).transpose(1, 2)
        ranks = x.view(B, C, -1).mean(1).argsort(-1, descending=True)
        selected_features = []
        for bi in range(B):
            rank_tmp = ranks[bi, :self.num_joints]
            sf = x_flat[bi, rank_tmp]
            selected_features.append(sf)
        selected_features = torch.stack(selected_features, 0).transpose(1, 2)
        
        # Graph convolution
        selected_features = self.conv1(selected_features)
        selected_features = torch.matmul(selected_features, self.adj)
        selected_features = self.bn1(selected_features)
        selected_features = self.relu(selected_features).transpose(1, 2) # ReLU or Sigmoid
        
        # Spatial Attention
        # new_features_all = []
        # for bi in range(B):
        #     rank_tmp = ranks[bi, :self.num_joints]
        #     new_features = []
        #     for i in range(x_flat.size(1)):
        #         if i in rank_tmp:
        #             new_features.append(x_flat[bi, i] + selected_features[bi, rank_tmp==i].squeeze()) # Replace, Addition or Multiplication
        #         else:
        #             new_features.append(x_flat[bi, i])
        #     new_features = torch.stack(new_features, 0)
        #     new_features_all.append(new_features)
        # new_features_all = torch.stack(new_features_all, 0)

        new_features = x_flat.clone()
        for bi in range(B):
            rank_tmp = ranks[bi, :self.num_joints]
            new_features[bi, rank_tmp] = new_features[bi, rank_tmp] + selected_features[bi] # Replace, Addition or Multiplication
        new_features = new_features.transpose(1, 2).view(B, C, H, W)
        
        new_features = self.ff(new_features)
        return new_features


class MobileViTBlock(nn.Module):
    def __init__(self, kernel_size, spatial_size, channel_size, number_of_heads):
        super(MobileViTBlock, self).__init__()
        # do we need to reduce channel dimension to save flops
        
        # Unfold, can play with kernel size
        self.new_spatial_size = int(math.ceil(spatial_size / kernel_size) * kernel_size)
        self.padding_ = True if self.new_spatial_size != spatial_size else False
        self.unfold = nn.Unfold(
            kernel_size=kernel_size, dilation=1, padding=0, 
            stride=kernel_size)
        
        # Layer Normalization, do we need it?
        # self.normalize = nn.LayerNorm(channel_size)
        
        # MHA, can use drop out, play with number of heads
        self.mha = MultiHeadedAttention(
            model_dimension=channel_size, number_of_heads=number_of_heads, 
            dropout_probability=0.2, log_attention_weights=False)
        
        # Feed-Forward Network, do we need it?
        # self.ffn = nn.Sequential(
        #     nn.Linear(channel_size, 2*channel_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2*channel_size, channel_size))

        # Fold
        self.fold = nn.Fold(
            output_size=spatial_size, kernel_size=kernel_size, 
            dilation=1, padding=1 if self.padding_ else 0, stride=kernel_size)

        # Projection
        # self.proj = nn.Sequential(
        #     nn.Conv2d(channel_size, channel_size, 1),
        #     nn.BatchNorm2d(channel_size),
        #     nn.ReLU(inplace=True))

        # Feature Fusion
        self.ff = nn.Sequential(
            nn.Conv2d(2*channel_size, channel_size, 3, 1, 1),
            nn.BatchNorm2d(channel_size),
            nn.ReLU(inplace=True))

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Unfold
        x_unfold = x.clone()
        if self.padding_:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x_unfold = F.interpolate(
                x_unfold, size=(self.new_spatial_size, self.new_spatial_size), 
                mode="bilinear", align_corners=False)
        x_unfold = self.unfold(x_unfold).view(B, C, self.unfold.kernel_size**2, -1) # B, C, P, N
        x_unfold = x_unfold.permute(0, 2, 3, 1) # B, P, N, C
        B, P, N, C = x_unfold.size()
        x_unfold = x_unfold.reshape(B*P, N, C)
        
        # Layer Norm
        # x_norm = self.normalize(x_unfold)

        # MHA
        x_mha = self.mha(x_unfold, x_unfold, x_unfold, None)

        # FFN
        # x_ffn = self.ffn(x_mha)

        # Fold
        x_fold = x_mha.view(B, P, N, C)
        x_fold = x_fold.permute(0, 3, 1, 2) # B, C, P, N
        x_fold = x_fold.reshape(B, -1, N)
        x_fold = self.fold(x_fold)
        if self.padding_:
            x_fold = F.interpolate(x_fold, size=(H, W), mode="bilinear", align_corners=False)

        # Projection
        # x_proj = self.proj(x_fold)

        # Feature Fusion
        x = torch.cat([x, x_fold], 1)
        x = self.ff(x)
        return x


""" AlexNet """
class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # lstm
        # self.outdim = 16
        # self.num_layers = 1
        # self.lstm  = nn.LSTM(input_size = 256, hidden_size = self.outdim, num_layers = self.num_layers, 
        #                     batch_first = True, bidirectional = False)
        # self.timedist = nn.ModuleList([nn.Linear(16, 64) for i in range(36)])
        # self.relu = nn.ReLU()
        # self.linear = nn.Linear(9216, 4096)
        # self.final_fc = nn.Sequential(
        #     nn.Linear(6400, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        # mha
        self.pos_enc = PositionalEncoding(256, 0, 6 * 6)
        self.mha = MultiHeadedAttention(256, 8, 0, False)
        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1)
        self.linear_proj = nn.ModuleList([nn.Linear(256, 256)] * 36)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)

        # # lstm
        # x_att = x.reshape((x.shape[0], 256, 36)).transpose(1,2)
        # x_att, (hidden, cell) = self.lstm(x_att)
        # x_att = nn.Sigmoid()(x_att)
        # # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/21
        # outs=[]
        # for i in range(x_att.shape[1]):
        #     outs.append(self.relu(self.timedist[i](x_att[:, i, :])).unsqueeze(axis = 1))
        # outs = torch.cat(outs, axis=1)
        # outs = torch.flatten(outs, 1)
        # # conv
        # x = torch.flatten(x, 1)
        # x = self.linear(x)
        # x = self.relu(x)
        # x_final = torch.cat((x, outs), axis = 1)
        # x_final = self.final_fc(x_final)

        # mha
        # x_mha = x.permute((0, 2, 3, 1))
        # b, h, w, c = x_mha.shape
        # x_mha = x_mha.reshape((x_mha.shape[0], h * w, c))
        b, c, h, w = x.shape
        x_mha = self.to_patch_embedding(x)
        x_mha_linear_proj = torch.tensor([], device = torch.device('cuda:0'))
        for i in range(len(self.linear_proj)):
            output_tmp = self.linear_proj[i](x_mha[:, i:i+1])
            x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis = 1)
        x_mha = x_mha_linear_proj
        x_mha = self.pos_enc(x_mha)
        x_mha = self.mha(x_mha, x_mha, x_mha, None)
        x_mha = x_mha.reshape((x_mha.shape[0], h, w, c))
        x_mha = x_mha.mean(axis = -1, keepdim = True)
        x_mha = nn.Sigmoid()(x_mha)
        x_mha = x_mha.permute((0, 3, 1, 2))
        x_final = x * x_mha
        x_final = torch.flatten(x_final, 1)
        x_final = self.classifier(x_final)
        return x_final


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
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


""" VGG16 """
class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # lstm
        # self.outdim = 16
        # self.num_layers = 1
        # self.lstm  = nn.LSTM(input_size = 512, hidden_size = self.outdim, num_layers = self.num_layers, 
        #                     batch_first = True, bidirectional = False)
        # self.timedist = nn.ModuleList([nn.Linear(16, 64) for i in range(49)])
        # self.relu = nn.ReLU()

        # fc for conv
        # self.linear = nn.Linear(25088, 4096)

        # fc
        # self.final_fc = nn.Sequential(
        #     nn.Linear(7232, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        # mha
        # self.pos_enc = PositionalEncoding(512, 0, 7 * 7)
        # self.mha = MultiHeadedAttention(512, 8, 0, False)
        # self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1)
        # self.linear_proj = nn.ModuleList([nn.Linear(512, 512)] * 49)

        # PMG
        self.layer1 = self.features[5:10]
        self.layer2 = self.features[10:17]
        self.layer3 = self.features[17:24]
        self.layer4 = self.features[24:]
        self.features = self.features[:5]
        self.conv_block1 = nn.Sequential(
            BasicConv(128, 128, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1+1, dilation=2, relu=True),
            nn.AvgPool2d(kernel_size=8, stride=8),
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(256, 128, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1+1, dilation=2, relu=True),
            nn.AvgPool2d(kernel_size=4, stride=4),
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(512, 128, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv_block4 = nn.Sequential(
            BasicConv(512, 128, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.combine = nn.Sequential(
                                    BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
        )
        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, num_classes),
        # )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)

        # lstm
        # x_att = x.reshape((x.shape[0], 512, 49)).transpose(1,2)
        # x_att, (hidden, cell) = self.lstm(x_att)
        # x_att = nn.Softmax()(x_att)
        # # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/21
        # outs=[]
        # for i in range(x_att.shape[1]):
        #     outs.append(self.relu(self.timedist[i](x_att[:, i, :])).unsqueeze(axis = 1))
        # outs = torch.cat(outs, axis=1)
        # outs = torch.flatten(outs, 1)

        # # conv
        # x = torch.flatten(x, 1)
        # x = self.linear(x)
        # x = self.relu(x)

        # x_final = torch.cat((x, outs), axis = 1)

        # x_final = self.final_fc(x_final)

        # mha
        # b, c, h, w = x.shape
        # x_mha = self.to_patch_embedding(x)
        # x_mha_linear_proj = torch.tensor([], device = torch.device('cuda:1'))
        # for i in range(len(self.linear_proj)):
        #     output_tmp = self.linear_proj[i](x_mha[:, i:i+1])
        #     x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis = 1)
        # x_mha = x_mha_linear_proj
        # x_mha = self.pos_enc(x_mha)
        # x_mha = self.mha(x_mha, x_mha, x_mha, None)
        # x_mha = x_mha.reshape((x_mha.shape[0], h, w, c))
        # x_mha = x_mha.mean(axis = -1, keepdim = True)
        # x_mha = nn.Sigmoid()(x_mha)
        # x_mha = x_mha.permute((0, 3, 1, 2))
        # x_final = x * x_mha
        # x_final = torch.flatten(x_final, 1)
        # x_final = self.classifier(x_final)

        # PMG
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x1 = self.conv_block1(x1)
        x2 = self.conv_block2(x2)
        x3 = self.conv_block3(x3)
        x4 = self.conv_block4(x4)
        x4 = torch.cat([x1, x2, x3, x4], 1)
        x4 = self.combine(x4)
        # out = F.adaptive_avg_pool2d(x4, (1, 1))
        out = self.avgpool(x4)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
       
        return out

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


""" GoogLeNet """
GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


def googlenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> "GoogLeNet":
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        # model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
        
        # 1. filter out unnecessary keys
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and not k.startswith('classifier') and not k.startswith('fc') \
                and not k.startswith('aux1') and not k.startswith('aux2'):
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        return model

    return GoogLeNet(**kwargs)


class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None
    ) -> None:
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

        # mha
        self.pos_enc = PositionalEncoding(1024, 0, 7 * 7)
        self.mha = MultiHeadedAttention(1024, 8, 0, False)
        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1)
        self.linear_proj = nn.ModuleList([nn.Linear(1024, 1024)] * 49)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        # x = self.avgpool(x)
        # # N x 1024 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 1024
        # x = self.dropout(x)
        # x = self.fc(x)
        # N x 1000 (num_classes)

        # mha
        # x_mha = x.permute((0, 2, 3, 1))
        # b, h, w, c = x_mha.shape
        # x_mha = x_mha.reshape((x_mha.shape[0], h * w, c))
        b, c, h, w = x.shape
        x_mha = self.to_patch_embedding(x)
        x_mha_linear_proj = torch.tensor([], device = torch.device('cuda:0'))
        for i in range(len(self.linear_proj)):
            output_tmp = self.linear_proj[i](x_mha[:, i:i+1])
            x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis = 1)
        x_mha = x_mha_linear_proj
        x_mha = self.pos_enc(x_mha)
        x_mha = self.mha(x_mha, x_mha, x_mha, None)
        x_mha = x_mha.reshape((x_mha.shape[0], h, w, c))
        x_mha = x_mha.mean(axis = -1, keepdim = True)
        x_mha = nn.Sigmoid()(x_mha)
        x_mha = x_mha.permute((0, 3, 1, 2))
        x_final = x * x_mha
        x_final = self.avgpool(x_final)
        x_final = torch.flatten(x_final, 1)
        x_final = self.fc(x_final)

        return x_final, aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x   # type: ignore[return-value]

    def forward(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)


class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

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
        self.dropout = nn.Dropout()
        # self.fc = nn.Linear(2048, num_classes)
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

        self.conv_block1 = nn.Sequential(
            BasicConv(192, 512, kernel_size=1, stride=1, padding=0, relu=True, groups=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=0, dilation=3, relu=True, groups=1),
            nn.AvgPool2d(kernel_size=8, stride=8),
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(288, 512, kernel_size=1, stride=1, padding=0, relu=True, groups=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1, dilation=3, relu=True, groups=1),
            nn.AvgPool2d(kernel_size=4, stride=4),
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(768, 512, kernel_size=1, stride=1, padding=0, relu=True, groups=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=0, relu=True, groups=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv_block4 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True, groups=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1, relu=True, groups=1),
        )
        self.combine = nn.Sequential(
            BasicConv(2048, 2048, kernel_size=1, stride=1, padding=0, relu=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.pmg_fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            # nn.Linear(2048, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, num_classes),
            nn.Linear(2048, num_classes),
        )
        
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
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x1 = self.Conv2d_4a_3x3(x)
        
        # N x 192 x 71 x 71
        x2 = self.maxpool2(x1)
        # N x 192 x 35 x 35
        x2 = self.Mixed_5b(x2)
        # N x 256 x 35 x 35
        x2 = self.Mixed_5c(x2)
        # N x 288 x 35 x 35
        x2 = self.Mixed_5d(x2)
        
        # N x 288 x 35 x 35
        x3 = self.Mixed_6a(x2)
        # N x 768 x 17 x 17
        x3 = self.Mixed_6b(x3)
        # N x 768 x 17 x 17
        x3 = self.Mixed_6c(x3)
        # N x 768 x 17 x 17
        x3 = self.Mixed_6d(x3)
        # N x 768 x 17 x 17
        x3 = self.Mixed_6e(x3)
        
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        
        # N x 768 x 17 x 17
        x4 = self.Mixed_7a(x3)
        # N x 1280 x 8 x 8
        x4 = self.Mixed_7b(x4)
        # N x 2048 x 8 x 8
        x4 = self.Mixed_7c(x4)
        
        # # N x 2048 x 8 x 8
        # # Adaptive average pooling
        # x = self.avgpool(x)
        # # N x 2048 x 1 x 1
        # x = self.dropout(x)
        # # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 2048
        # x = self.fc(x)
        # # N x 1000 (num_classes)
    
        x1 = self.conv_block1(x1)
        x2 = self.conv_block2(x2)
        x3 = self.conv_block3(x3)
        x4 = self.conv_block4(x4)
        x4 = torch.cat([x1, x2, x3, x4], 1)
        B, C, H, W = x4.size()
        x4 = self.combine(x4).view(B, -1, H * W)
        x4 = self.gap(x4).squeeze(-1)        
        x = self.pmg_fc(x4)
        
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
        norm_layer: Optional[Callable[..., nn.Module]] = None
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

        # self.se = SE_Mine(r = 16, C = planes * 4)
        # self.se = SE_Mine1(r = 16, C = planes * 4, )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.se(out)

        # out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck_Inv(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_size: int = 14, # I added this
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck_Inv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # Involution(width, 3, stride), conv3x3(width, width, stride, groups, dilation)
        # self.conv2 = SELC(16, width, output_size, False, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

        # self.se = SE_Mine(r = 16, C = planes * 4)
        # self.se = SE_Mine1(r = 16, C = planes * 4, )

        self.selc = SELC(16, planes * self.expansion, output_size, False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.se(out)

        # out = self.eca(out)

        # out = self.cbam(out)

        out = self.selc(out)

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

        # Conv3
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        # self.bn1 = norm_layer(32)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2),
        #                             norm_layer(self.inplanes),
        #                             nn.ReLU(inplace=True))
        
        # conv-convt
        # self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2)
        # self.bn1 = norm_layer(128)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, 2),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(inplace=True))

        # Original
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, # 
                                       dilate=replace_stride_with_dilation[2])

        # self.layer1 = self._make_layer_Inv(Bottleneck_Inv, 64, layers[0], 56)
        # self.layer2 = self._make_layer_Inv(Bottleneck_Inv, 128, layers[1], 28, stride=2,
        #                                     dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer_Inv(Bottleneck_Inv, 256, layers[2], 14, stride=2,
        #                                     dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer_Inv(Bottleneck_Inv, 512, layers[3], 7, stride=2, # 
        #                                     dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  # nn.Linear(7*7*2048, num_classes) nn.Linear(512 * block.expansion, num_classes) 

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
        
        # lstm
        # self.outdim = 49
        # self.num_layers = 1
        # self.lstm  = nn.LSTM(input_size = 49, hidden_size = 2048, 
        #                     num_layers = self.num_layers, 
        #                     batch_first = True, bidirectional = False)
        # self.timedist = nn.ModuleList([nn.Linear(16, 64) for i in range(49)])
        # self.relu = nn.ReLU()

        # fc for conv
        # self.linear = nn.Linear(25088, 4096)

        # fc
        # self.final_fc = nn.Sequential(
        #     nn.Linear(7232, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        # mha
        # self.pos_enc = PositionalEncoding(2048, 0, 7 * 7) # 7
        # self.mha = MultiHeadedAttention(2048, 8, 0, False)

        # mha XCiT
        # self.pos_enc = PositionalEncoding(2048, 0, 128)
        # self.mha = MultiHeadedAttention_XCiT(2048, 8, 0, False)

        # mha channel
        # self.linear_proj0 = nn.ModuleList([nn.Linear(64, 64)] * 56 * 56)
        # self.pos_enc0 = PositionalEncoding(64, 0, 56 * 56)
        # self.mha0 = MultiHeadedAttention_ChannelTest(64, 1, 0, False) 
        # self.linear_proj1 = nn.ModuleList([nn.Linear(256, 256)] * 56 * 56)
        # self.pos_enc1 = PositionalEncoding(256, 0, 56 * 56)
        # self.mha1 = MultiHeadedAttention_ChannelTest(256, 1, 0, False)
        # self.linear_proj2 = nn.ModuleList([nn.Linear(512, 512)] * 28 * 28) 
        # self.pos_enc2 = PositionalEncoding(512, 0, 28 * 28)
        # self.mha2 = MultiHeadedAttention_ChannelTest(512, 1, 0, False) 
        # self.linear_proj3 = nn.ModuleList([nn.Linear(1024, 1024)] * 14 * 14)
        # self.pos_enc3 = PositionalEncoding(1024, 0, 14 * 14)
        # self.mha3 = MultiHeadedAttention_ChannelTest(1024, 1, 0, False) 
        # self.linear_proj4 = nn.ModuleList([nn.Linear(2048, 2048)] * 7 * 7)
        # self.pos_enc4 = PositionalEncoding(2048, 0, 7 * 7)
        # self.mha4 = MultiHeadedAttention_ChannelTest(2048, 1, 0, False)

        # self.pos_embedding = nn.Parameter(torch.randn(1, 7 * 7, 2048))

        # self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)
        # self.linear_proj = nn.ModuleList([nn.Linear(2048, 2048)] * 49) # 49

        # GFMA, FSRA
        # self.GFMA = GFMA(isChannel=True) # True
        # self.FSRA = FSRA()

        # SE is placed within bottleneck 
        
        # BAM
        # Refer link below for Official site from paper on how BAM in included in ResNet50
        # https://github.com/Jongchan/attention-module/blob/master/MODELS/model_resnet.py
        # self.bam1 = BAM(16, 256)
        # self.bam2 = BAM(16, 512)
        # self.bam3 = BAM(16, 1024)

        # CBAM is placed within bottleneck

        # self.att0 = SE_Mine(r = 16, C = 64)
        # self.att1 = SE_Mine(r = 16, C = 256)
        # self.att2 = SE_Mine(r = 16, C = 512)
        # self.att3 = SE_Mine(r = 16, C = 1024)
        # self.att4 = SE_Mine(r = 16, C = 2048)

        # self.att0 = SE_Mine1(r = 16, C = 64, S = 56 * 56)
        # self.att1 = SE_Mine1(r = 16, C = 256, S = 56 * 56)
        # self.att2 = SE_Mine1(r = 16, C = 512, S = 28 * 28)
        # self.att3 = SE_Mine1(r = 16, C = 1024, S = 14 * 14)
        # self.att4 = SE_Mine1(r = 16, C = 2048, S = 7 * 7)

        # self.att0 = MHA_SE(seq_len = 56 * 56, number_of_heads = 8, patch_size = 1, r = 16, C = 64)
        # self.att1 = MHA_SE(seq_len = 56 * 56, number_of_heads = 8, patch_size = 1, r = 16, C = 256)
        # self.att2 = MHA_SE(seq_len = 28 * 28, number_of_heads = 8, patch_size = 1, r = 16, C = 512)
        # self.att3 = MHA_SE(seq_len = 14 * 14, number_of_heads = 8, patch_size = 1, r = 16, C = 1024)
        # self.att4 = MHA_SE(seq_len = 7 * 7, number_of_heads = 8, patch_size = 1, r = 16, C = 2048)

        # self.att_test = att_test()
        # self.attn_test_linear = nn.Linear(1795, num_classes)

        # self.gap_weight = nn.Conv2d(2048, 2048 * 9, 1, 1)
        # self.swp = nn.Parameter(torch.normal(0, 0.005, size=(9, 7, 7)))
        # self.bn1_swp = nn.BatchNorm2d(num_features=9 * 2048)
        # self.conv_swp = nn.Conv2d(9 * 2048, 1024, 1, 1)
        # self.bn2_swp = nn.BatchNorm2d(num_features=1024)
        # self.linear2_swp = nn.Linear(1024, num_classes)

        # self.involution = Involution(channels = 2048, kernel_size = 1, stride = 1)

        # self.localconnect = LocallyConnected2d(in_channels = 2048, out_channels = 1, 
        #                                         output_size = 7, kernel_size = 1, 
        #                                         stride = 1, bias = True)

        # self.gap_weight = nn.Parameter(torch.rand(1, 1, 7, 7))

        # self.convtrans = nn.Sequential(nn.ConvTranspose2d(2048, 2048, 3, 1),
        #                                 nn.BatchNorm2d(2048),
        #                                 nn.ReLU(inplace=True),
        #                                 )

        # self.seq2seq_att = seq2seq_att()

        # self.CorBasedPool = CorBasedPool()

        # self.invse1 = InvolutionSE(size = 56, C = 256, r = 16, channel_per_group = 1, avg = False)
        # self.invse2 = InvolutionSE(size = 28, C = 512, r = 16, channel_per_group = 1, avg = False)
        # self.invse3 = InvolutionSE(size = 14, C = 1024, r = 16, channel_per_group = 1, avg = False)
        # self.invse4 = InvolutionSE(size = 7, C = 2048, r = 16, channel_per_group = 1, avg = False)

        # self.mha = MultiHeadedAttention_Conv1d(2048, 8, 0, False)

        # self.perceiver_like1 = Perceiver_like(q_dim = 256, q_seq_len = 56 * 56,
        #                                     kv_dim = 512, kv_seq_len = 28 * 28,
        #                                     number_of_heads = 8, patch_size = 2,
        #                                     reduce_by_conv = False, concat = True)
        # self.perceiver_like2 = Perceiver_like(q_dim = 512, q_seq_len = 28 * 28,
        #                                     kv_dim = 1024, kv_seq_len = 14 * 14,
        #                                     number_of_heads = 8, patch_size = 2,
        #                                     reduce_by_conv = False, concat = True)
        # self.perceiver_like3 = Perceiver_like(q_dim = 1024, q_seq_len = 14 * 14,
        #                                     kv_dim = 2048, kv_seq_len = 7 * 7,
        #                                     number_of_heads = 8, patch_size = 2,
        #                                     reduce_by_conv = False, concat = True)
        
        # self.perceiver_like1 = Perceiver_like(q_dim = 512, q_seq_len = 28 * 28,
        #                                     kv_dim = 256, kv_seq_len = 56 * 56, 
        #                                     number_of_heads = 8, patch_size = 2,
        #                                     reduce_by_conv = True, concat = True)
        # self.perceiver_like2 = Perceiver_like(q_dim = 1024, q_seq_len = 14 * 14,
        #                                     kv_dim = 512, kv_seq_len = 28 * 28,
        #                                     number_of_heads = 8, patch_size = 2,
        #                                     reduce_by_conv = True, concat = True)
        # self.perceiver_like3 = Perceiver_like(q_dim = 2048, q_seq_len = 7 * 7,
        #                                     kv_dim = 1024, kv_seq_len = 14 * 14,
        #                                     number_of_heads = 8, patch_size = 2,
        #                                     reduce_by_conv = True, concat = True)
        
        # self.perceiver_HT = Perceiver_HT(q_dim = 2048, q_seq_len = 7 * 7,
        #                                     kv_dim = 64, kv_seq_len = 56 * 56,
        #                                     number_of_heads = 8)

        # self.part_att = PartAttention(2048, 1536)

        self.conv_block0 = nn.Sequential(
            BasicConv(256, 512, kernel_size=1, stride=1, padding=0, relu=True, groups=1),
            # BasicConv(16, 16, kernel_size=3, stride=2, padding=1, relu=True),
            # BasicConv(16, 16, kernel_size=3, stride=2, padding=1, relu=True),
            # ShuffleChannel(groups=8),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=3, dilation=3, relu=True, groups=1),
            # BasicConv(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, relu=True),
            # Involution(512, 3, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # MultiConvKernel(512, True),
            # BasicConv(16, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(256, 512, kernel_size=1, stride=1, padding=1, relu=True),
            # BasicConv(256, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(2048, 2048, kernel_size=3, stride=1, padding=1, relu=True),
            # BasicConv(256, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(512, 512, kernel_size=3, stride=1, padding=1, relu=True, isInvolution=True),
            # nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(kernel_size=8, stride=8),
            # SELC(16, 512, 7, False, 1, 3),
            # SE(8, 512)
            # nn.MaxPool2d(kernel_size=8, stride=8)
            # BasicConv_v1(256, 512, True),
            # BasicConv_v1(512, 512, True),
            # BasicConv_v1(512, 512, True)
        )
        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True, groups=1),
            # BasicConv(32, 32, kernel_size=3, stride=2, padding=1, relu=True),
            # ShuffleChannel(groups=8),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=3, dilation=3, relu=True, groups=1),
            # BasicConv(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, relu=True),
            # Involution(512, 3, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # MultiConvKernel(512, True),
            # BasicConv(32, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(512, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(2048, 2048, kernel_size=3, stride=1, padding=1, relu=True),
            # BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(512, 512, kernel_size=3, stride=1, padding=1, relu=True, isInvolution=True),
            # nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(kernel_size=4, stride=4),
            # nn.MaxPool2d(kernel_size=4, stride=4)
            # SELC(16, 512, 7, False, 1, 3),
            # SE(8, 512),
            # BasicConv_v1(512, 512, True),
            # BasicConv_v1(512, 512, True),
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True, groups=1),
            # ShuffleChannel(groups=8),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1, relu=True, groups=1),
            # BasicConv(1024, 682, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(682, 682, kernel_size=3, stride=1, padding=1, relu=True),
            # Involution(512, 3, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # MultiConvKernel(512, False),
            # BasicConv(64, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(1024, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(2048, 2048, kernel_size=3, stride=1, padding=1, relu=True),
            # BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(512, 512, kernel_size=3, stride=1, padding=1, relu=True, isInvolution=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # nn.AdaptiveAvgPool2d(1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            # SELC(16, 512, 7, False, 1, 3),
            # SE(8, 512),
            # BasicConv_v1(1024, 512, True),
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True, groups=1),
            # ShuffleChannel(groups=8),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1, relu=True, groups=1),
            # BasicConv(2048, 682, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(682, 682, kernel_size=3, stride=1, padding=1, relu=True),
            # Involution(512, 3, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # MultiConvKernel(512, False),
            # BasicConv(128, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(2048, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(2048, 2048, kernel_size=3, stride=1, padding=1, relu=True)
            # BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(512, 512, kernel_size=3, stride=1, padding=1, relu=True, isInvolution=True),
            # SELC(16, 512, 7, False, 1, 3),
            # SE(8, 512)
            # nn.AdaptiveAvgPool2d(1),
            # BasicConv_v1(2048, 512, False),
        )
        self.combine = nn.Sequential(
                                    # BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
                                    # BasicConv(512, 2048, kernel_size=3, stride=1, padding=1, relu=True),
                                    BasicConv(2048, 2048, kernel_size=1, stride=1, padding=0, relu=True),
                                    # BasicConv(2048, 2048, kernel_size=3, stride=1, padding=1, relu=True),
                                    # nn.Sigmoid(),
                                    # MultiConvKernel(2048),
                                    # Involution(2048, 3, 1),
                                    # nn.BatchNorm2d(2048),
                                    # nn.ReLU(inplace=True)
                                    # SE_avgmax(16, 2048),
                                    # SE(16, 2048),
                                    # SimpleChannelMHA(2048)
                                    # nn.Dropout2d(0.2, True),
        )
        # self.combine = nn.LSTM(input_size=512, hidden_size=2048, batch_first=True)
        # self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
        #                                     p1 = 1, p2 = 1)
        # self.pos_enc = PositionalEncoding(2048, 0, 49)
        # # self.pos_emb = nn.Parameter(torch.zeros(1, 49, 2048)) 
        # self.mha = MultiHeadedAttention(2048, 8, 0, False)
        # self.xcit = MultiHeadedAttention_XCiT(2048, 8, 0, False)
        # self.mha_test = MultiHeadedAttention_ChannelTest(2048, 8, 0, False)
        # self.se = nn.Sequential(
        #                         nn.Linear(2048, 2048 // 16),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(2048 // 16, 2048),
        #                         nn.Sigmoid()
        # )
        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.swp = nn.Parameter(torch.normal(0, 0.005, size=(1, 7, 7)))
        # self.swp = nn.Parameter(torch.zeros((1, 7, 7)))
        # self.swp = nn.Linear(49, 1, bias=False)
        # self.swp_bn = nn.BatchNorm1d(2048)
        # self.selc = SELC(16, 2048, 7, False, 1, 3)
        self.pmg_fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.SiLU(inplace=True),
            nn.Linear(512, num_classes),
            # nn.Linear(2048, num_classes),
        )
        # self.mha1 = MultiHeadedAttention(2048, 8, 0, False)

        # self.cor = Cor()

        # self.layer3 = MultiHeadedAttention(512, 8, 0, False)
        # self.layer3_mlp = Mlp(512)
        # self.layer4 = MultiHeadedAttention(1024, 8, 0, False)
        # self.layer4_mlp = Mlp(1024)
        # self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.ext_att = ExternalAttention(2048)
        # self.mhse1 = MHSE(4, 256, 56, 56, 4)
        # self.mhse2 = MHSE(4, 512, 28, 28, 4)
        # self.mhse3 = MHSE(4, 1024, 14, 14, 4)
        # self.mhse4 = MHSE(4, 2048, 7, 7, 4)

        # self.selc1 = SELC(8, 256, 56, False, 1, 3)
        # self.selc2 = SELC(8, 512, 28, False, 1, 3)
        # self.selc3 = SELC(16, 1024, 14, False, 1, 3)
        # self.selc4 = SELC(16, 2048, 7, False, 1, 3)
        # self.xcit = MultiHeadedAttention_XCiT(2048, 8, 0, False)
        # self.fc_selc = nn.Sequential(nn.Linear(2048, 512),
        #                             nn.BatchNorm1d(512),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(512, num_classes))

        # self.fc_dropout = nn.Dropout(0.2, inplace=True)

        # self.att1 = SE(16, 256) # mhase(256, 16)
        # self.att2 = SE(16, 512) # mhase(512, 16)
        # self.att3 = SE(16, 1024) # mhase(1024, 16)
        # self.att4 = SE(16, 2048) # mhase(2048, 16)

        # self.dual_att = dual_att(16, 2048, 8, 1)

        # self.ca = CA(2048)

        # self.spp = SpatialPyramidPooling([1,2,3], 'avg')
        # self.tpp = TemporalPyramidPooling([1,2,3], 'avg')

        # self.cbam = CBAM(16, 2048)

        # self.mha_like_ca = MHA_like_CA()

        # self.aam1 = AggregatedAttentionModule(256)
        # self.aam2 = AggregatedAttentionModule(512)
        # self.aam3 = AggregatedAttentionModule(1024)
        # self.aam4 = AggregatedAttentionModule(2048)

        # self.vanillasa = VanillaSA(2048)

        # stand alone stn using resnet18
        # self.resnet18 = models.resnet18(pretrained=True, num_classes=1000)
        # self.resnet18.avgpool = nn.Sequential(
        #     nn.Conv2d(512, 128, 1),
        #     nn.Flatten(start_dim=1)
        # )
        # self.resnet18.fc = nn.Sequential(
        #     nn.Linear(128 * 7 * 7, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 4), # 3 * 2
        #     nn.Tanh(),
        # )

        # self.gru = nn.GRU(2048, 512, batch_first=True)
        # self.convlstm = ConvLSTM(
        #     input_dim=512, hidden_dim=512, kernel_size=(3, 3), num_layers=1, 
        #     batch_first=True, bias=True, return_all_layers=True
        #     )

        # self.att = VanillaSA1()

        # self.spinalnet = SpinalNet_ResNet(431, 431)

        # self.select_features = SelectFeatures(2048, num_classes, 32)

        # num_joints = torch.tensor([2048, 512, 128, 32])/torch.tensor([112**2, 56**2, 28**2, 14**2])
        # num_joints *= torch.tensor([56**2, 28**2, 14**2, 7**2])
        # self.gcn_sa1 = GCN_SA(num_joints=int(num_joints[0].item()), in_features=256)
        # self.gcn_sa2 = GCN_SA(num_joints=int(num_joints[1].item()), in_features=512)
        # self.gcn_sa3 = GCN_SA(num_joints=int(num_joints[2].item()), in_features=1024)
        # self.gcn_sa4 = GCN_SA(num_joints=int(num_joints[3].item()), in_features=2048)

        # self.sa2 = MobileViTBlock(
        #     kernel_size=3, spatial_size=28, channel_size=512,
        #     number_of_heads=4)
        # self.sa3 = MobileViTBlock(
        #     kernel_size=2, spatial_size=14, channel_size=1024,
        #     number_of_heads=8)
        # self.sa4 = MobileViTBlock(
        #     kernel_size=2, spatial_size=7, channel_size=2048,
        #     number_of_heads=8)
        # self.ca2 = SE(16, 512)
        # self.ca3 = SE(16, 1024)
        # self.ca4 = SE(16, 2048)

        # self.mldecoder = MLDecoder(
        #     num_classes, num_of_groups=100, decoder_embedding=768, 
        #     initial_num_features=2048, zsl=0)
        
        # self.attentions = BasicConv2d(2048, 32, kernel_size=1)

    def stn(self, x):
        # stn using conv shared with resnet50
        # xs = self.conv1(x)
        # xs = self.bn1(xs)
        # xs = self.relu(xs)
        # xs = self.maxpool(xs)
        # xs = self.layer1(xs)
        # xs = self.layer2(xs)

        # stn using resnet18
        theta = self.resnet18(x)

        # fc
        # xs = nn.AdaptiveAvgPool2d(1)(xs).squeeze()
        # theta = theta.view(-1, 2, 3)
        with torch.no_grad():
            theta_new = torch.zeros(theta.size(0), 2, 3, device=theta.device)
            theta_new[:,0,0], theta_new[:,1,1], theta_new[:,0,2], theta_new[:,1,2] =\
                theta[:,0], theta[:,1], theta[:,2], theta[:,3]
            grid = F.affine_grid(theta_new, x.size())
            x = F.grid_sample(x, grid)

        return x

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_Inv(self, block: Type[Union[BasicBlock, Bottleneck_Inv]], planes: int, blocks: int, 
                    output_size: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
        layers.append(block(self.inplanes, planes, stride, output_size, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, output_size = output_size,
                                groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # x = self.stn(x)

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        # GFMA
        # x1 = self.GFMA(x1)

        # x = self.att0(x)

        # mha channel 0
        # x_mha = x.permute((0, 2, 3, 1))
        # b, h, w, c = x_mha.shape
        # x_mha = x_mha.reshape((x_mha.shape[0], h * w, c))
        # b, c, h, w = x.shape
        # x_mha = self.to_patch_embedding(x)
        # x_mha_linear_proj = torch.tensor([], device = torch.device('cuda:1'))
        # for i in range(len(self.linear_proj0)):
        #     output_tmp = self.linear_proj0[i](x_mha[:, i:i+1])
        #     x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis = 1)
        # x_mha = x_mha_linear_proj
        # x_mha = self.pos_enc0(x_mha)
        # x = self.mha0(x_mha, x_mha, x_mha, None)
        # x_mha = nn.Sigmoid()(x_mha)
        # x_mha = x.reshape((b, -1, 1, 1))
        # x = x * x_mha
        # x = x.reshape(b, c, h, w)
        x2 = self.layer1(x1)
        # x2 = self.mhse1(x2)
        # mha channel 1
        # x_mha = x.permute((0, 2, 3, 1))
        # b, h, w, c = x_mha.shape
        # x_mha = x_mha.reshape((x_mha.shape[0], h * w, c))
        # b, c, h, w = x.shape
        # x_mha = self.to_patch_embedding(x)
        # x_mha_linear_proj = torch.tensor([], device = torch.device('cuda:1'))
        # for i in range(len(self.linear_proj1)):
        #     output_tmp = self.linear_proj1[i](x_mha[:, i:i+1])
        #     x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis = 1)
        # x_mha = x_mha_linear_proj
        # x_mha = self.pos_enc1(x_mha)
        # x = self.mha1(x_mha, x_mha, x_mha, None)
        # x_mha = nn.Sigmoid()(x_mha)
        # x_mha = x_mha.reshape((b, -1, 1, 1))
        # x = x * x_mha
        # x = x.reshape(b, c, h, w)

        # x2 = self.bam1(x2)

        # x2 = self.att1(x2)

        # x = self.invse1(x)

        # x2 = self.selc1(x2)

        # x2 = self.aam1(x2)

        # x2 = self.gcn_sa1(x2)

        x3 = self.layer2(x2)
        # x3 = self.mhse2(x3)
        # mha channel2
        # x_mha = x.permute((0, 2, 3, 1))
        # b, h, w, c = x_mha.shape
        # x_mha = x_mha.reshape((x_mha.shape[0], h * w, c))
        # b, c, h, w = x.shape
        # x_mha = self.to_patch_embedding(x)
        # x_mha_linear_proj = torch.tensor([], device = torch.device('cuda:1'))
        # for i in range(len(self.linear_proj2)):
        #     output_tmp = self.linear_proj2[i](x_mha[:, i:i+1])
        #     x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis = 1)
        # x_mha = x_mha_linear_proj
        # x_mha = self.pos_enc2(x_mha)
        # x = self.mha2(x_mha, x_mha, x_mha, None)
        # x_mha = nn.Sigmoid()(x_mha)
        # x_mha = x_mha.reshape((b, -1, 1, 1))
        # x = x * x_mha
        # x = x.reshape(b, c, h, w)

        # x3 = self.bam2(x3)

        # x3 = self.att2(x3)

        # x = self.invse2(x)

        # x2 = self.perceiver_like1(x2, x1)

        # x3 = self.selc2(x3)

        # x3 = self.aam2(x3)

        # x3 = self.gcn_sa2(x3)

        # x3 = self.sa2(x3)
        # x3 = x3 + self.ca2(x3)

        # x3 = x3.view(x3.size(0), x3.size(1), -1).transpose(1, 2)
        x4 = self.layer3(x3)
        # x3 = self.layer3_mlp(x3)
        # x3 = x3.transpose(1, 2)
        # x4 = self.maxpool1d(x3)
        # mha channel 3
        # x_mha = x.permute((0, 2, 3, 1))
        # b, h, w, c = x_mha.shape
        # x_mha = x_mha.reshape((x_mha.shape[0], h * w, c))
        # b, c, h, w = x.shape
        # x_mha = self.to_patch_embedding(x)
        # x_mha_linear_proj = torch.tensor([], device = torch.device('cuda:1'))
        # for i in range(len(self.linear_proj3)):
        #     output_tmp = self.linear_proj3[i](x_mha[:, i:i+1])
        #     x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis = 1)
        # x_mha = x_mha_linear_proj
        # x_mha = self.pos_enc3(x_mha)
        # x = self.mha3(x_mha, x_mha, x_mha, None)
        # x_mha = nn.Sigmoid()(x_mha)
        # x_mha = x_mha.reshape((b, -1, 1, 1))
        # x = x * x_mha
        # x = x.reshape(b, c, h, w)

        # x4 = self.bam3(x4)

        # x4 = self.att3(x4)

        # x = self.invse3(x)

        # x3 = self.perceiver_like2(x3, x2)

        # x4 = self.mhse3(x4)

        # x4 = self.selc3(x4)

        # x4 = self.aam3(x4)

        # x4 = x4.transpose(1, 2)

        # x4 = self.gcn_sa3(x4)

        # x4 = self.sa3(x4)
        # x4 = x4 + self.ca3(x4)

        x5_ori = self.layer4(x4)
        # x4 = self.layer4_mlp(x4)
        # x4 = x4.transpose(1, 2)
        # x5 = self.maxpool1d(x4)
        
        # x5_ori = self.att4(x5_ori)

        # x5 = self.dual_att(x5)

        # x = self.att_test(x)

        # x5_ori = self.selc4(x5_ori)

        # x5_ori = self.FSRA(x5_ori)

        # x5_ori = self.gcn_sa4(x5_ori)

        # x5 = self.to_patch_embedding(x5)
        # x5 = self.xcit(x5, x5, x5, None)
        # x5 = x5.transpose(1,2).view(-1, 2048, 7, 7)

        # x5 = x5.transpose(1,2).view(x5.size(0), 2048, 14, 14)
        # x5 = self.ext_att(x5)
        # x5 = self.mhse4(x5)
        # x_final = self.avgpool(x5)
        # swp_tmp = nn.Softmax(1)(self.swp.reshape(1, 49)).reshape(1, 7, 7)
        # swp_tmp = nn.Sigmoid()(self.swp)
        # x_final = (x5 * swp_tmp).sum(2).sum(2)
        # x_final = nn.ReLU(inplace=True)(self.swp_bn(self.swp(x5.view(-1, 2048, 49))))

        # lstm
        # x_att = x.reshape((x.size(0), x.size(1), -1)).transpose(1,2)
        # x_att, (hidden, cell) = self.lstm(x_att)
        # # x_att = nn.Softmax()(x_att)
        # # # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/21
        # # outs=[]
        # # for i in range(x_att.shape[1]):
        # #     outs.append(self.relu(self.timedist[i](x_att[:, i, :])).unsqueeze(axis = 1))
        # # outs = torch.cat(outs, axis=1)
        # # outs = torch.flatten(outs, 1)
        # hidden = hidden.view(hidden.size(1), 1, int(hidden.size(2) ** 0.5), int(hidden.size(2) ** 0.5))
        # hidden = nn.ReLU(inplace=True)(hidden)
        # x = x * hidden

        # x_att = x.reshape((x.size(0), x.size(1), -1))
        # x_att, (hidden, cell) = self.lstm(x_att)
        # hidden = hidden.view(hidden.size(1), hidden.size(2), 1, 1)
        # hidden = nn.ReLU(inplace=True)(hidden)
        # x = x * hidden

        # # conv
        # x = torch.flatten(x, 1)
        # x = self.linear(x)
        # x = self.relu(x)

        # x_final = torch.cat((x, outs), axis = 1)

        # x_final = torch.flatten(x_final, 1)
        # x_final = self.fc_selc(x_final)
        # x_final = self.fc(x_final)
        # x_final = self.fc_dropout(x_final)

        # mha
        # # x_mha = x.permute((0, 2, 3, 1))
        # # b, h, w, c = x_mha.shape
        # # x_mha = x_mha.reshape((x_mha.shape[0], h * w, c))
        # b, c, h, w = x5_ori.size()
        # x_mha = self.to_patch_embedding(x5_ori)
        # x_mha_linear_proj = torch.tensor([], device=x_mha.device)
        # for i in range(len(self.linear_proj)):
        #     output_tmp = self.linear_proj[i](x_mha[:, i:i+1])
        #     x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis=1)
        # x_mha = x_mha_linear_proj
        # x_mha = self.pos_enc(x_mha) # pos encoding
        # # pos_emb = self.pos_embedding[:, :(h * w)] # pos embeddings
        # # x_mha = x_mha + pos_emb
        # x_mha = self.mha(x_mha, x_mha, x_mha, None)
        # x_mha = x_mha.reshape(b, h, w, c)
        # x_mha = x_mha.mean(axis=-1, keepdim=True)
        # x_mha = nn.Sigmoid()(x_mha)
        # x_mha = x_mha.permute((0, 3, 1, 2))
        # x_final = x5_ori * x_mha

        # mha channel
        # x_mha = x.permute((0, 2, 3, 1))
        # b, h, w, c = x_mha.shape
        # x_mha = x_mha.reshape((x_mha.shape[0], h * w, c))
        # b, c, h, w = x.shape
        # x_mha = self.to_patch_embedding(x)
        # x_mha_linear_proj = torch.tensor([], device = torch.device('cuda:1'))
        # for i in range(len(self.linear_proj4)):
        #     output_tmp = self.linear_proj4[i](x_mha[:, i:i+1])
        #     x_mha_linear_proj = torch.cat((x_mha_linear_proj, output_tmp), axis = 1)
        # x_mha = x_mha_linear_proj
        # x_mha = self.pos_enc4(x_mha)
        # x = self.mha4(x_mha, x_mha, x_mha, None)
        # x_mha = nn.Sigmoid()(x_mha)
        # x_mha = x_mha.reshape((b, -1, 1, 1))
        # x_final = x * x_mha
        # x = x.reshape(b, c, h, w)

        # FSRA
        # x_final = self.FSRA(x5_ori)

        # x_final = self.att4(x)

        # x = self.att_test(x)

        # x = self.gap_weight(x)
        # gap_weight = self.involution(x)
        # gap_weight = self.localconnect(x)
        # B, C, H, W = gap_weight.size()
        # gap_weight = gap_weight.view(B, -1)
        # gap_weight = nn.Sigmoid()(gap_weight)
        # gap_weight = gap_weight.view(B, C, H, W)
        # x_final = x * gap_weight
        # x_final = x_final.sum(axis = (2, 3))

        # x_convtrans = self.convtrans(x)
        # x = self.seq2seq_att(x, x_convtrans)

        # x = self.invse4(x)
        
        # x = self.perceiver_like3(x4, x3)

        # x5 = self.perceiver_HT(x4, x)

        # x_final = self.part_att(x)

        # x5_ori = self.aam4(x5_ori)

        # x5_ori = self.sa4(x5_ori)
        # x5_ori = x5_ori + self.ca4(x5_ori)

        # PMG
        # x5_ori = self.to_patch_embedding(x5)
        # x5_ori = self.pos_enc(x5_ori)
        x2 = self.conv_block0(x2)
        x3 = self.conv_block1(x3)
        x4 = self.conv_block2(x4)
        x5 = self.conv_block3(x5_ori)
        # x5 = x2 * x3 * x4 * x5
        x5 = torch.cat([x2, x3, x4, x5], 1)
        # x5 = torch.stack([x2, x3, x4, x5], 1)
        # B, C, H, W = x5.size()
        # output, state = self.combine(x5.squeeze())
        # x5, cell_state = state
        # x5 = x5.squeeze()
        x5 = self.combine(x5).view(B, -1, H * W)
        # x5 = self.vanillasa(self.combine(x5)).view(B, -1, H * W)
        # x5 = (x5_ori + x5_ori * self.combine(x5)).view(B, -1, H * W)
        # x5 = (x5 + self.cbam(x5)).view(B, -1, H * W)
        # x5 = self.combine(x5).view(B, C, -1).transpose(1, 2)
        # x5 = self.to_patch_embedding(x5)
        # x5 = self.pos_enc(x5)
        # x5 = x5 + self.pos_emb
        # x5 = (x5 + self.mha(x5, x5 , x5, None)).transpose(1, 2) # add relu after mha?
        # x5 = self.xcit(x5, x5, x5, None)
        # x5 = self.mha_test(x5, x5, x5, None)
        # x5 = self.selc(x5).view(B, C, -1)
        # x5 = nn.ReLU(inplace=True)(x5)
        # x5 = self.mha1(x5, x5_ori, x5_ori, None)
        # x5 = x5 + x5 * self.se(x5.mean(1)).view(B, -1, C)
        # x5 = self.GFMA(x5).view(B, C, -1)
        # x5 = (x5 + self.ca(x5))
        # x5 = nn.ReLU(inplace=True)(self.gru(x5.transpose(1, 2))[1].permute(1, 2, 0))
        # all_hidden_states, last_hidden_cell_state, = self.convlstm(x5.flip(1))
        # last_hidden_state, last_cell_state = last_hidden_cell_state[0]
        # x5 = nn.Flatten(2)(last_hidden_state)
        # x5 = self.att(x5_ori, x5, x5)
        x5 = self.gap(x5).squeeze(-1)
        # x5 = self.spp(x5.view(B, -1, H, W))
        # x5 = self.tpp(x5.view(B, -1, H, W))
        # x5 = (self.swp * x5).sum((2, 3))
        # x5 = self.swp(x5.transpose(1, 2)).squeeze()
        # x5 = x5.transpose(1, 2).reshape(-1, 2048, 7, 7)
        # x5 = torch.sum(x5 * self.swp, dim = (2, 3))
        
        x_final = self.pmg_fc(x5)
        # x_final = self.spinalnet(x5)

        # x5_ori = self.mha_like_ca(x5_ori)
        
        # attention_maps = self.attentions(x5_ori)

        # x_final = self.avgpool(x5_ori) # x5_ori
        # x_final = x_final.squeeze() + x5
        # x_final = torch.cat([x_final.squeeze(), x5], -1)
        # x_final = self.CorBasedPool(x)
        # x_final = x.mean(dim=1)
        # x_final = self.gap_weight * x
        # x_final = x_final.mean(axis = (2, 3))
        # x_final = torch.flatten(x_final, 1)
        # x_final = self.fc(x_final)
        # x_final = self.attn_test_linear(x_final)

        # swp
        # B, C, H, W = x5.size()
        # x_swp = []
        # for layer in self.swp:
        #     x_swp.append(layer.view(1, 1, H, W) * x5)
        # x5=torch.cat(x_swp, axis = 1)
        # x5=x5.sum(axis=(2, 3), keepdim=True)
        # x5=self.bn1_swp(x5)
        # x5=nn.ReLU()(x5)
        # x5=self.conv_swp(x5)
        # x5=self.bn2_swp(x5)
        # x5=nn.ReLU()(x5)
        # x5=nn.Flatten()(x5)
        # x_final=self.linear2_swp(x5)

        # x_final = self.attn_test_linear(x_final)
        
        # self.cor(x2, x3, x4, x5)

        # x_final = self.select_features(x5.view(B, C, H, W))

        # x_final = self.mldecoder(x5_ori)
        
        return x_final

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


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
            if k in model_dict and not k.startswith('classifier') and not k.startswith('fc'): # and not k.startswith('conv1') and not k.startswith('bn1'):
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    
    # Fine-tune only added layer
    # for param in model.conv1.parameters():
    #     param.requires_grad = False
    # for param in model.bn1.parameters():
    #     param.requires_grad = False
    # for param in model.layer1.parameters():
    #     param.requires_grad = False
    # for param in model.layer2.parameters():
    #     param.requires_grad = False
    # for param in model.layer3.parameters():
    #     param.requires_grad = False
    # for param in model.layer4.parameters():
    #     param.requires_grad = False

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

        # PMG for DenseNet169
        self.conv_block1 = nn.Sequential(
            BasicConv(128, 416, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(416, 416, kernel_size=3, stride=1, padding=3, dilation=3, relu=True),
            nn.AvgPool2d(kernel_size=4, stride=4),
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(256, 416, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(416, 416, kernel_size=3, stride=1, padding=3, dilation=3, relu=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(640, 416, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(416, 416, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.conv_block4 = nn.Sequential(
            BasicConv(1664, 416, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(416, 416, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.combine = nn.Sequential(
                                    BasicConv(1664, 1664, kernel_size=1, stride=1, padding=0, relu=True),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1664),
            # nn.Linear(1664, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, num_classes),
            nn.Linear(1664, num_classes),
        )

        # PMG for DenseNet161
        # self.conv_block1 = nn.Sequential(
        #     BasicConv(192, 552, kernel_size=1, stride=1, padding=0, relu=True),
        #     BasicConv(552, 552, kernel_size=3, stride=1, padding=1+1, dilation=2, relu=True),
        #     nn.AvgPool2d(kernel_size=4, stride=4),
        # )
        # self.conv_block2 = nn.Sequential(
        #     BasicConv(384, 552, kernel_size=1, stride=1, padding=0, relu=True),
        #     BasicConv(552, 552, kernel_size=3, stride=1, padding=1+1, dilation=2, relu=True),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        # )
        # self.conv_block3 = nn.Sequential(
        #     BasicConv(1056, 552, kernel_size=1, stride=1, padding=0, relu=True),
        #     BasicConv(552, 552, kernel_size=3, stride=1, padding=1, relu=True),
        # )
        # self.conv_block4 = nn.Sequential(
        #     BasicConv(2208, 552, kernel_size=1, stride=1, padding=0, relu=True),
        #     BasicConv(552, 552, kernel_size=3, stride=1, padding=1, relu=True),
        # )
        # self.combine = nn.Sequential(
        #                             BasicConv(2208, 2208, kernel_size=1, stride=1, padding=0, relu=True),
        # )
        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(2208),
        #     nn.Linear(2208, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, num_classes),
        # )


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

        x1 = self.features.denseblock1(x1)
        x1 = self.features.transition1(x1)

        x2 = self.features.denseblock2(x1)
        x2 = self.features.transition2(x2)

        x3 = self.features.denseblock3(x2)
        x3 = self.features.transition3(x3)

        x4 = self.features.denseblock4(x3)
        x4 = self.features.norm5(x4)
        x4 = F.relu(x4, inplace=True)

        # PMG
        x1 = self.conv_block1(x1)
        x2 = self.conv_block2(x2)
        x3 = self.conv_block3(x3)
        x4 = self.conv_block4(x4)
        x4 = torch.cat([x1, x2, x3, x4], 1)
        x4 = self.combine(x4)
        out = F.adaptive_avg_pool2d(x4, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


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


if __name__ == '__main__':

    import torch.optim as optim

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    x_train = torch.rand((64, 3, 224, 224)).to(device)
    labels = torch.randint(0, 6, (64,)).to(device)
    model = resnet50(pretrained=False, num_classes=6)
    # model = densenet161(pretrained=False, num_classes=6)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()