from functools import partial

import torch
import torch.nn as nn
from collections import OrderedDict
from .layers.anti_aliasing import AntiAliasDownsampleLayer
from .layers.avg_pool import FastGlobalAvgPool2d
from .layers.squeeze_and_excite import SEModule
from .layers.space_to_depth import SpaceToDepthModule
from inplace_abn import InPlaceABN
from ..ml_decoder import MLDecoder


model_urls = {
    # pretrained on imagenet
    "tresnet_m_224": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_m.pth",
    "tresnet_m_448": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_m_448.pth",
    "tresnet_l_224": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_l.pth",
    "tresnet_l_448": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_l_448.pth",
    "tresnet_xl_224": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_xl.pth",
    "tresnet_xl_448": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_xl_448.pth",

    # pretrained on stanford car
    "tresnet_l_v2_368": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/stanford_cars_tresnet-l-v2_96_27.pth",
    
    # pretrained on mscoco
    # "tresnet_m_mldecoder_224": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_m_COCO_224_84_2.pth",
    "tresnet_l_mldecoder_448": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_COCO__448_90_0.pth",
    "tresnet_xl_mldecoder_640": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_xl_COCO_640_91_4.pth",

    # pretrained on openimages
    "tresnet_m_mldecoder_224": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_m_open_images_200_groups_86_8.pth",

    # pretrained on stanford-car
    # "tresnet_l_mldecoder_384": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_stanford_card_96.41.pth",
}

def IABN2Float(module: nn.Module) -> nn.Module:
    "If `module` is IABN don't use half precision."
    if isinstance(module, InPlaceABN):
        module.float()
    for child in module.children(): IABN2Float(child)
    return module


def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(nn.Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, remove_aa_jit=False,
                 first_two_layers=BasicBlock, use_my_model=True, use_ml_decoder=False, **kwargs):
        super(TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=remove_aa_jit)
        global_pool_layer = nn.Identity() if use_ml_decoder else FastGlobalAvgPool2d(flatten=True)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(first_two_layers,
                                  self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(first_two_layers,
                                  self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        if use_ml_decoder:
            self.head = MLDecoder(
                num_classes=num_classes, initial_num_features=self.num_features,
                num_of_groups=kwargs['num_of_groups'], decoder_embedding=kwargs['decoder_embedding'],
                zsl=kwargs['zsl']
            )
        else:
            fc = nn.Linear(self.num_features, num_classes)
            self.head = nn.Sequential(OrderedDict([('fc', fc)]))

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits


def TResnetS(model_params):
    """Constructs a small TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    args = model_params['args']
    model = TResNet(layers=[3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans)
    return model


def TResnetM(pretrained, num_classes, remove_aa_jit=False, use_my_model=True, use_ml_decoder=False, **kwargs):
    """ Constructs a medium TResnet model.
    """
    in_chans = 3
    # num_classes = model_params['num_classes']
    # remove_aa_jit = model_params['remove_aa_jit']
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans,
                    remove_aa_jit=remove_aa_jit, use_my_model=use_my_model,
                    use_ml_decoder=use_ml_decoder, **kwargs)
    
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls.get('tresnet_m_mldecoder_224' if use_ml_decoder else 'tresnet_m_448'),
            map_location='cpu'
        )['model']
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and not k.startswith('classifier') and not k.startswith('fc') and 'head' not in k:
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


def TResnetL(pretrained, num_classes, remove_aa_jit=False, use_my_model=True, use_ml_decoder=False, **kwargs):
    """ Constructs a large TResnet model.
    """
    in_chans = 3
    # num_classes = model_params['num_classes']
    # remove_aa_jit = model_params['remove_aa_jit']
    layers = [3, 4, 23, 3] if use_ml_decoder else [4, 5, 18, 3]
    model = TResNet(layers=layers, num_classes=num_classes, in_chans=in_chans,
                    width_factor=1 if use_ml_decoder else 1.2,
                    remove_aa_jit=remove_aa_jit,
                    first_two_layers=Bottleneck if use_ml_decoder else BasicBlock,
                    use_my_model=use_my_model,
                    use_ml_decoder=use_ml_decoder, **kwargs)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls.get('tresnet_l_mldecoder_448' if use_ml_decoder else 'tresnet_l_448'),
            map_location='cpu'
        )['model']
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and not k.startswith('classifier') and not k.startswith('fc') and 'head' not in k:
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


def TResnetXL(pretrained, num_classes, remove_aa_jit=False, use_my_model=True, use_ml_decoder=False, **kwargs):
    """ Constructs an extra-large TResnet model.
    """
    in_chans = 3
    # num_classes = model_params['num_classes']
    # remove_aa_jit = model_params['remove_aa_jit']
    layers = [3, 8, 34, 5] if use_ml_decoder else [4, 5, 24, 3]
    model = TResNet(layers=layers, num_classes=num_classes, in_chans=in_chans,
                    width_factor=1 if use_ml_decoder else 1.3,
                    remove_aa_jit=remove_aa_jit,
                    first_two_layers=Bottleneck if use_ml_decoder else BasicBlock,
                    use_my_model=use_my_model,
                    use_ml_decoder=use_ml_decoder, **kwargs)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls.get('tresnet_xl_mldecoder_640' if use_ml_decoder else 'tresnet_xl_448'),
            map_location='cpu'
        )['model']
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and not k.startswith('classifier') and not k.startswith('fc') and 'head' not in k:
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model
