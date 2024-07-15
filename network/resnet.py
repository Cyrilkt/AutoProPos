# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : resnet18.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:25 PM 
'''

import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import conv3x3, conv1x1, BasicBlock, Bottleneck
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

settings = {
    'light_resnet18': [[2, 2, 0, 0], BasicBlock],
    'resnet18': [[2, 2, 2, 2], BasicBlock],
    'resnet34': [[3, 4, 6, 3], BasicBlock],
    'resnet50': [[3, 4, 6, 3], Bottleneck],
}

# v1 0.956 sur mnist 'light_resnet18': [[2, 2, 2, 0], BasicBlock],
#v2 0.961 sur mnist 'light_resnet18': [[2, 2, 0, 0], BasicBlock],
#v3 0.823 sur usps light_resnet18': [[2, 1, 0, 0], BasicBlock] 
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any,
) -> resnet.ResNet:
    model = resnet.ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    
    return model


def modify_conv_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=max(module.out_channels // 2, 1),  # Ensure at least 1 output channel
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode
            )
            # Replace the original module with the new module in its parent module
            parent_name, child_name = name.rsplit('.', 1)
            getattr(model, parent_name)._modules[child_name] = new_conv
    return model
    
class ResNet(object):

    def __init__(self,
                 net_name,
                 cifar=False,
                 preact=False,grayscale=False):
        self.net_name = net_name
        self.cifar = cifar
        self.preact = preact
        self.grayscale=grayscale

    def __call__(self, pretrained: bool = False, progress: bool = True, **kwargs):
        layers, block = settings[self.net_name]
        kwargs.update({
            'arch': self.net_name,
            'layers': layers,
            'block': block,
        })
        if self.preact:
            kwargs['block'] = PreActBasicBlock
        model = _resnet(pretrained=pretrained, progress=progress, **kwargs)
        nets = []
        for name, module in model.named_children():
            if self.cifar:
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if isinstance(module, nn.MaxPool2d):
                    continue
            elif self.grayscale:
                if name == 'conv1':
                    module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if isinstance(module, nn.MaxPool2d):
                    continue
                    
                #if isinstance(module, nn.Conv2d) and name!='conv1':
                #    module = nn.Conv2d(
                #        in_channels=module.in_channels,
                #        out_channels=max(module.out_channels // 2, 1),  # Ensure at least 1 output channel
                #        kernel_size=module.kernel_size,
                #        stride=module.stride,
                #        padding=module.padding,
                #        dilation=module.dilation,
                #        groups=module.groups,
                #        bias=(module.bias is not None),
                #        padding_mode=module.padding_mode
                #    )
            if isinstance(module, nn.Linear):
                nets.append(nn.Flatten(1))
                continue
            nets.append(module)
        
        model = nn.Sequential(*nets)
        #model=modify_conv_layers(model)
        return model


class PreActBasicBlock(BasicBlock):
    expansion = 1

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
    ) -> None:
        super(PreActBasicBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                               norm_layer)
        self.bn1 = norm_layer(inplanes)
        if self.downsample is not None:
            self.downsample = self.downsample[0]  # remove norm

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # print(x.size())
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out


if __name__ == '__main__':
    model = ResNet('resnet18',
                   cifar=True,
                   preact=True)
    model = model()
    print(model)
    import torch

    inputs = torch.randn(2, 3, 32, 32)
    print(model(inputs))
