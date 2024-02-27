import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional

# ref : https://github.com/pytorch/vision/blob/1fda0e850a2ff5dca7f10a2e12286c87645d6760/torchvision/models/resnet.py#L241

def conv3x3(in_planes: int, out_planes:int, stride:int, groups: int = 1, dilation: int = 1)->nn.Conv2d:
    """3x3 conv w/ padding"""
    return nn.Conv2d(
        in_channels = in_planes,
        out_channels=out_planes,
        kernel_size=3, ##
        stride=stride,
        groups=groups, # input과 output의 connection 제어 (default = 1)
        dilation=dilation,
        bias=False # the biases are omitted for simplifying notations (paper 3page)
    )

def conv1x1(in_planes: int, out_planes:int, stride:int)->nn.Conv2d:
    """ 1x1 conv """
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1, ##
        stride=stride,
        bias =False
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes:int ,planes:int, stride:int, groups: int = 1, dilation: int = 1,
                 base_width: int = 64, # depth
                 downsample: Optional[nn.Module]=None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64") # base block은 모두 64 width사용 -> 이후 128, 256, 512, ...
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock") # https://m.blog.naver.com/hatuheart/222278289776 
        
        # Filters
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) # can optionally do the operation in-place
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample # downsampling 없으면 pooling/stride할때 size변화로 error
        self.stride = stride 

    def forward(self, x: Tensor)->Tensor:
        identity = x # 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is None: ### 
            identity = self.downsample(x)

        out = out + x # F(x) + x
        out = self.relu(out)

        return out
"""
Argument "groups"
> ex. groups = 2
  : half the input channels => half the output channels
  (both subsequently concatenated.)(Useful for parallel processing.)
> ex. # of input channels = 4 
group 1:    [ .  .  .  ]
group 2:    [ .  |  .  ]
> Cin = Cin, Cout = Cin x K (groups = Cin) depthwise convolution
  Cin = Cin, Cout = K (groups = 1)


Argument "dilation"
> controls the spacing between the kernel points
e.g. Kernel: 3x 3 (dilation rate = 2)
1 0 1 0 1 0 0   0 1 0 1 0 1 0
0 0 0 0 0 0 0   0 0 0 0 0 0 0   
1 0 1 0 1 0 0   0 1 0 1 0 1 0
0 0 0 0 0 0 0   0 0 0 0 0 0 0
1 0 1 0 1 0 0   0 1 0 1 0 1 0
0 0 0 0 0 0 0   0 0 0 0 0 0 0
0 0 0 0 0 0 0   0 0 0 0 0 0 0
"""
class Bottleneck(nn.Module):
    expansion = 4
    """
    > For ResNet-50/101/152
    -> downsample : conv3 1, conv4 1, and conv5 1 w/ stride of 2
    https://yhkim4504.tistory.com/3 
    """
    def __init__(self, 
                 in_planes:int ,
                 planes:int, 
                 stride:int, 
                 groups: int = 1, #  the connections between inputs and outputs
                 dilation: int = 1,
                base_width:int = 64, 
                downsample: Optional[nn.Module]=None,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                #expansion: int = 4 
                 )->None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Filters
        # [ 1x1, 64
        # 3x3, 64
        # 1x1, 256 ]-> x3
            
        # groups impact
        width = int(planes * (base_width / 64.0)) * groups #####

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(in_planes, width) # output channel = 64
        self.bn1 = norm_layer(width) # output channel = 64
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # output channel = 64
        self.bn2 = norm_layer(width) # output channel = 64
        self.conv3 = conv1x1(width, planes * self.expansion) # output channel = 64 * (4) = 256
        self.bn3 = norm_layer(planes * self.expansion)

    def forward(self, x:Tensor)->Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # 1x1, 64

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out) # 3x3, 64

        out = self.conv3(out)
        out = self.bn3(out) # 1x1, 256

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out) # 256-d

        return out