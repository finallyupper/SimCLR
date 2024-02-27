import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional

from typing import Tuple, Union, Type, List
from SimCLR.Resnet.blocks import conv3x3, conv1x1
from SimCLR.Resnet.blocks import BasicBlock, Bottleneck
# ref : https://github.com/pytorch/vision/blob/1fda0e850a2ff5dca7f10a2e12286c87645d6760/torchvision/models/resnet.py#L241
"""
[ nn.init ] https://pytorch.org/docs/stable/nn.init.html 
= initialize neural network parameters (torch.no_grad() mode)

1) torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)
= Fill the input Tensor with values using a 'Kaiming uniform distribution'
= U(-bound, bound) where bound=gain * sqrt(3 / fan_mode)

2) torch.nn.init.constant_(tensor, val)
= Fill the input Tensor with the value val.
"""
class ResNet(nn.Module):
    def __init__(self, 
                 #in_planes:int,planes:int,stride:int,
                 block: Type[Union[BasicBlock, Bottleneck]], # Union[X, Y] means X or Y => gets type X or Y
                 layers: List[int], # [3, 4, 6, 3]
                 num_classes: int = 1000, ##
                 dilation:int = 1,
                 groups:int = 1,
                 zero_init_residual: bool = False,
                 width_per_group: int = 64, # base_width
                 replace_stride_with_dilation: Optional[List[bool]] = None, # stride or dilation?
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 )->None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d 
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False] # use stride
        # else -> [True, False, False] can be an example (layer1:True, 2:False, 3:False)
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation length is not 3, \
                             got {replace_stride_with_dilation} instead.")
        
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = 64
        self.dilation = 1 
        self.groups = groups 
        # input image = 224x224  (3)  --->   feature map size = 112x112 (64)
        # padding = (kernel size - 1) / 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, biase=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, blocks= layers[0], planes = 64, stride = 1, dilate=replace_stride_with_dilation[0])
        self.layer2 = self.make_layer(block, blocks= layers[1], planes = 128, stride = 2, dilate=replace_stride_with_dilation[1])
        self.layer3 = self.make_layer(block, blocks= layers[2], planes = 256, stride = 2, dilate=replace_stride_with_dilation[2])
        self.layer4 = self.make_layer(block, blocks= layers[3], planes = 512, stride = 2, dilate=replace_stride_with_dilation[3])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes) # 512*4(=2048) --> 1000
        #self.softmax = nn.Softmax()

        # Tensor Initialization
        for m in self.modules():
            # 1. conv layer
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") #'fan_out' preserves the magnitudes in the backwards pass.
            # 2. bn layer 
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        # Branch Initialization (for better performance)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)
        """
        [...] ------------
          |bn1           |
        [...]            |
          |bn2           |
        [...]            |
          |bn3(0)<-------- + x (identity)
        """
    
    def make_layer(self, 
                   block:Type[Union[BasicBlock, Bottleneck]],
                   blocks:int, # = # of blocks (in paper, 3->4->6->3)
                   planes:int,
                   stride:int = 1,
                   dilate: bool=False)->nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None # init
        previous_dilation = self.dilation ##``

        if dilate: # replace stride w/ dilation
            self.dilation *= stride  
            stride = 1 
        
        # If downsample condition, update the variable 'downsample'
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            # inplanes != output channels case
            """
            identity from output of [ 3x3 conv, 64 ](256) -> in block [3x3 conv, 128]
            => Needs downsampling!
            """
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion)
            )

        layers = []
        # first layer
        layers.append(
            block(in_planes = self.inplanes, 
                  planes = planes, 
                  stride = stride,
                  groups = self.groups,
                  dilation = previous_dilation,
                  base_width = self.base_width,
                  downsample = downsample,
                  norm_layer = norm_layer
                  )
        )

        self.inplanes = planes * self.expansion # Update #input channels before making other layers

        # Remaining layers
        for _ in range(1, blocks):
            layers.append(
                block(in_planes = self.inplanes, 
                    planes = planes, 
                    #stride = stride, #####
                    groups = self.groups,
                    dilation = previous_dilation,
                    base_width = self.base_width,
                    #downsample = downsample,
                    norm_layer = norm_layer
                    ))
        
        return nn.Sequential(*layers)

    # define structure (Convenience of maintenance)
    def _forward(self, x:Tensor)->Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        # x = self.Softmax(x)
        return x
    # compute
    def forward(self, x:Tensor)->Tensor:
        return self._forward(x) 
    
def _resnet(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        weights: Optional[WeightsEnum], #api
        progress: bool,
        **kwargs: any)->ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    # define model
    model = ResNet(block, layers=layers, **kwargs)

    if weights is not None: ##
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash = True)) ### 
    
    return model 

