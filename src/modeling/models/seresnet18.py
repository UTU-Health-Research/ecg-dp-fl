import torch.nn as nn
import torch
import random
import numpy as np

def set_seed(seed):
    print(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SELayer(nn.Module):
    ''' Squeeze-and-Excitation block'''
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    '''The convolutional block implementation for Resnet 
    as its architecture uses the CNN blocks multiple times
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride) # 3x3 padding
        self.bn1 = nn.Identity() # 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.Identity()
        self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''Class for implementing the ResNet architecture
    
    "The improved ResNet can be decomposed into....
     1) Feature extraction: 1 convolutional layers followed by a batch normalization layer, 
                            a ReLU activation function, a max pooling layer, 
                            N=8 residual blocks (= 2 convolutional layers, an SE block and an average pooling layer)
     2) Feature fusion: concatenates deep features from the previous part and 
                        the additional age and gender information
     3) Classifier: constitutes a fully connected layer and Sigmoid layer, 
                    outputs of the probabilities of belonging to a disease class" 
     (Zhao et al. 2022)
    '''

    def __init__(self, block, planes, layers, in_channel=1, out_channel=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0]) 
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(3, 10) # AGE AND GENDER LAYER - input size the same with the array size of attributes
        self.fc = nn.Linear(planes[3] * block.expansion + 10, out_channel)
        #self.sig = nn.Sigmoid() ! DON'T USE HERE CAUSE IMPLEMENTED IN THE TRAINING LOOP IN ./utils/train_utils_clip_ag.py

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.Identity):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.Identity(),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, ag):
        x = self.conv1(x) # Input layer, convolution operation
        x = self.bn1(x) # Applies Batch Normalization over a 2D or 3D input
        x = self.relu(x) # Applies the rectified linear unit function element-wise
        x = self.maxpool(x) # Applies 1D max pooling over an input signal composed of several input planes

        x = self.layer1(x) # 2nd convolution layer -> output size 56x56 ()
        x = self.layer2(x) # 3rd convolution layer -> output size 28x28
        x = self.layer3(x) # 4th convolution layer -> output size 14x14
        x = self.layer4(x) # 5th convolution layer -> output size 7x7

        x = self.avgpool(x) # Applies a 1D adaptive average pooling over an input signal composed of several input planes
        x = x.view(x.size(0), -1)
        ag = self.fc1(ag)  # Fully connected layer, deep features augmented with age and gender features
        x = torch.cat((ag, x), dim=1)
        x = self.fc(x) # Fully connected layers, outputs

        return x

ARCH_PLANES = {
        "v1": [64, 128, 256, 512],
        "v2": [32, 64, 128, 256],
        "v3": [16, 32, 64, 128],
        "v4": [8, 16, 32, 64]
}

def resnet18(seed, arch="v4", **kwargs):
    """Constructing a ResNet-18 model.
    """
    set_seed(seed)
    model = ResNet(BasicBlock, ARCH_PLANES[arch], [2, 2, 2, 2], **kwargs)
    return model

# ECG and age/gender features before the classifier
class ExtractorAgeECG8(ResNet):
    def forward(self, x, ag):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)

        return x

# ECG features after 8 residual-SE blocks
class ExtractorECG8(ResNet):
    def forward(self, x, ag):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

# ECG features after 7 residual-SE blocks
class ExtractorECG7(ResNet):
    def forward(self, x, ag):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.layer4[0](x)

        return x

# ECG features after 6 residual-SE blocks
class ExtractorECG6(ResNet):
    def forward(self, x, ag):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x

class SingleLinear(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)

class AgeLinear(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc = nn.Linear(in_features + 10, n_classes)

    def forward(self, x, ag):
        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)
        return self.fc(x)

class ResidualPlusLinear(nn.Module):
    def __init__(self, n_classes, planes):
        super().__init__()
        self.rb = BasicBlock(planes, planes)   # stride=1, downsample=None
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(3, 10)
        self.fc = nn.Linear(planes + 10, n_classes)

    def forward(self, x, ag):
        x = self.rb(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)
        return self.fc(x)

class Residual2x(nn.Module):
    def __init__(self, n_classes, planes):
        super().__init__()
        inplanes = planes // 2 # prev residual block
        downsample = nn.Sequential(
            conv1x1(inplanes, planes, 2),
            nn.Identity(),
        )
        layers = []
        layers.append(BasicBlock(inplanes, planes, 2, downsample))
        layers.append(BasicBlock(planes, planes))

        self.layer4 = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(3, 10)
        self.fc = nn.Linear(planes + 10, n_classes)

    def forward(self, x, ag):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)
        return self.fc(x)

def finetuning_model(strategy, n_classes, arch="v4"):
    planes = ARCH_PLANES[arch][-1]
    if strategy == "classif":
        model = SingleLinear(planes + 10, n_classes)
    elif strategy == "ag_classif":
        model = AgeLinear(planes, n_classes)
    elif strategy in ["residual", "ag_residual"]:
        model = ResidualPlusLinear(n_classes, planes)
    elif strategy == "res2x":
        model = Residual2x(n_classes, planes)
    else:
        return None

    if strategy in ["residual", "res2x"]:
        # disable training of age/gender features
        for param in model.fc1.parameters():
            param.requires_grad = False
    return model

