import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False, return_features=False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        #self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.fc = NormedLinear(64, num_classes)
        else:
            self.fc = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        self.return_encoding = return_features

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        #out = self.layer3(out)
        #out = F.avg_pool2d(out, out.size()[3])
        #encoding = out.view(out.size(0), -1)
        #out = self.fc(encoding)

        return out



def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, use_norm=False, return_features=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm, return_features=return_features)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))

model_dict = {
    'resnet32': [resnet32, 64],

}

class BCLModel(nn.Module):
    def __init__(self ,num_classes=100, name='resnet32', head='mlp', use_norm=True, feat_dim=100,norm_layer = None):
        super(BCLModel, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        #self.inplanes = 32
        self.groups =int(1)
        #self.base_width = int(64)

        model_fun, dim_in = model_dict[name]
        medium_dim = 128
        self.encoder = model_fun()
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)
        self.indivdual = nn.ModuleList([self._make_layer(BasicBlock, 64, 5, stride=2) for _ in range(3)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.encoder = model_fun()
        #self.batch_size = 32
        self.DownSample = downsample(dim_in)
        if head == 'mlp':
            self.head = nn.Sequential(nn.Linear(dim_in, medium_dim), nn.BatchNorm1d(medium_dim), nn.ReLU(inplace=True),
                                      nn.Linear(medium_dim, feat_dim))
        else:
            raise NotImplementedError(
                'head not supported'
            )
        if use_norm:
            self.fc = NormedLinear(dim_in, num_classes)
        else:
            self.fc = nn.Linear(dim_in, num_classes)
        self.head_fc = nn.Sequential(nn.Linear(dim_in, medium_dim), nn.BatchNorm1d(medium_dim), nn.ReLU(inplace=True),
                                   nn.Linear(medium_dim, feat_dim))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        inplanes=32
        for stride in strides:
            layers.append(block(inplanes, planes, stride))
            inplanes = planes

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind):
        # x = (self.stage2[ind])(x)
        # x = (self.stage3[ind])(x)
        x = (self.indivdual[ind])(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        batch_size1 = x.shape[0]
        batch_size = int(batch_size1/3)
        f1, f2, f3= torch.split(x, [batch_size, batch_size, batch_size], dim=0)
        X = []
        X.append(f1)
        X.append(f2)
        X.append(f3)
        outs = []
        # self.feat=[]
        for ind in range(3):
            outs.append(self._separate_part(X[ind], ind))
        x1 = self.DownSample(torch.cat([outs[0], outs[1], outs[2]], dim=1))
        #x1 = torch.cat([f1, f2, f3], dim=1)
        #x1 = self.DownSample(x1)
        #x1 = self.avgpool(x1)
        #feat1 = torch.flatten(x1, 1)
        x1 = F.avg_pool2d(x1, x1.size()[3])
        feat1 = x1.view(x1.size(0), -1)

        x = self.layer3(x)
        #x = self.avgpool(x)
        #feat = torch.flatten(x, 1)
        x = F.avg_pool2d(x, x.size()[3])
        feat = x.view(x.size(0), -1)

        feat_mlp = F.normalize(self.head(feat), dim=1)
        logits = self.fc(feat1)
        centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1)
        return feat_mlp, logits, centers_logits

    def resnet32(num_classes=100, use_norm=False, return_features=False):
        return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm,
                        return_features=return_features)

class downsample(nn.Module):
    def __init__(self, in_dim):
        super(downsample, self).__init__()
        # self.down=InvertedResidual(2 * in_dim, in_dim, 1)
        self.down = nn.Sequential(
            nn.Conv2d(3 * in_dim, in_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x
