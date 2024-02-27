import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import graphviz

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y),True)

def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
        else:
            blocks += [BasicBlock(c_out,c_out),]
    return nn.Sequential(*blocks)

class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False, square=False, embedding_size=256):
        super(Net,self).__init__()
        # 3 128 64 (people) 3 128 128 (vehicles)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 64 64 32 (people) 64 64 64 (vehicles)
        # self.layer1 = make_layers(64, 64, 1, False)
        self.layer2 = make_layers(64, 128, 1, True)
        # 128 32 16 (people) 128 32 32 (vehicles)
        self.layer3 = make_layers(128, 256, 1, True)
        # 256 16 8 (people) 256 16 16 (vehicles)
        if square:
            self.avgpool = nn.AvgPool2d((16, 16), 1)
        else:
            self.avgpool = nn.AvgPool2d((16, 8), 1)
        self.reid_linear = nn.Linear(256, embedding_size)
        self.reid_batchnorm = nn.BatchNorm1d(embedding_size)
        # 512 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(embedding_size, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        # x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.reid_linear(x)
        x = self.reid_batchnorm(x)
        # B x embedding_size
        if self.reid:
            # embedding
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            x = x.view(x.size(0), x.size(1), 1, 1)
            return x
        else:
            # classifier
            x = self.classifier(x)
            return x


if __name__ == '__main__':
    net = Net(num_classes=1261, reid=True, square=True, embedding_size=128)
    x = torch.randn(4,3,128,128)
    y = net(x)
    dot = make_dot(net(x), params=dict(net.named_parameters()))
    dot.format = 'png'
    output_path = "neural_network.png"
    dot.render(filename=output_path)
