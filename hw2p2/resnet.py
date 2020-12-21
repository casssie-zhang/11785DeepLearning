import torch.nn as nn
import torch
import torch.nn.functional as F

"""
creating a baseline model from scratch
"""


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super(BasicBlock, self).__init__()


        self.input_channel = input_channel
        self.output_channel = output_channel

        self.conv1 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)

        # if input_channel != output_channel,
        if input_channel == output_channel:
            shortcut_stride = 1
        else:
            shortcut_stride = 2
        self.shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=shortcut_stride, bias=False)

    def forward(self, x):
        # change channel
        if self.input_channel != self.output_channel:
            # conv with stride 2
            out = F.relu(self.bn1(self.conv2(x))) # conv + bn + relu
        else:
            out = F.relu(self.bn1(self.conv1(x))) # conv + bn + relu

        out = self.bn1(self.conv1(out))
        out += self.shortcut(x) #shortcut summation
        out = F.relu(out)
        return out


class MyResNet34(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(MyResNet34, self).__init__()

        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        # print("hidden sizes:", self.hidden_sizes)

        layers = []
        #conv before res block
        layers.append(nn.Conv2d(in_channels=num_feats, out_channels= hidden_sizes[0],
                                     padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm2d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        # self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        for idx, channel_size in enumerate(hidden_sizes):
            if idx == 0:
                continue

            layers.append(BasicBlock(input_channel=self.hidden_sizes[idx],
                                          output_channel=self.hidden_sizes[idx+1]))

        self.layers = nn.Sequential(*layers)
        # classification layer
        self.linear_label = nn.Sequential(
            nn.Linear(self.hidden_sizes[-2], 1024, bias=False),
            nn.Linear(1024, self.hidden_sizes[-1], bias=False))

        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])  # hidden output


        label_output = self.linear_label(output)
        # label_output = label_output / torch.norm(self.linear_label.weight, dim=1)

        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output, output


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

# if __name__ == '__main__':
#     hidden_sizes = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
#
#     num_classes = 5
#     feat_dim = 10
#     input = torch.randn((10, 3, 64, 64))
#     import numpy as np
#     labels = np.random.randint(0, num_classes, 10)
#     network = MyResNet34(3, hidden_sizes, num_classes, feat_dim)
#     a,b,c = network.forward(input)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     from centerloss import CenterLoss
#     criterion_closs = CenterLoss(num_classes, feat_dim, device)
#
#     print(a.shape,b.shape, c.shape)
#     from torch import tensor
#     print(criterion_closs(tensor(a), tensor(labels)).item())

