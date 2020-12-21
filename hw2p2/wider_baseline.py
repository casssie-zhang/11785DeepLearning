import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, Dropout, MaxPool2d
import torch.nn.functional as F

#wider and deeper baseline model

class baselineModel(torch.nn.Module):
    def __init__(self, input_channel=3, output_size=4000, feat_dim = 1024):
        super().__init__()

        self.layer1 = nn.Sequential(
            Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            # MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer2 = nn.Sequential(
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.layer3 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )

        self.layer4 = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),

        )
        self.layer5 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),

        )

        self.layer6 = nn.Sequential(
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )

        self.layer7 = nn.Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),

        )

        self.layer8 = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )

        self.layer9 = nn.Sequential(
            Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(1024),
            ReLU(),
        )

        self.max_pool = MaxPool2d(2,2)

        self.linear_label = nn.Sequential(
            nn.Linear(1024, 1024),
            ReLU(),
            nn.Linear(1024, output_size)
        )

        self.linear_closs = nn.Linear(1024, feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)


    def forward(self, x, evalMode=False):
        out = self.layer1(x)
        for i in range(3):
            out = self.layer2(out) #64

        out = self.max_pool(out)
        out = self.layer3(out)

        for i in range(4):
            out = self.layer4(out)
        # print(out.shape)

        out = self.max_pool(out)
        out = self.layer5(out)

        for i in range(6):
            out = self.layer6(out)

        out = self.max_pool(out)
        out = self.layer7(out)

        for i in range(3):
            out = self.layer8(out)
        # print(out.shape)

        out = self.layer9(out)
        # print("layer9:", out.shape)
        out = self.max_pool(out)
        # print("final Maxpool:", out.shape)


        out = F.avg_pool2d(out, [out.size(2), out.size(3)], stride=1)
        out = out.reshape(out.shape[0], out.shape[1])
        # print("shape before linear:", out.shape)

        label_output = self.linear_label(out)
        # print("Label output shape:", label_output.shape)

        closs_output = self.linear_closs(out)
        closs_output = self.relu_closs(closs_output)

        
        return closs_output, label_output, out

# if __name__ == '__main__':
#     num_classes = 5
#     feat_dim = 10
#     input = torch.randn((10, 3, 64, 64))
#     network = baselineModel(3, num_classes)
#     network.forward(input)