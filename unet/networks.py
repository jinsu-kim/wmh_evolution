import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activ=nn.ReLU()):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_ch+out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = activ


    def forward(self, x):
        x = self.bn1(x)

        out = self.act(self.bn2(self.conv1(x)))
        out = torch.cat([x, out], dim=1)
        out = self.act(self.conv2(out))
        out = torch.cat([x, out], dim=1)

        return out


class UNet2D_BN(nn.Module):
    def __init__(self, activ='tanh'):
        super().__init__()

        if activ == 'none':
            self.activation = nn.Identity(inplace=True)
        elif activ == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activ == 'tanh':
            self.activation = nn.Tanh()
        elif activ == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activ == 'lrelu':
            self.activation = nn.LeakyReLU(inplace=True)

        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.convblk1 = ConvBlock(1, 32, self.activation)
        self.convblk2 = ConvBlock(33, 64, self.activation)
        self.convblk3 = ConvBlock(97, 128, self.activation)
        self.convblk4 = ConvBlock(225, 256, self.activation)

        self.convblk5 = ConvBlock(481, 512, self.activation)

        self.up6 = nn.ConvTranspose2d(993, 256, kernel_size=2, stride=2, padding=0)
        self.convblk6 = ConvBlock(737, 256, self.activation)
        self.up7 = nn.ConvTranspose2d(993, 128, kernel_size=2, stride=2, padding=0)
        self.convblk7 = ConvBlock(353, 128, self.activation)
        self.up8 = nn.ConvTranspose2d(481, 64, kernel_size=2, stride=2, padding=0)
        self.convblk8 = ConvBlock(161, 64, self.activation)
        self.up9 = nn.ConvTranspose2d(225, 32, kernel_size=2, stride=2, padding=0)
        self.convblk9 = ConvBlock(65, 32, self.activation)

        self.conv10 = nn.Conv2d(97, 1, kernel_size=1)

    def forward(self, x):

        cblk1 = self.pool(self.convblk1(x))
        cblk2 = self.pool(self.convblk2(cblk1))
        cblk3 = self.pool(self.convblk2(cblk2))
        cblk4 = self.pool(self.convblk2(cblk3))

        cblk5 = self.convblk5(cblk4)

        up6 = torch.cat([self.up6(cblk5), cblk4], dim=1)
        cblk6 = self.convblk6(up6)

        up7 = torch.cat([self.up7(cblk6), cblk3], dim=1)
        cblk7 = self.convblk7(up7)

        up8 = torch.cat([self.up8(cblk7), cblk2], dim=1)
        cblk8 = self.convblk6(up8)

        up9 = torch.cat([self.up9(cblk8), cblk1], dim=1)
        cblk9 = self.convblk6(up9)

        conv10 = self.conv10(cblk9)

        return self.tanh(conv10)


class CNN2D_Disc(nn.Module):
    def __init__(self):
        super(CNN2D_Disc, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.conv11 = nn.Conv2d(1, 16, kernel_size=5, padding=1)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=5, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.bn11 = nn.BatchNorm2d(16)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=5, padding=1)
        self.bn21 = nn.BatchNorm2d(32)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)

        self.conv51 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv61 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv71 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(12544,100)
        self.fc2 = nn.Linear(100, 2)


    def forward(self, x):
        x = self.bn11(self.activation(self.conv11(x)))
        x = self.pool2(self.activation(self.conv12(x)))

        x = self.bn21(self.activation(self.conv21(x)))
        x = self.pool2(self.activation(self.conv22(x)))

        x = self.bn31(self.activation(self.conv31(x)))
        x = self.pool2(self.activation(self.conv32(x)))

        x = self.bn41(self.activation(self.conv41(x)))
        x = self.pool2(self.activation(self.conv42(x)))

        x = self.activation(self.conv51(x))
        x = self.activation(self.conv61(x))
        x = self.activation(self.conv71(x))
        x = self.activation(self.fc1(x.reshape(x.size(0), -1)))
        x = self.sigmoid(self.fc2(x))

        return x