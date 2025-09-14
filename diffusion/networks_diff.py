import torch
import torch.nn as nn
import os
from typing import Optional

from generative.networks.nets import (
    AutoencoderKL,
    PatchDiscriminator,
    DiffusionModelUNet,
    ControlNet
)

class Autoencoder2D(nn.Module):
    def __init__(self):
        super(Autoencoder2D, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [batch, 16, 64, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [batch, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 1, 128, 128]
            nn.Sigmoid()  # output을 [0,1] 사이로 만들기 위해 Sigmoid 사용
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNet2D_BN(nn.Module):
    def __init__(self, activ='tanh', p_drop=0):
        super(UNet2D_BN, self).__init__()

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
        self.dropout = nn.Dropout(p=p_drop)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv11 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(33, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv21 = nn.Conv2d(33, 64, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(97, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv31 = nn.Conv2d(97, 128, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(225, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv41 = nn.Conv2d(225, 256, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(481, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv51 = nn.Conv2d(481, 512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(993, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.up6 = nn.ConvTranspose2d(993, 256, kernel_size=2, stride=2, padding=0)
        self.conv61 = nn.Conv2d(737, 256, kernel_size=3, padding=1)
        self.conv62 = nn.Conv2d(993, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.up7 = nn.ConvTranspose2d(993, 128, kernel_size=2, stride=2, padding=0)
        self.conv71 = nn.Conv2d(353, 128, kernel_size=3, padding=1)
        self.conv72 = nn.Conv2d(481, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.up8 = nn.ConvTranspose2d(481, 64, kernel_size=2, stride=2, padding=0)
        self.conv81 = nn.Conv2d(161, 64, kernel_size=3, padding=1)
        self.conv82 = nn.Conv2d(225, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)

        self.up9 = nn.ConvTranspose2d(225, 32, kernel_size=2, stride=2, padding=0)
        self.conv91 = nn.Conv2d(65, 32, kernel_size=3, padding=1)
        self.conv92 = nn.Conv2d(97, 32, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(32)

        self.conv10 = nn.Conv2d(97, 1, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        conv11 = self.conv11(x)
        conv11 = self.activation(conv11)
        conc11 = torch.cat([x, conv11], dim=1)
        conv12 = self.conv12(conc11)
        conv12 = self.activation(conv12)
        conc12 = torch.cat([x, conv12], dim=1)
        pool1 = self.pool(conc12)
        # pool1 = self.bn12(pool1)

        conv21 = self.conv21(pool1)
        conv21 = self.activation(conv21)
        conc21 = torch.cat([pool1, conv21], dim=1)
        conv22 = self.conv22(conc21)
        conv22 = self.activation(conv22)
        conc22 = torch.cat([pool1, conv22], dim=1)
        pool2 = self.pool(conc22)

        conv31 = self.conv31(pool2)
        conv31 = self.activation(conv31)
        conc31 = torch.cat([pool2, conv31], dim=1)
        conv32 = self.conv32(conc31)
        conv32 = self.activation(conv32)
        conc32 = torch.cat([pool2, conv32], dim=1)
        pool3 = self.pool(conc32)

        conv41 = self.conv41(pool3)
        conv41 = self.activation(conv41)
        conc41 = torch.cat([pool3, conv41], dim=1)
        conv42 = self.conv42(conc41)
        conv42 = self.activation(conv42)
        conc42 = torch.cat([pool3, conv42], dim=1)
        pool4 = self.pool(conc42)

        conv51 = self.conv51(pool4)
        conv51 = self.activation(conv51)
        conc51 = torch.cat([pool4, conv51], dim=1)
        conv52 = self.conv52(conc51)
        conv52 = self.activation(conv52)
        conc52 = torch.cat([pool4, conv52], dim=1)

        up6 = torch.cat([self.up6(conc52), conc42], dim=1)
        conv61 = self.conv61(up6)  # 수정
        conv61 = self.activation(conv61)
        conc61 = torch.cat([up6, conv61], dim=1)
        conv62 = self.conv62(conc61)
        conv62 = self.activation(conv62)
        conc62 = torch.cat([up6, conv62], dim=1)

        up7 = torch.cat([self.up7(conc62), conc32], dim=1)
        conv71 = self.conv71(up7)  # 수정
        conv71 = self.activation(conv71)
        conc71 = torch.cat([up7, conv71], dim=1)
        conv72 = self.conv72(conc71)
        conv72 = self.activation(conv72)
        conc72 = torch.cat([up7, conv72], dim=1)

        up8 = torch.cat([self.up8(conc72), conc22], dim=1)
        conv81 = self.conv81(up8)  # 수정
        conv81 = self.activation(conv81)
        conc81 = torch.cat([up8, conv81], dim=1)
        conv82 = self.conv82(conc81)
        conv82 = self.activation(conv82)
        conc82 = torch.cat([up8, conv82], dim=1)

        up9 = torch.cat([self.up9(conc82), conc12], dim=1)
        conv91 = self.conv91(up9)  # 수정
        conv91 = self.activation(conv91)
        conc91 = torch.cat([up9, conv91], dim=1)
        conv92 = self.conv92(conc91)
        conv92 = self.activation(conv92)
        conc92 = torch.cat([up9, conv92], dim=1)

        conv10 = self.conv10(conc92)  # linear activation instead of sigmoid
        conv10 = self.tanh(conv10)

        return conv10

class UNet2D_BN_latent(nn.Module):
    def __init__(self, dtype='dem', p_drop=0):
        super(UNet2D_BN_latent, self).__init__()

        # if activ == 'none':
        #     self.activation = nn.Identity(inplace=True)
        # elif activ == 'relu':
        #     self.activation = nn.ReLU(inplace=True)
        # elif activ == 'tanh':
        #     self.activation = nn.Tanh()
        # elif activ == 'elu':
        #     self.activation = nn.ELU(inplace=True)
        # elif activ == 'lrelu':
        #     self.activation = nn.LeakyReLU(inplace=True)

        self.activation = nn.ReLU(inplace=True)  #nn.ReLU(inplace=True) #nn.Tanh()
        if dtype == 'dem':
            self.activation = nn.ReLU(inplace=True)
            self.last_activation = nn.Tanh()
        else:
            self.activation = nn.Tanh()
            self.last_activation = nn.ReLU(inplace=True)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=p_drop)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv11 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(33, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv21 = nn.Conv2d(33, 64, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(97, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv31 = nn.Conv2d(97, 128, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(225, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv41 = nn.Conv2d(225, 256, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(481, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # self.conv51 = nn.Conv2d(481, 512, kernel_size=3, padding=1)
        # self.conv52 = nn.Conv2d(993, 512, kernel_size=3, padding=1)
        # self.bn5 = nn.BatchNorm2d(512)

        self.conv5_mu = nn.Conv2d(481, 3, kernel_size=3, padding=1)
        self.conv5_logvar = nn.Conv2d(481, 3, kernel_size=3, padding=1)

        self.conv6_1d = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        self.up6 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv61 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv62 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.up7 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=0)
        self.conv71 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv72 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.up8 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0)
        self.conv81 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv82 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)

        self.up9 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, padding=0)
        self.conv91 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv92 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(32)

        self.conv10 = nn.Conv2d(96, 1, kernel_size=1)

    def encoder(self,x):
        x = self.bn(x)
        conv11 = self.conv11(x)
        conv11 = self.activation(conv11)
        conc11 = torch.cat([x, conv11], dim=1)
        conv12 = self.conv12(conc11)
        conv12 = self.activation(conv12)
        conc12 = torch.cat([x, conv12], dim=1)
        pool1 = self.pool(conc12)
        # pool1 = self.bn12(pool1)

        conv21 = self.conv21(pool1)
        conv21 = self.activation(conv21)
        conc21 = torch.cat([pool1, conv21], dim=1)
        conv22 = self.conv22(conc21)
        conv22 = self.activation(conv22)
        conc22 = torch.cat([pool1, conv22], dim=1)
        pool2 = self.pool(conc22)

        conv31 = self.conv31(pool2)
        conv31 = self.activation(conv31)
        conc31 = torch.cat([pool2, conv31], dim=1)
        conv32 = self.conv32(conc31)
        conv32 = self.activation(conv32)
        conc32 = torch.cat([pool2, conv32], dim=1)
        pool3 = self.pool(conc32)

        conv41 = self.conv41(pool3)  # (b, 256, 16, 16)
        conv41 = self.activation(conv41)
        conc41 = torch.cat([pool3, conv41], dim=1)
        conv42 = self.conv42(conc41)
        conv42 = self.activation(conv42)
        conc42 = torch.cat([pool3, conv42], dim=1)

        mu = self.conv5_mu(conc42)
        logvar = self.conv5_logvar(conc42)

        return mu, logvar

    def decoder(self, z):
        z = self.conv6_1d(z)
        up6 = self.up6(z)  # (b, 256, 16, 16)
        conv61 = self.conv61(up6)  # 수정
        conv61 = self.activation(conv61)
        conc61 = torch.cat([up6, conv61], dim=1)
        conv62 = self.conv62(conc61)
        conv62 = self.activation(conv62)
        conc62 = torch.cat([up6, conv62], dim=1)  # (16, 993, 16, 16)

        up7 = self.up7(conc62)
        conv71 = self.conv71(up7)  # 수정
        conv71 = self.activation(conv71)
        conc71 = torch.cat([up7, conv71], dim=1)
        conv72 = self.conv72(conc71)
        conv72 = self.activation(conv72)
        conc72 = torch.cat([up7, conv72], dim=1)

        up8 = self.up8(conc72)
        conv81 = self.conv81(up8)  # 수정
        conv81 = self.activation(conv81)
        conc81 = torch.cat([up8, conv81], dim=1)
        conv82 = self.conv82(conc81)
        conv82 = self.activation(conv82)
        conc82 = torch.cat([up8, conv82], dim=1)

        up9 = self.up9(conc82)
        conv91 = self.conv91(up9)  # 수정
        conv91 = self.activation(conv91)
        conc91 = torch.cat([up9, conv91], dim=1)
        conv92 = self.conv92(conc91)
        conv92 = self.activation(conv92)
        conc92 = torch.cat([up9, conv92], dim=1)

        conv10 = self.conv10(conc92)  # linear activation instead of sigmoid
        conv10 = self.last_activation(conv10)

        return conv10


    def sampling(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_logvar)
        z_vae = z_mu + eps * std
        return z_vae

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.encoder(x)
        z = self.sampling(z_mu, z_sigma)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        dec = self.decoder(z)
        return dec

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        image = self.decode(z)
        return image

    def forward(self, x):

        z_mu, z_sigma = self.encoder(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)

        return reconstruction, z_mu, z_sigma

class UNet2D_BN_vae_wmh(nn.Module):
    def __init__(self, activ='tanh', p_drop=0):
        super(UNet2D_BN_vae_wmh, self).__init__()

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
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p_drop)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv11 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(33, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv21 = nn.Conv2d(33, 64, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(97, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv31 = nn.Conv2d(97, 128, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(225, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv41 = nn.Conv2d(225, 256, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(481, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # self.conv51 = nn.Conv2d(481, 512, kernel_size=3, padding=1)
        # self.conv52 = nn.Conv2d(993, 512, kernel_size=3, padding=1)
        # self.bn5 = nn.BatchNorm2d(512)

        self.conv5_mu = nn.Conv2d(481, 3, kernel_size=3, padding=1)
        self.conv5_logvar = nn.Conv2d(481, 3, kernel_size=3, padding=1)

        self.up6 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.conv61 = nn.Conv2d(737, 256, kernel_size=3, padding=1)
        self.conv62 = nn.Conv2d(993, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.up7 = nn.ConvTranspose2d(993, 128, kernel_size=2, stride=2, padding=0)
        self.conv71 = nn.Conv2d(353, 128, kernel_size=3, padding=1)
        self.conv72 = nn.Conv2d(481, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.up8 = nn.ConvTranspose2d(481, 64, kernel_size=2, stride=2, padding=0)
        self.conv81 = nn.Conv2d(161, 64, kernel_size=3, padding=1)
        self.conv82 = nn.Conv2d(225, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)

        self.up9 = nn.ConvTranspose2d(225, 32, kernel_size=2, stride=2, padding=0)
        self.conv91 = nn.Conv2d(65, 32, kernel_size=3, padding=1)
        self.conv92 = nn.Conv2d(97, 32, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(32)

        self.conv10 = nn.Conv2d(97, 1, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        conv11 = self.conv11(x)
        conv11 = self.activation(conv11)
        conc11 = torch.cat([x, conv11], dim=1)
        conv12 = self.conv12(conc11)
        conv12 = self.activation(conv12)
        conc12 = torch.cat([x, conv12], dim=1)
        pool1 = self.pool(conc12)
        # pool1 = self.bn12(pool1)

        conv21 = self.conv21(pool1)
        conv21 = self.activation(conv21)
        conc21 = torch.cat([pool1, conv21], dim=1)
        conv22 = self.conv22(conc21)
        conv22 = self.activation(conv22)
        conc22 = torch.cat([pool1, conv22], dim=1)
        pool2 = self.pool(conc22)

        conv31 = self.conv31(pool2)
        conv31 = self.activation(conv31)
        conc31 = torch.cat([pool2, conv31], dim=1)
        conv32 = self.conv32(conc31)
        conv32 = self.activation(conv32)
        conc32 = torch.cat([pool2, conv32], dim=1)
        pool3 = self.pool(conc32)

        conv41 = self.conv41(pool3)  # (b, 256, 16, 16)
        conv41 = self.activation(conv41)
        conc41 = torch.cat([pool3, conv41], dim=1)
        conv42 = self.conv42(conc41)
        conv42 = self.activation(conv42)
        conc42 = torch.cat([pool3, conv42], dim=1)
        # pool4 = self.pool(conc42)  # (b, 481, 8, 8)

        # conv51 = self.conv51(pool4)  # (b, 512, 8, 8)
        # conv51 = self.activation(conv51)
        # conc51 = torch.cat([pool4, conv51], dim=1)
        # conv52 = self.conv52(conc51)
        # conv52 = self.activation(conv52)
        # conc52 = torch.cat([pool4, conv52], dim=1)  # (b, 993, 8, 8)

        mu = self.conv5_mu(conc42)
        logvar = self.conv5_logvar(conc42)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std

        up6 = torch.cat([self.up6(latent), conc42], dim=1)  # (b, 737, 16, 16)
        conv61 = self.conv61(up6)  # 수정
        conv61 = self.activation(conv61)
        conc61 = torch.cat([up6, conv61], dim=1)
        conv62 = self.conv62(conc61)
        conv62 = self.activation(conv62)
        conc62 = torch.cat([up6, conv62], dim=1)   # (16, 993, 16, 16)

        up7 = torch.cat([self.up7(conc62), conc32], dim=1)
        conv71 = self.conv71(up7)  # 수정
        conv71 = self.activation(conv71)
        conc71 = torch.cat([up7, conv71], dim=1)
        conv72 = self.conv72(conc71)
        conv72 = self.activation(conv72)
        conc72 = torch.cat([up7, conv72], dim=1)

        up8 = torch.cat([self.up8(conc72), conc22], dim=1)
        conv81 = self.conv81(up8)  # 수정
        conv81 = self.activation(conv81)
        conc81 = torch.cat([up8, conv81], dim=1)
        conv82 = self.conv82(conc81)
        conv82 = self.activation(conv82)
        conc82 = torch.cat([up8, conv82], dim=1)

        up9 = torch.cat([self.up9(conc82), conc12], dim=1)
        conv91 = self.conv91(up9)  # 수정
        conv91 = self.activation(conv91)
        conc91 = torch.cat([up9, conv91], dim=1)
        conv92 = self.conv92(conc91)
        conv92 = self.activation(conv92)
        conc92 = torch.cat([up9, conv92], dim=1)

        conv10 = self.conv10(conc92)  # linear activation instead of sigmoid
        conv10 = self.relu(conv10)

        return conv10, mu, logvar

class UNet2D_BN_vae_dem(nn.Module):
    def __init__(self, activ='tanh', p_drop=0):
        super(UNet2D_BN_vae_dem, self).__init__()

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
        self.dropout = nn.Dropout(p=p_drop)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv11 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(33, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv21 = nn.Conv2d(33, 64, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(97, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv31 = nn.Conv2d(97, 128, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(225, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv41 = nn.Conv2d(225, 256, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(481, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # self.conv51 = nn.Conv2d(481, 512, kernel_size=3, padding=1)
        # self.conv52 = nn.Conv2d(993, 512, kernel_size=3, padding=1)
        # self.bn5 = nn.BatchNorm2d(512)

        self.conv5_mu = nn.Conv2d(481, 3, kernel_size=3, padding=1)
        self.conv5_logvar = nn.Conv2d(481, 3, kernel_size=3, padding=1)

        self.up6 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.conv61 = nn.Conv2d(737, 256, kernel_size=3, padding=1)
        self.conv62 = nn.Conv2d(993, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.up7 = nn.ConvTranspose2d(993, 128, kernel_size=2, stride=2, padding=0)
        self.conv71 = nn.Conv2d(353, 128, kernel_size=3, padding=1)
        self.conv72 = nn.Conv2d(481, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.up8 = nn.ConvTranspose2d(481, 64, kernel_size=2, stride=2, padding=0)
        self.conv81 = nn.Conv2d(161, 64, kernel_size=3, padding=1)
        self.conv82 = nn.Conv2d(225, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)

        self.up9 = nn.ConvTranspose2d(225, 32, kernel_size=2, stride=2, padding=0)
        self.conv91 = nn.Conv2d(65, 32, kernel_size=3, padding=1)
        self.conv92 = nn.Conv2d(97, 32, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(32)

        self.conv10 = nn.Conv2d(97, 1, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        conv11 = self.conv11(x)
        conv11 = self.activation(conv11)
        conc11 = torch.cat([x, conv11], dim=1)
        conv12 = self.conv12(conc11)
        conv12 = self.activation(conv12)
        conc12 = torch.cat([x, conv12], dim=1)
        pool1 = self.pool(conc12)
        # pool1 = self.bn12(pool1)

        conv21 = self.conv21(pool1)
        conv21 = self.activation(conv21)
        conc21 = torch.cat([pool1, conv21], dim=1)
        conv22 = self.conv22(conc21)
        conv22 = self.activation(conv22)
        conc22 = torch.cat([pool1, conv22], dim=1)
        pool2 = self.pool(conc22)

        conv31 = self.conv31(pool2)
        conv31 = self.activation(conv31)
        conc31 = torch.cat([pool2, conv31], dim=1)
        conv32 = self.conv32(conc31)
        conv32 = self.activation(conv32)
        conc32 = torch.cat([pool2, conv32], dim=1)
        pool3 = self.pool(conc32)

        conv41 = self.conv41(pool3)  # (b, 256, 16, 16)
        conv41 = self.activation(conv41)
        conc41 = torch.cat([pool3, conv41], dim=1)
        conv42 = self.conv42(conc41)
        conv42 = self.activation(conv42)
        conc42 = torch.cat([pool3, conv42], dim=1)
        # pool4 = self.pool(conc42)  # (b, 481, 8, 8)

        # conv51 = self.conv51(pool4)  # (b, 512, 8, 8)
        # conv51 = self.activation(conv51)
        # conc51 = torch.cat([pool4, conv51], dim=1)
        # conv52 = self.conv52(conc51)
        # conv52 = self.activation(conv52)
        # conc52 = torch.cat([pool4, conv52], dim=1)  # (b, 993, 8, 8)

        mu = self.conv5_mu(conc42)
        logvar = self.conv5_logvar(conc42)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std

        up6 = torch.cat([self.up6(latent), conc42], dim=1)  # (b, 737, 16, 16)
        conv61 = self.conv61(up6)  # 수정
        conv61 = self.activation(conv61)
        conc61 = torch.cat([up6, conv61], dim=1)
        conv62 = self.conv62(conc61)
        conv62 = self.activation(conv62)
        conc62 = torch.cat([up6, conv62], dim=1)   # (16, 993, 16, 16)

        up7 = torch.cat([self.up7(conc62), conc32], dim=1)
        conv71 = self.conv71(up7)  # 수정
        conv71 = self.activation(conv71)
        conc71 = torch.cat([up7, conv71], dim=1)
        conv72 = self.conv72(conc71)
        conv72 = self.activation(conv72)
        conc72 = torch.cat([up7, conv72], dim=1)

        up8 = torch.cat([self.up8(conc72), conc22], dim=1)
        conv81 = self.conv81(up8)  # 수정
        conv81 = self.activation(conv81)
        conc81 = torch.cat([up8, conv81], dim=1)
        conv82 = self.conv82(conc81)
        conv82 = self.activation(conv82)
        conc82 = torch.cat([up8, conv82], dim=1)

        up9 = torch.cat([self.up9(conc82), conc12], dim=1)
        conv91 = self.conv91(up9)  # 수정
        conv91 = self.activation(conv91)
        conc91 = torch.cat([up9, conv91], dim=1)
        conv92 = self.conv92(conc91)
        conv92 = self.activation(conv92)
        conc92 = torch.cat([up9, conv92], dim=1)

        conv10 = self.conv10(conc92)  # linear activation instead of sigmoid
        conv10 = self.tanh(conv10)

        return conv10, mu, logvar
def load_if(device, checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        network.load_state_dict(torch.load(checkpoints_path, map_location=device))
    return network.to(device)

def init_autoencoder(device, checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    autoencoder = AutoencoderKL(spatial_dims=2,
                                in_channels=1,
                                out_channels=1,
                                latent_channels=4,
                                num_channels=(64, 256, 512, 512),
                                num_res_blocks=4,
                                norm_num_groups=64,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False),
                                with_decoder_nonlocal_attn=False,
                                with_encoder_nonlocal_attn=False).to(device=device)
    return load_if(device, checkpoints_path, autoencoder)


def init_patch_discriminator(device, checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the parch discriminator
    """
    patch_discriminator = PatchDiscriminator(spatial_dims=2,
                                             num_layers_d=3,
                                             num_channels=32,
                                             in_channels=1,
                                             out_channels=1)
    return load_if(device, checkpoints_path, patch_discriminator)


def init_latent_diffusion(device, n_res, ch_opt, checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the UNet from the diffusion model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """
    if ch_opt==0:
        norm_num_groups = 256
        num_channels = (256, 512, 768)
        num_head_channels = (0, 512, 768)
    elif ch_opt==1:
        norm_num_groups=64
        num_channels = (64,128,256)
        num_head_channels = (0,128,256)
    elif ch_opt==2:
        norm_num_groups = 32
        num_channels = (64,96,128)
        num_head_channels = (0, 96, 128)
    elif ch_opt==3:
        norm_num_groups = 32
        num_channels = (32,64,128)
        num_head_channels = (0, 64, 128)
    elif ch_opt==4:
        norm_num_groups = 16
        num_channels = (16, 32, 64)
        num_head_channels = (0, 32, 64)
    elif ch_opt==5:
        norm_num_groups = 8
        num_channels = (8, 16, 32)
        num_head_channels = (0, 16, 32)
    elif ch_opt==6:
        norm_num_groups = 4
        num_channels = (4, 8, 16)
        num_head_channels = (0, 8, 16)


    latent_diffusion = DiffusionModelUNet(spatial_dims=2,
                                          in_channels=3,
                                          out_channels=3,
                                          num_res_blocks=n_res,   # 2
                                          num_channels=num_channels,  # (256, 512, 768) ch1(64,128,256), ch2(64,96,128), ch3(16,32,64) / (16, 32, 64)
                                          attention_levels=(False, True, True),  # False, True, True
                                          norm_num_groups=norm_num_groups,
                                          norm_eps=1e-6,
                                          resblock_updown=True,
                                          num_head_channels=num_head_channels,  # (0, 512, 768)
                                          transformer_num_layers=1,
                                          with_conditioning=True, # (with_conditioning=False, cross_attention_dim=None)
                                          cross_attention_dim=10,  # None,  # 8 (org), 9 (jinsu) 10
                                          num_class_embeds=None,
                                          upcast_attention=True,
                                          use_flash_attention=False)
    return load_if(device, checkpoints_path, latent_diffusion)

def init_controlnet(device, n_res, ch_opt, checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """

    if ch_opt == 0:
        norm_num_groups = 256
        num_channels = (256, 512, 768)
        num_head_channels = (0, 512, 768)
    elif ch_opt==1:
        norm_num_groups=64
        num_channels = (64,128,256)
        num_head_channels = (0,128,256)
    elif ch_opt==2:
        norm_num_groups = 32
        num_channels = (64,96,128)
        num_head_channels = (0, 96, 128)
    elif ch_opt==3:
        norm_num_groups = 32
        num_channels = (32,64,128)
        num_head_channels = (0, 64, 128)
    elif ch_opt==4:
        norm_num_groups = 16
        num_channels = (16, 32, 64)
        num_head_channels = (0, 32, 64)
    elif ch_opt==5:
        norm_num_groups = 8
        num_channels = (8, 16, 32)
        num_head_channels = (0, 16, 32)
    elif ch_opt==6:
        norm_num_groups = 4
        num_channels = (4, 8, 16)
        num_head_channels = (0, 8, 16)

    controlnet = ControlNet(spatial_dims=2,
                            in_channels=3,
                            num_res_blocks=n_res,
                            num_channels=num_channels,   # (256, 512, 768)
                            attention_levels=(False, True, True),
                            norm_num_groups=norm_num_groups,
                            norm_eps=1e-6,
                            resblock_updown=True,
                            num_head_channels=num_head_channels,  # (0, 512, 768)
                            transformer_num_layers=1,
                            with_conditioning=True,  # True, (with_conditioning=False, cross_attention_dim=None)
                            cross_attention_dim=10,  # 8 (org), 9 (jinsu) 10
                            num_class_embeds=None,
                            upcast_attention=True,
                            use_flash_attention=False,
                            conditioning_embedding_in_channels=6,
                            conditioning_embedding_num_channels=(256,))
    return load_if(device, checkpoints_path, controlnet)


def init_diffusion_thesis(device, n_res, checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the UNet from the diffusion model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """
    latent_diffusion = DiffusionModelUNet(spatial_dims=2,
                                          in_channels=1,
                                          out_channels=1,
                                          num_res_blocks=n_res,  # 2
                                          num_channels=(256, 512, 768),
                                          attention_levels=(False, True, True),  # False, True, True
                                          norm_num_groups=256,
                                          norm_eps=1e-6,
                                          resblock_updown=True,
                                          num_head_channels=(0, 512, 768),  # (0, 512, 768)
                                          transformer_num_layers=1,
                                          with_conditioning=True,  # (with_conditioning=False, cross_attention_dim=None)
                                          cross_attention_dim=10,  # None,  # 8 (org), 9 (jinsu) 10
                                          num_class_embeds=None,
                                          upcast_attention=True,
                                          use_flash_attention=False)
    return load_if(device, checkpoints_path, latent_diffusion)

def init_controlnet_thesis(device, checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """

    controlnet = ControlNet(spatial_dims=2,
                            in_channels=1,
                            num_res_blocks=2,
                            num_channels=(256, 512, 768),   # (256, 512, 768)
                            attention_levels=(False, True, True),
                            norm_num_groups=256,
                            norm_eps=1e-6,
                            resblock_updown=True,
                            num_head_channels=(0, 512, 768),  # (0, 512, 768)
                            transformer_num_layers=1,
                            with_conditioning=True,  # True, (with_conditioning=False, cross_attention_dim=None)
                            cross_attention_dim=10,  # 8 (org), 9 (jinsu) 10
                            num_class_embeds=None,
                            upcast_attention=True,
                            use_flash_attention=False,
                            conditioning_embedding_in_channels=2,
                            conditioning_embedding_num_channels=(256,))
    return load_if(device, checkpoints_path, controlnet)