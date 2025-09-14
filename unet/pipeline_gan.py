import torch
import math
from torch import nn
from upload.unet.networks import UNet2D_BN, CNN2D_Disc


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

# Calculate Dice coefficient loss
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


class DEMGAN(nn.Module):
    def __init__(self, activ, dim, cuda_idx):

        super(DEMGAN, self).__init__()
        self.dim = dim
        self.G_lr = 1e-5
        self.D_lr = 1e-4
        self.G_beta = 0.5
        self.min_lr = 1e-8
        self.G_lambda_gan = 0.4
        self.DISC_l1 = 0.3
        self.DISC_l2 = 0.3
        self.lambda_disc = 1
        self.lambda_mae_dem = 1
        self.lambda_mse_dem = 1
        self.lambda_mae_les = 1
        self.lambda_mse_les = 1
        self.lambda_dsc = 20
        self.begin_anneal = 50
        self.decay_rate = 0.9
        self.total_epoch = 150
        self.activ = activ
        self.device = torch.device(f'cuda:{cuda_idx}' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.G = UNet2D_BN(self.activ).to(self.device)
        self.D_dem = CNN2D_Disc().to(self.device)

        # Criterion
        self.criterion_L1 = nn.L1Loss(reduction='sum')
        self.criterion_L2 = nn.MSELoss(reduction='sum')
        self.criterion_bce = nn.BCELoss(reduction='sum')

        # Optimizer
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(self.G_beta, 0.999))
        self.optimizer_D_dem = torch.optim.Adam(self.D_dem.parameters(), lr=self.D_lr, betas=(self.G_beta, 0.999))
        self.optimizers_G = [self.optimizer_G]
        self.optimizers_D = [self.optimizer_D_dem]

    def gan_loss(self, pred, label):
        return self.criterion_L2(pred, label)

    def set_input(self, base_brain, foll_brain, base_les, foll_les, noise):
        self.real_base_brain = base_brain.to(self.device)
        self.real_foll_brain = foll_brain.to(self.device)
        self.real_base_les = base_les.to(self.device)
        self.real_foll_les = foll_les.to(self.device)
        self.noise = noise.to(self.device)

        self.mask_foll_les = torch.where(self.real_foll_les > 0, 1, 0)
        self.mask_dem = torch.where((self.real_base_les > 0)|(self.real_foll_les > 0), 1, 0).to(self.device)

        self.real_dem = (self.real_foll_les - self.real_base_les).to(self.device)
        # self.noise = 32 # noisesize=32

    def optimize_parameters(self):
        self.forward()
        self.update_D()
        self.update_G()

    def optimize_D_only(self):
        self.forward()
        self.update_D()

    def optimize_G_only(self):
        self.forward()
        self.set_requires_grad([self.G], True)
        self.loss_D = self.compute_loss_D(self.D_dem, self.real_dem, self.fake_dem, 'disc')
        self.update_G()

    def without_optimize(self):
        self.forward()
        self.loss_D = self.compute_loss_D(self.D_dem, self.real_dem, self.fake_dem, 'disc')
        self.compute_loss_G()

    def forward(self):
        self.fake_dem = self.G(self.real_base_les, self.noise)
        self.fake_foll_les = self.real_base_les + self.fake_dem
        self.fake_foll_brain = self.real_base_brain + self.fake_dem

    def update_G(self):
        self.set_requires_grad(self.D_dem, False)
        self.optimizer_G.zero_grad()
        self.compute_loss_G()
        self.loss_G.backward(retain_graph=True)
        self.optimizer_G.step()

    def compute_loss_G(self):

        # discrimianator
        self.gan_loss_dem = self.compute_loss_D(self.D_dem, self.real_dem, self.fake_dem, 'gen')

        # regularization term
        self.loss_mae_les = self.criterion_L1(self.fake_foll_les, self.real_foll_les) * self.lambda_mae_les
        self.loss_mse_les = self.criterion_L2(self.fake_foll_les, self.real_foll_les) * self.lambda_mse_les
        self.loss_dsc_les = dice_coef_loss(self.fake_foll_les, self.real_foll_les) * self.lambda_dsc


        self.fake_foll_les_vol = torch.sum(self.fake_foll_les, dim=(2, 3)).to(float) / 100  # fake_les_vol
        self.real_foll_les_vol = torch.sum(self.real_foll_les, dim=(2, 3)).to(float) / 100  # real_les_vol

        self.loss_mse_vol = self.criterion_L2(self.fake_foll_les_vol, self.real_foll_les_vol) * self.lambda_mse_les

        self.reg_term = self.loss_mae_les + self.loss_dsc_les + self.loss_mse_vol
        self.loss_G = self.gan_loss_dem + self.reg_term


    def update_D(self):

        self.set_requires_grad(self.D_dem, True)

        self.optimizer_D_dem.zero_grad()
        self.loss_D_dem = self.compute_loss_D(self.D_dem, self.real_dem, self.fake_dem, 'disc')
        self.loss_D_dem.backward(retain_graph=True)
        self.optimizer_D_dem.step()


    def compute_loss_D(self, D, real, fake, model):

        pred_real = D(real)
        label_real = torch.ones(real.size(0), 1).to(self.device)  # MSE

        pred_fake = D(fake)
        label_fake = torch.zeros(fake.size(0), 1).to(self.device)

        if model == 'disc':
            loss_D_real = self.gan_loss(pred_real, label_real).requires_grad_()
            loss_D_fake = self.gan_loss(pred_fake, label_fake)
            return (loss_D_real + loss_D_fake)/2 * self.lambda_disc

        elif model == 'gen':
            return self.gan_loss(pred_fake, label_real) * self.lambda_disc

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def adjust_learning_rate(self, optimizer, epoch):

        if self.begin_anneal == 0 or self.begin_anneal == self.total_epoch:
            self.learning_rate = self.BG_lr * 1.0
        elif epoch > self.begin_anneal:
            self.learning_rate = max(self.min_lr, self.G_lr * math.exp(-self.decay_rate * (epoch - self.begin_anneal)))
        else:
            self.learning_rate = self.G_lr * 1.0

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate


    def optimize_G_only_wo_D(self):
        self.fake_dem = self.G(self.real_base_les, self.noise)
        self.fake_foll_les = self.real_base_les + self.fake_dem
        self.set_requires_grad([self.G], True)
        self.optimizer_G.zero_grad()

        self.loss_mae_les = self.criterion_L1(self.fake_foll_les, self.real_foll_les) * self.lambda_mae_les
        self.loss_mse_les = self.criterion_L2(self.fake_foll_les, self.real_foll_les) * self.lambda_mse_les

        self.loss_mae_dem = self.criterion_L1(self.fake_dem, self.real_dem) * self.lambda_mae_les
        self.loss_mse_dem = self.criterion_L2(self.fake_dem, self.real_dem) * self.lambda_mse_les

        self.loss_dsc_les = dice_coef_loss(self.fake_foll_les, self.real_foll_les) * self.lambda_dsc

        self.fake_foll_les_vol = torch.sum(self.fake_foll_les * self.mask_foll_les, dim=(2, 3, 4)).to(float) / 1000  # fake_les_vol
        self.real_foll_les_vol = torch.sum(self.real_foll_les * self.mask_foll_les, dim=(2, 3, 4)).to(float) / 1000  # real_les_vol
        self.loss_mse_vol = self.criterion_L2(self.fake_foll_les_vol, self.real_foll_les_vol) * self.lambda_mse_les

        self.reg_term = self.loss_mae_les + self.loss_dsc_les + self.loss_mse_vol
        self.loss_G =  self.reg_term

        self.optimizer_G.zero_grad()
        self.loss_G.backward(retain_graph=True)
        self.optimizer_G.step()

    def optimize_G_only_w_D(self):
        self.fake_dem = self.G(self.real_base_les, self.noise)
        self.fake_foll_les = self.real_base_les + self.fake_dem
        self.set_requires_grad([self.G], True)

        self.set_requires_grad([self.D_dem], True)

        # from discriminator
        self.optimizer_D_dem.zero_grad()
        self.loss_D_dem = self.compute_loss_D(self.D_dem, self.real_dem, self.fake_dem, 'disc')
        self.loss_D_dem.backward(retain_graph=True)
        self.optimizer_D_dem.step()
        
        self.gan_loss_dem = self.compute_loss_D(self.D_dem, self.real_dem, self.fake_dem, 'gen')

        self.loss_mae_les = self.criterion_L1(self.fake_foll_les, self.real_foll_les) * self.lambda_mae_les
        self.loss_mse_les = self.criterion_L2(self.fake_foll_les, self.real_foll_les) * self.lambda_mse_les

        self.loss_mae_dem = self.criterion_L1(self.fake_dem, self.real_dem) * self.lambda_mae_les
        self.loss_mse_dem = self.criterion_L2(self.fake_dem, self.real_dem) * self.lambda_mse_les

        self.loss_dsc_les = dice_coef_loss(self.fake_foll_les, self.real_foll_les) * self.lambda_dsc

        self.fake_foll_les_vol = torch.sum(self.fake_foll_les * self.mask_foll_les, dim=(2, 3, 4)).to(float) / 1000  # fake_les_vol
        self.real_foll_les_vol = torch.sum(self.real_foll_les * self.mask_foll_les, dim=(2, 3, 4)).to(float) / 1000  # real_les_vol
        self.loss_mse_vol = self.criterion_L2(self.fake_foll_les_vol, self.real_foll_les_vol) * self.lambda_mse_les

        self.reg_term = self.loss_mae_les + self.loss_dsc_les + self.loss_mse_vol
        self.loss_G = self.gan_loss_dem + self.reg_term

        self.optimizer_G.zero_grad()
        self.loss_G.backward(retain_graph=True)
        self.optimizer_G.step()
