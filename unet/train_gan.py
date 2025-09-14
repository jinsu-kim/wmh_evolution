import os
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from datetime import datetime
from argparse import ArgumentParser
from pipeline_gan import DEMGAN

parser = ArgumentParser()
parser.add_argument("-c", "--cuda", default=1)
parser.add_argument("-t", "--treatment", default="aspirin", type=str)
parser.add_argument("-l", "--lr", default="0.00008")
parser.add_argument("-e", "--epochs", default=10)
parser.add_argument("-b","--batch", default=8)
parser.add_argument("-j", "--loss", default="mse")
parser.add_argument("-d", "--decay", default=1)
parser.add_argument("-opt", "--opt", default='adam')
parser.add_argument("-act", "--act", default='relu', type=str, help='activation function')
parser.add_argument("-proc", "--proc", default='white', type=str, help='data process type')
parser.add_argument("-fw", "--framework", default='dem', type=str, help='wmh or flair')
args = parser.parse_args()

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

project_name="GAN_dem"

treatment = args.treatment
batch_size = int(args.batch)
total_epoch = int(args.epochs)
cuda_idx = str(args.cuda)
lr = float(args.lr)
loss_type = args.loss
n_type = args.n_type
decay = float(args.decay)
opt = args.opt
activation = args.act
data_proc = args.proc
framework = args.fw
device = torch.device(f'cuda:{cuda_idx}' if torch.cuda.is_available() else 'cpu')

dt_now = datetime.now()

opponent = 'aspirin' if treatment == 'cilostazol' else 'cilostazol'
print('treatment: ', treatment)
print('loss_type: ', loss_type)

save_path = (f'./results/{dt_now.strftime("%Y%m%d_%H%m%s")}_{treatment}_gan2d_batch_{batch_size}_'
             f'epoch_{total_epoch}_activ_{activation}_proc_{data_proc}_pred-fw_{framework}')


class CustomLoader(Dataset):
    def __init__(self, treatment, data_proc,data_idcs):
        self.data_idcs = data_idcs
        self.treatment = treatment
        self.data_proc = data_proc


    def __len__(self):
        return len(self.data_idcs)

    def __getitem__(self, idx):
        base_brain_path = f'./dataset/{self.treatment}_{self.data_proc}_brain_slice/{self.data_idcs[idx]}'
        foll_brain_path = f'./dataset/{self.treatment}_{self.data_proc}_brain_slice/{self.data_idcs[idx].replace("base","foll")}'
        base_lesion_path = f'./dataset/{self.treatment}_{self.data_proc}_wmh_slice/{self.data_idcs[idx]}'
        foll_lesion_path = f'./dataset/{self.treatment}_{self.data_proc}_wmh_slice/{self.data_idcs[idx].replace("base","foll")}'

        base_brain = torch.unsqueeze(torch.from_numpy(np.load(base_brain_path).astype(np.float32)), 0)
        foll_brain = torch.unsqueeze(torch.from_numpy(np.load(foll_brain_path).astype(np.float32)), 0)
        base_lesion = torch.unsqueeze(torch.from_numpy(np.load(base_lesion_path).astype(np.float32)), 0)
        foll_lesion = torch.unsqueeze(torch.from_numpy(np.load(foll_lesion_path).astype(np.float32)), 0)

        return base_brain, foll_brain, base_lesion, foll_lesion


dataset_all = sorted(os.listdir(f'./dataset/{treatment}_{data_proc}'))
dataset_all = sorted(list(set([file for file in dataset_all if 'base' in file])))
subjects_all = sorted(list(set([file[:7] for file in dataset_all])))
fold_sz = len(subjects_all) // 10

for fold_id in range(10):

    list_lr = []
    min_loss = float('inf')
    best_epoch = 1

    model = DEMGAN(activ=activation, dim=2, cuda_idx=cuda_idx)

    dataset_all_copy = copy.deepcopy(dataset_all)
    subjects_copy = copy.deepcopy(subjects_all)

    test_idcs = set(subjects_copy[fold_sz * fold_id:fold_sz * (fold_id + 1)])
    test_list = sorted([file for file in dataset_all if file[:7] in test_idcs])
    train_list = sorted(set(dataset_all_copy) - set(test_list))

    train_dataset = CustomLoader(treatment=treatment, data_proc=data_proc, data_idcs=list(train_list))
    test_dataset = CustomLoader(treatment=treatment, data_proc=data_proc, data_idcs=list(test_list))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    total_step = len(train_loader)

    train_mse_list = []
    train_g_loss_list = []
    train_reg_loss_list = []
    train_dem_loss_list = []
    train_foll_loss_list = []
    train_gan_loss_dem_list = []
    train_gan_loss_foll_list = []

    valid_mse_list = []
    valid_g_loss_list = []
    valid_reg_loss_list = []
    valid_dem_loss_list = []
    valid_foll_loss_list = []
    valid_gan_loss_dem_list = []
    valid_gan_loss_foll_list = []

    for epoch in range(total_epoch):
        model.train()
        train_total_sample = 0
        train_mae_sum = 0
        train_g_loss_sum = 0
        train_reg_loss_sum = 0
        train_dem_loss_sum = 0
        train_foll_loss_sum = 0
        train_gan_loss_dem_sum = 0
        train_gan_loss_foll_sum = 0

        valid_total_sample = 0
        valid_mse_sum = 0
        valid_g_loss_sum = 0
        valid_reg_loss_sum = 0
        valid_dem_loss_sum = 0
        valid_foll_loss_sum = 0
        valid_gan_loss_dem_sum = 0
        valid_gan_loss_foll_sum = 0

        oppo_total_sample = 0
        oppo_mse_sum = 0
        oppo_g_loss_sum = 0
        oppo_reg_loss_sum = 0
        oppo_dem_loss_sum = 0
        oppo_foll_loss_sum = 0
        oppo_gan_loss_dem_sum = 0
        oppo_gan_loss_foll_sum = 0

        for i, (base_brain, foll_brain, base_les, foll_les) in enumerate(train_loader):

            torch.manual_seed(seed=42)
            noise = torch.randn((base_brain.size(0),32))
            model.set_input(base_brain, foll_brain, base_les, foll_les, noise)

            model.optimize_parameters()

            g_loss = model.loss_G
            g_reg_loss = model.reg_term
            gan_loss_dem = model.gan_loss_dem
            gan_loss_foll = model.gan_loss_les
            d_dem_loss = model.loss_D_dem
            d_foll_loss = model.loss_D_les
            mae_loss = model.loss_mae_les


            for optm in model.optimizers_G:
                model.adjust_learning_rate(optm, epoch)

            for optm in model.optimizers_D:
                model.adjust_learning_rate(optm, epoch)

            train_total_sample += base_brain.size(0)
            train_mae_sum += mae_loss.cpu().item()
            train_g_loss_sum += g_loss.cpu().item()
            train_reg_loss_sum += g_reg_loss.cpu().item()
            train_gan_loss_dem_sum += gan_loss_dem.cpu().item()
            train_gan_loss_foll_sum += gan_loss_foll.cpu().item()
            train_dem_loss_sum += d_dem_loss.cpu().item()
            train_foll_loss_sum += d_foll_loss.cpu().item()

            if (i + 1) % 10 == 0:
                print(f'Fold {fold_id+1}, Epoch [{epoch + 1}/{total_epoch}], Step [{i + 1}/{total_step}], lr: {model.learning_rate}, '
                      f'mae_Loss: {round(train_mae_sum / train_total_sample, 6)}')

        train_mse_list.append(train_mae_sum / train_total_sample)
        train_g_loss_list.append(train_g_loss_sum / train_total_sample)
        train_reg_loss_list.append(train_reg_loss_sum / train_total_sample)
        train_dem_loss_list.append(train_dem_loss_sum / train_total_sample)
        train_foll_loss_list.append(train_foll_loss_sum / train_total_sample)
        train_gan_loss_dem_list.append(train_gan_loss_dem_sum / train_total_sample)
        train_gan_loss_foll_list.append(train_gan_loss_foll_sum / train_total_sample)
        list_lr.append(model.learning_rate)

        # test
        model.eval()
        with torch.no_grad():

            for (base_brain, foll_brain, base_les, foll_les) in test_loader:
                torch.manual_seed(seed=42)
                noise = torch.randn((base_brain.size(0), 32))
                model.set_input(base_brain, foll_brain, base_les, foll_les, noise)

                model.without_optimize()

                g_loss = model.loss_G
                g_reg_loss = model.reg_term
                gan_loss_dem = model.gan_loss_dem
                gan_loss_foll = model.gan_loss_les
                d_dem_loss = model.loss_D_dem
                d_foll_loss = model.loss_D_les
                mse_loss = model.loss_mse_les

                valid_total_sample += base_brain.size(0)
                valid_mse_sum += mse_loss.cpu().item()
                valid_g_loss_sum += g_loss.cpu().item()
                valid_reg_loss_sum += g_reg_loss.cpu().item()
                valid_gan_loss_dem_sum += gan_loss_dem.cpu().item()
                valid_gan_loss_foll_sum += gan_loss_foll.cpu().item()
                valid_dem_loss_sum += d_dem_loss.cpu().item()
                valid_foll_loss_sum += d_foll_loss.cpu().item()

            valid_mse_list.append(valid_mse_sum / valid_total_sample)
            valid_g_loss_list.append(valid_g_loss_sum / valid_total_sample)
            valid_reg_loss_list.append(valid_reg_loss_sum / valid_total_sample)
            valid_dem_loss_list.append(valid_dem_loss_sum / valid_total_sample)
            valid_foll_loss_list.append(valid_foll_loss_sum / valid_total_sample)
            valid_gan_loss_dem_list.append(valid_gan_loss_dem_sum / valid_total_sample)
            valid_gan_loss_foll_list.append(valid_gan_loss_foll_sum / valid_total_sample)

            print(f'Fold {fold_id+1}, Kfold CV Loss: {round(valid_mse_sum / valid_total_sample, 6)}')


            if (valid_g_loss_sum / valid_total_sample) < min_loss:
                model_path = save_path.replace("results", "models")

                if os.path.isfile(f'{model_path}/fold{fold_id + 1}/model_best_{best_epoch + 1}.pt'):
                    os.remove(f'{model_path}/fold{fold_id + 1}/model_best_{best_epoch + 1}.pt')

                min_loss = valid_g_loss_sum / valid_total_sample
                best_epoch = epoch

                if not os.path.exists(f'{model_path}/fold{fold_id + 1}'):
                    os.makedirs(f'{model_path}/fold{fold_id + 1}')
                torch.save(model.state_dict(), f'{model_path}/fold{fold_id + 1}/model_best_{best_epoch + 1}.pt')

            if ((epoch + 1) % 200 == 0) & ((epoch + 1) != total_epoch):
                torch.save(model.state_dict(), f'{model_path}/fold{fold_id + 1}/model_{epoch + 1}.pt')


    os.makedirs(f'{save_path}/csv', exist_ok=True)
    torch.save(model.state_dict(), f'{model_path}/fold{fold_id+1}/model_last_{epoch+1}.pt')
