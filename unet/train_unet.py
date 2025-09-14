import os
import shutil
import random
import pandas as pd
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from networks import UNet2D_BN
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.nn import L1Loss

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

save_path = (f'./results/{dt_now.strftime("%Y%m%d_%H%m%s")}_{treatment}_unet2d_batch_{batch_size}_'
             f'epoch_{total_epoch}_activ_{activation}_proc_{data_proc}_pred-fw_{framework}')


def mean_absolute_percentage_error(y_true, y_pred):
    absolute_percentage_error = torch.abs((y_true - y_pred) / (y_true+1))
    mape = torch.mean(absolute_percentage_error) * 100 * batch_size
    return mape

def bbox_index(data):
    non_zero_indices = torch.nonzero(data)

    x_min = torch.min(non_zero_indices[:, 0])
    x_max = torch.max(non_zero_indices[:, 0])
    y_min = torch.min(non_zero_indices[:, 1])
    y_max = torch.max(non_zero_indices[:, 1])
    z_min = torch.min(non_zero_indices[:, 2])
    z_max = torch.max(non_zero_indices[:, 2])

    return x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()

def apply_random_mask(data, mask_size=(3, 3, 3), mask_ratio=0.1):  # data: tensor
    depth, height, width = data.shape

    depth_r, height_r, width_r = depth // mask_size[0], height // mask_size[1], width // mask_size[2]

    total_chunks = depth_r*height_r*width_r
    num_masked_chunks = int(total_chunks * mask_ratio)

    chunk_indices = list(range(total_chunks))

    random.shuffle(chunk_indices)
    masked_chunk_indices = chunk_indices[:num_masked_chunks]
    mask = torch.ones((depth, height, width), dtype=torch.float32)

    # Randomly select positions to apply mask
    for idx in masked_chunk_indices:

        x_order = idx % depth_r
        y_order = ( idx // depth_r ) % height_r
        z_order =  idx // (depth_r * height_r)

        mask[x_order*mask_size[0]:(x_order+1)*mask_size[0],
            y_order*mask_size[1]:(y_order+1)*mask_size[1],
            z_order*mask_size[2]:(z_order+1)*mask_size[2]] = 0

    # Apply mask to data
    masked_data = data * mask

    return masked_data


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



dataset_all = sorted(os.listdir(f'./dataset/{treatment}_{data_proc}_wmh_slice'))
dataset_all = sorted(list(set([file for file in dataset_all if 'base' in file])))
subjects_all = sorted(list(set([file[:7] for file in dataset_all])))
fold_sz = len(subjects_all) // 10


repeat = None
for fold_id in range(10):

    list_lr = []
    min_loss = float('inf')
    best_epoch = 1

    model = UNet2D_BN(activ=activation).to(device)
    criterion = nn.MSELoss(reduction='sum')
    l1_loss_fn = L1Loss(reduction='sum')
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: decay ** epoch,
                                            last_epoch=-1,
                                            verbose=False)

    dataset_all_copy = copy.deepcopy(dataset_all)
    subjects_copy = copy.deepcopy(subjects_all)

    test_idcs = set(subjects_copy[fold_sz * fold_id:fold_sz * (fold_id + 1)])
    test_list = sorted([file for file in dataset_all if file[:7] in test_idcs])
    train_list = sorted(set(dataset_all_copy) - set(test_list))

    train_dataset = CustomLoader(treatment=treatment, data_proc=data_proc, data_idcs=list(train_list))
    test_dataset = CustomLoader(treatment=treatment, data_proc=data_proc, data_idcs=list(test_list))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    total_step = len(train_loader)


    train_loss_list = []
    train_mape_list = []
    valid_loss_list = []
    valid_mape_list = []

    for epoch in range(total_epoch):
        model.train()
        train_total_sample = 0
        train_loss_sum = 0
        train_mape_sum = 0

        for i, (base_brain, foll_brain, base_les, foll_les) in enumerate(train_loader):

            base_brain = base_brain.to(device)
            foll_brain = foll_brain.to(device)
            base_les = base_les.to(device)
            foll_les = foll_les.to(device)

            mask_dem = torch.where((base_les > 0) | (foll_les > 0), 1, 0).to(device)

            base_les_mask = torch.where(base_les > 0, base_brain, torch.tensor(0)).to(device)
            foll_les_mask = torch.where(foll_les > 0, foll_brain, torch.tensor(0)).to(device)

            real_dem = (foll_les_mask - base_les_mask).to(device)

            if framework == 'flair':
                output = model(base_brain)
                loss_mse = l1_loss_fn(output, foll_brain)
            elif framework == 'wmh':
                output = model(base_les)
                loss_mse = l1_loss_fn(output, foll_les)
            elif framework == 'dem':
                output = model(base_les)
                loss_mse = l1_loss_fn(output, real_dem)

            # Backward and optimize
            optimizer.zero_grad()
            loss_mse.backward()
            optimizer.step()

            train_total_sample += real_dem.size(0)
            train_loss_sum += loss_mse.cpu().item()

            if (i + 1) % 10 == 0:
                print(f'Fold {fold_id+1}, Epoch [{epoch + 1}/{total_epoch}], Step [{i + 1}/{total_step}], lr: {optimizer.param_groups[0]["lr"]}, Loss: {round(train_loss_sum / train_total_sample, 6)}')

        train_loss_list.append(train_loss_sum / train_total_sample)
        train_mape_list.append(train_mape_sum / train_total_sample)
        list_lr.append(optimizer.param_groups[0]['lr'])

        if ((epoch+1) % 200 == 0) & ((epoch+1) != total_epoch):
            torch.save(model.state_dict(), f'{model_path}/fold{fold_id+1}/model_{epoch+1}.pt')

        if optimizer.param_groups[0]['lr'] > 1e-8:
            scheduler.step()

        # test
        model.eval()
        with torch.no_grad():
            valid_total_sample = 0
            valid_loss_sum = 0

            for i, (base_brain, foll_brain, base_les, foll_les) in enumerate(test_loader):
                base_brain = base_brain.to(device)
                foll_brain = foll_brain.to(device)
                base_les = base_les.to(device)
                foll_les = foll_les.to(device)

                mask_dem = torch.where((base_les > 0) | (foll_les > 0), 1, 0).to(device)

                base_les_mask = torch.where(base_les > 0, base_brain, torch.tensor(0)).to(device)
                foll_les_mask = torch.where(foll_les > 0, foll_brain, torch.tensor(0)).to(device)

                real_dem = (foll_les_mask - base_les_mask).to(device)

                if framework == 'brain':
                    output = model(base_brain)
                    loss_mse = l1_loss_fn(output, foll_brain)
                elif framework == 'wmh':
                    output = model(base_les)
                    loss_mse = l1_loss_fn(output, foll_les)
                elif framework == 'dem':
                    output = model(base_les)
                    loss_mse = l1_loss_fn(output, real_dem)

                valid_total_sample += real_dem.size(0)
                valid_loss_sum += loss_mse.cpu().item()

            valid_loss_list.append(valid_loss_sum / valid_total_sample)
            print(f'Fold {fold_id+1}, Kfold CV Loss: {round(valid_loss_sum / valid_total_sample, 6)}')

            if (valid_loss_sum / valid_total_sample) < min_loss:
                model_path = save_path.replace("results", "models")

                if os.path.isfile(f'{model_path}/fold{fold_id+1}/model_best_{best_epoch+1}.pt'):
                    os.remove(f'{model_path}/fold{fold_id+1}/model_best_{best_epoch+1}.pt')

                min_loss = valid_loss_sum / valid_total_sample
                best_epoch = epoch

                if not os.path.exists(f'{model_path}/fold{fold_id + 1}'):
                    os.makedirs(f'{model_path}/fold{fold_id+1}')
                torch.save(model.state_dict(), f'{model_path}/fold{fold_id+1}/model_best_{best_epoch+1}.pt')


    os.makedirs(f'{save_path}/csv', exist_ok=True)
    df_fold = pd.DataFrame({
        'train': train_loss_list,
        'valid': valid_loss_list,
    })
    df_fold.to_csv(f'{save_path}/csv/fold{fold_id+1}_unet_loss.csv')

    torch.save(model.state_dict(), f'{model_path}/fold{fold_id+1}/model_last_{epoch+1}.pt')






