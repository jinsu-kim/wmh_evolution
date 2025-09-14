import os
import argparse

import torch
import copy
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from tqdm import tqdm
from customloader import CustomLoader_diff, CustomLoader_diff_trans

import utils
import networks
from sampling import sample_using_diffusion
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

set_determinism(0)

def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    scale_factor,
    context
):
    """
    Visualize the generation on tensorboard
    """

    for tag_i, size in enumerate([ 'small', 'medium', 'large' ]):

        image = sample_using_diffusion(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            context=context[0,:,:],
            device=DEVICE,
            scale_factor=scale_factor
        )

        utils.tb_display_generation(
            writer=writer, 
            step=epoch, 
            tag=f'{mode}/{size}_ventricles',
            image=image
        )


def compute_mae(z_wmh, z_dem):

    mae_loss = F.l1_loss(z_wmh, z_dem, reduction='mean')
    return mae_loss


def compute_kld(p, q):

    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)

    kld_loss = F.kl_div(p.log(), q, reduction='batchmean')
    return kld_loss

def combined_loss(z_wmh, z_dem, lambda_mae=1.0, lambda_kld=1.0):

    mae_loss = compute_mae(z_wmh, z_dem)
    kld_loss = compute_kld(z_wmh, z_dem)
    total_loss = lambda_mae * mae_loss + lambda_kld * kld_loss

    return total_loss

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',      default='diff_test', type=str)
    parser.add_argument('--figure_dir',      default='diff_loss', type=str)
    parser.add_argument('--save_dir',        default='diff_latent_recon', type=str)
    parser.add_argument('--aekl_wmh_ckpt',   default='ae_test/save_autoencoder-vae-1.5-wmh-ep-50.pth', type=str)
    parser.add_argument('--aekl_dem_ckpt',   default='ae_test/save_autoencoder-vae-1.5-dem-ep-90.pth', type=str)
    parser.add_argument('--diff_ckpt',       default=None, type=str)
    parser.add_argument('--num_workers',     default=8,     type=int)
    parser.add_argument('--n_epochs',        default=100,     type=int)
    parser.add_argument('--batch_size',      default=8,    type=int)
    parser.add_argument('--lr',              default=5e-6,  type=float)
    parser.add_argument('--gpu_idx',         default=1, type=int)
    parser.add_argument('--prcs_idx',        default=0, type=int)
    parser.add_argument('--n_res',           default=2, type=int)
    parser.add_argument('--ch_opt',          default=3, type=int)

    args = parser.parse_args()

    DEVICE = f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv('./picasso_filtered.csv')[
                                               ['SUBJNO', 'AGE', 'BASE_LOAD', 'SC_Fazekas', 'PROBUCOL',
                                                'STATIN_INTENSITY', 'SBP', 'PULSE', 'HX_ICH', 'Current_SM']
                                              ]
    df_subj = df['SUBJNO'].tolist()
    df = df[df.columns[1:]]

    # load latents that DEM != 0
    treatment = 'aspirin'
    all_wmh_files = os.listdir(f'./dataset/{treatment}_wmh_slice_latent_trans_aug')
    all_files = sorted(list(set([file for file in all_wmh_files if 'base' in file and file[:7] in df_subj])))
    all_subjects = sorted(list(set([file[:7] for file in all_files])))

    fold_sz = len(all_subjects) // 10

    best_loss = 1000
    best_epoch = 0
    for fold_id in range(args.prcs_idx, args.prcs_idx+1):

        all_files_copy = copy.deepcopy(all_files)
        all_subjects_copy = copy.deepcopy(all_subjects)

        valid_idcs = set(all_subjects_copy[fold_sz * fold_id:fold_sz * (fold_id + 1)])
        valid_list = sorted([file for file in all_files_copy if file[:7] in valid_idcs])
        train_list = sorted(set(all_files_copy) - set(valid_list))

        # wmh, dem latents
        trainset = CustomLoader_diff_trans(treatment=treatment, data_idcs=train_list)
        validset = CustomLoader_diff_trans(treatment=treatment, data_idcs=valid_list)

        train_loader = DataLoader(dataset=trainset,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  persistent_workers=True,
                                  pin_memory=True)

        valid_loader = DataLoader(dataset=validset,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  persistent_workers=True,
                                  pin_memory=True)


        autoencoder_wmh = networks.UNet2D_BN_latent(dtype='wmh').to(DEVICE)
        autoencoder_wmh.load_state_dict(torch.load(args.aekl_wmh_ckpt))
        autoencoder_dem = networks.UNet2D_BN_latent(dtype='dem').to(DEVICE)
        autoencoder_dem.load_state_dict(torch.load(args.aekl_dem_ckpt))

        autoencoder_wmh.decoder = autoencoder_dem.decoder
        autoencoder = autoencoder_wmh

        diffusion = networks.init_latent_diffusion(device=DEVICE,
                                                   n_res=args.n_res,
                                                   ch_opt=args.ch_opt,
                                                   checkpoints_path=args.diff_ckpt).to(DEVICE)

        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            schedule='scaled_linear_beta',
            beta_start=0.0015,
            beta_end=0.0205
        )

        inferer = DiffusionInferer(scheduler=scheduler)
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
        scaler = GradScaler()

        with torch.no_grad():
            with autocast(enabled=True):
                z = trainset[0][0]

        scale_factor = 1 / torch.std(z)
        print(f"Scaling factor set to {scale_factor}")


        writer = SummaryWriter()
        global_counter  = { 'train': 0, 'valid': 0 }
        loaders         = { 'train': train_loader, 'valid': valid_loader }
        datasets        = { 'train': trainset, 'valid': validset }

        train_loss_list = []
        valid_loss_list = []

        for epoch in range(args.n_epochs):

            for mode in loaders.keys():

                loader = loaders[mode]
                diffusion.train() if mode == 'train' else diffusion.eval()
                epoch_loss = 0
                progress_bar = tqdm(enumerate(loader), total=len(loader))
                progress_bar.set_description(f"Epoch {epoch}")

                for step, batch in progress_bar:

                    # if step > 3:
                    #     break

                    with autocast(enabled=True):

                        if mode == 'train': optimizer.zero_grad(set_to_none=True)
                        wmh_latents = batch[0].to(DEVICE) * scale_factor
                        dem_latents = batch[1].to(DEVICE)
                        idcs = batch[2].numpy()  # 이걸 고민해야 contribution
                        slice_order_feat = batch[3]
                        context_npy = np.concatenate([df.loc[idcs].values.astype(np.float32),np.expand_dims(slice_order_feat,1)],axis=1)
                        context = torch.from_numpy(context_npy).type(torch.float32).unsqueeze(1).to(DEVICE)
                        context = torch.zeros_like(context).type(torch.float32)

                        n = wmh_latents.shape[0]

                        with torch.set_grad_enabled(mode == 'train'):

                            noise = torch.randn_like(wmh_latents).to(DEVICE)
                            timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()

                            noise_pred = inferer(
                                inputs=dem_latents,
                                diffusion_model=diffusion,
                                noise=noise,
                                timesteps=timesteps,
                                condition=context,   # context가 있을 때
                                mode='crossattn'
                            )  # latents, contexts = (batch,3,16,16)

                            loss = F.mse_loss( noise.float(), noise_pred.float() )
                            # loss = combined_loss( noise.float(), noise_pred.float() )

                    if mode == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                    global_counter[mode] += 1

                # end of epoch
                epoch_loss = epoch_loss / len(loader)
                writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

                # visualize results
                images_to_tensorboard(
                    writer=writer,
                    epoch=epoch,
                    mode=mode,
                    autoencoder=autoencoder,
                    diffusion=diffusion,
                    scale_factor=scale_factor,
                    context=context,
                )
                if mode == 'train':
                    train_loss_list.append(epoch_loss)
                if mode == 'valid':
                    valid_loss_list.append(epoch_loss)

            if epoch_loss < best_loss:
                if os.path.isfile(f'{args.output_dir}/unet_dem-res{args.n_res}-ch{args.ch_opt}-trans-ctxt-ep-{best_epoch+1}-fold-{fold_id+1}_best.pth'):
                    os.remove(f'{args.output_dir}/unet_dem-res{args.n_res}-ch{args.ch_opt}-trans-ctxt-ep-{best_epoch+1}-fold-{fold_id+1}_best.pth')
                best_loss = epoch_loss
                best_epoch = epoch
                os.makedirs(args.output_dir, exist_ok=True)
                savepath = os.path.join(args.output_dir, f'unet_dem-res{args.n_res}-ch{args.ch_opt}-trans-ctxt-ep-{epoch+1}-fold-{fold_id+1}_best.pth')
                torch.save(diffusion.state_dict(), savepath)

            # save the model
            if (epoch + 1) == args.n_epochs:
                os.makedirs(args.output_dir, exist_ok=True)
                savepath = os.path.join(args.output_dir, f'unet_dem-res{args.n_res}-ch{args.ch_opt}-trans-ctxt-ep-{epoch+1}-fold-{fold_id+1}.pth')
                torch.save(diffusion.state_dict(), savepath)

                os.makedirs(args.figure_dir, exist_ok=True)
                figure_path = os.path.join(args.figure_dir, f'loss_curve-res{args.n_res}-ch{args.ch_opt}-trans-ctxt-fold-{fold_id+1}.png')
                utils.plot_loss(figure_path, fold_id, epoch, train_loss_list, valid_loss_list)