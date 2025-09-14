import os
import argparse
import warnings

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai import transforms
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm
from customloader import CustomLoader_cnet_train
import networks_diff
import utils
from sampling import sample_using_controlnet_and_z



warnings.filterwarnings("ignore")

def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    controlnet, 
    dataset,
    scale_factor,
):
    """
    Visualize the generation on tensorboard
    """
    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice( range(len(dataset)), 3 )
    wmh_path_ = './dataset/aspirin_wmh_slice_trans_aug'
    dem_path_ = './dataset/aspirin_dem_slice_trans_aug'

    for tag_i, i in enumerate(random_indices):

        wmh_latents   = dataset[i][0] * scale_factor
        context       = df.loc[dataset[i][2]].tolist()
        baseline_age  = torch.from_numpy(np.array(df['AGE'].loc[dataset[i][2]].astype(np.float32)))

        base_path = f'{wmh_path_}/{trainset.data_idcs[i]}'
        dem_path  = f'{dem_path_}/{trainset.data_idcs[i].replace("base","dem")}'
        slice_feat = float(trainset.data_idcs[i].split('_')[3])*0.01
        context_npy = df.loc[dataset[i][2]].tolist() + [slice_feat]
        context = torch.from_numpy(context_npy).type(torch.float32).unsqueeze(1).to(DEVICE)

        base_image = torch.from_numpy(np.load(base_path)).unsqueeze(0)
        dem_image  = torch.from_numpy(np.load(dem_path)).unsqueeze(0)
        base_image = resample_fn(base_image).squeeze(0)
        dem_image  = resample_fn(dem_image).squeeze(0)

        predicted_image = sample_using_controlnet_and_z(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            controlnet=controlnet, 
            starting_z=wmh_latents,
            starting_a=baseline_age,
            context=context, 
            device=DEVICE,
            scale_factor=scale_factor
        )

        utils.tb_display_cond_generation(
            writer=writer, 
            step=epoch, 
            tag=f'{mode}/comparison_{tag_i}',
            baseline=base_image,
            ground_truth=dem_image,
            predicted=predicted_image
        )


def compute_mae(z_wmh, z_dem):

    mae_loss = F.l1_loss(z_wmh, z_dem, reduction='mean')
    return mae_loss


def compute_kld(p, q):

    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)

    kld_loss = F.kl_div(p.log(), q, reduction='batchmean')
    return kld_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',    default='cnet_test',   type=str)
    parser.add_argument('--figure_dir',    default='cnet_loss',   type=str)
    parser.add_argument('--aekl_wmh_ckpt', default='./ae_test/save_autoencoder-vae-1.5-wmh-ep-50.pth', type=str)
    parser.add_argument('--aekl_dem_ckpt', default='./ae_test/save_autoencoder-vae-1.5-dem-ep-90.pth', type=str)
    parser.add_argument('--diff_ckpt',     default='./diff_test/unet_dem-res2-ch0-trans-fold-1_best.pth', type=str)
    parser.add_argument('--cnet_ckpt',     default=None,            type=str)
    parser.add_argument('--num_workers',   default=8,               type=int)
    parser.add_argument('--n_epochs',      default=300,             type=int)
    parser.add_argument('--batch_size',    default=8,              type=int)
    parser.add_argument('--lr',            default=1e-6,          type=float)
    parser.add_argument('--gpu_idx',       default=1,               type=int)
    parser.add_argument('--prcs_idx',      default=0,               type=int)
    parser.add_argument('--n_res',         default=2,               type=int)
    parser.add_argument('--ch_opt',        default=0,               type=int)

    
    args = parser.parse_args()


    DEVICE = f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu'
    df_org = pd.read_csv('./picasso_filtered.csv')[
                                                   ['SUBJNO', 'AGE', 'BASE_LOAD', 'SC_Fazekas', 'PROBUCOL',
                                                    'STATIN_INTENSITY', 'SBP', 'PULSE', 'HX_ICH','Current_SM']
                                                   ]
    df_subj = df_org['SUBJNO'].tolist()
    df = df_org[df_org.columns[1:]]

    # load latents that DEM != 0
    treatment = 'aspirin'
    all_wmh_files = os.listdir(f'./dataset/{treatment}_wmh_slice_latent_nz_50')
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
        trainset = CustomLoader_cnet_train(treatment=treatment, data_idcs=train_list)
        validset = CustomLoader_cnet_train(treatment=treatment, data_idcs=valid_list)

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


        autoencoder_wmh = networks_diff.UNet2D_BN_latent(dtype='wmh').to(DEVICE)
        autoencoder_wmh.load_state_dict(torch.load(args.aekl_wmh_ckpt))
        autoencoder_dem = networks_diff.UNet2D_BN_latent(dtype='dem').to(DEVICE)
        autoencoder_dem.load_state_dict(torch.load(args.aekl_dem_ckpt))

        autoencoder_wmh.decoder = autoencoder_dem.decoder
        autoencoder = autoencoder_wmh

        diffusion   = networks_diff.init_latent_diffusion(device=DEVICE,
                                                          n_res=args.n_res,
                                                          ch_opt=args.ch_opt,
                                                          checkpoints_path=args.diff_ckpt.replace("ch0",f"ch{args.ch_opt}"))
        controlnet  = networks_diff.init_controlnet(device=DEVICE, n_res=args.n_res, ch_opt=args.ch_opt, checkpoints_path=None)
        controlnet.time_embed.to(DEVICE)

        if args.cnet_ckpt is not None:
            print('Resuming training...')
            controlnet.load_state_dict(torch.load(args.cnet_ckpt))
        else:
            print('Copying weights from diffusion model')
            controlnet.load_state_dict(diffusion.state_dict(), strict=False)

        # freeze the unet weights
        for p in diffusion.parameters():
            p.requires_grad = False

        scaler = GradScaler()
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

        with torch.no_grad():
            with autocast(enabled=True):
                z = trainset[0][0]


        scale_factor = 1 / torch.std(z)
        print(f"Scaling factor set to {scale_factor}")

        scheduler = DDPMScheduler(num_train_timesteps=1000,
                                  schedule='scaled_linear_beta',
                                  beta_start=0.0015,
                                  beta_end=0.0205)

        writer = SummaryWriter()

        global_counter  = { 'train': 0, 'valid': 0 }
        loaders         = { 'train': train_loader, 'valid': valid_loader }
        datasets        = { 'train': trainset, 'valid': validset }

        train_loss_list = []
        valid_loss_list = []

        best_loss = 1000
        for epoch in range(args.n_epochs):

            for mode in loaders.keys():
                print('mode:', mode)
                loader = loaders[mode]
                controlnet.train() if mode == 'train' else controlnet.eval()
                epoch_loss = 0.
                progress_bar = tqdm(enumerate(loader), total=len(loader))
                progress_bar.set_description(f"Epoch {epoch}")

                for step, batch in progress_bar:
                    # if step > 5:
                    #     break

                    if mode == 'train':
                        optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(mode == 'train'):

                        wmh_latents = batch[0].to(DEVICE) * scale_factor
                        dem_latents = batch[1].to(DEVICE)  * scale_factor
                        idcs = batch[2].numpy()
                        slice_order_feat = batch[3]
                        context_npy = np.concatenate([df.loc[idcs].values.astype(np.float32), np.expand_dims(slice_order_feat, 1)], axis=1)
                        context = torch.from_numpy(context_npy).type(torch.float32).unsqueeze(1).to(DEVICE)
                        # context = torch.zeros_like(context).to(DEVICE)
                        starting_a  = torch.from_numpy(df['AGE'].loc[idcs].values.astype(np.float32)).to(DEVICE)

                        n = wmh_latents.shape[0]

                        with autocast(enabled=True):

                            concatenating_age      = starting_a.view(n, 1, 1, 1).expand(n, * wmh_latents.shape[-3:])
                            controlnet_condition   = torch.cat([ wmh_latents, concatenating_age ], dim=1)

                            noise = torch.randn_like(dem_latents).to(DEVICE)
                            timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
                            images_noised = scheduler.add_noise(dem_latents, noise=noise, timesteps=timesteps)

                            down_h, mid_h = controlnet(
                                x=images_noised.float(),
                                timesteps=timesteps,
                                context=context.float(),    # (b, 1, 9)
                                controlnet_cond=controlnet_condition.float()   # (b, 6, 16, 16)
                            )

                            noise_pred = diffusion(
                                x=images_noised.float(),
                                timesteps=timesteps,
                                context=context.float(),
                                down_block_additional_residuals=down_h,
                                mid_block_additional_residual=mid_h
                            )

                            loss = F.mse_loss(noise_pred.float(), noise.float())

                    if mode == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    #-------------------------------
                    # Iteration end
                    writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                    global_counter[mode] += 1

                # Epoch loss
                epoch_loss = epoch_loss / len(loader)
                writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

                # # Logging visualization
                # images_to_tensorboard(
                #     writer=writer,
                #     epoch=epoch,
                #     mode=mode,
                #     autoencoder=autoencoder,
                #     diffusion=diffusion,
                #     controlnet=controlnet,
                #     dataset=datasets[mode],
                #     scale_factor=scale_factor
                # )
                if mode == 'train':
                    train_loss_list.append(epoch_loss)
                if mode == 'valid':
                    valid_loss_list.append(epoch_loss)

            if epoch_loss < best_loss:
                if os.path.isfile(f'{args.output_dir}/cnet-res{args.n_res}-ch{args.ch_opt}-trans-ep-{best_epoch + 1}-fold-{fold_id + 1}_best.pth'):
                    os.remove(f'{args.output_dir}/cnet-res{args.n_res}-ch{args.ch_opt}-trans-ep-{best_epoch + 1}-fold-{fold_id + 1}_best.pth')
                best_loss = epoch_loss
                best_epoch = epoch
                os.makedirs(args.output_dir, exist_ok=True)
                savepath = os.path.join(args.output_dir,f'cnet-res{args.n_res}-ch{args.ch_opt}-trans-ep-{epoch + 1}-fold-{fold_id + 1}_best.pth')
                torch.save(diffusion.state_dict(), savepath)

            if (epoch + 1) == args.n_epochs:
                os.makedirs(args.output_dir, exist_ok=True)
                savepath = os.path.join(args.output_dir, f'cnet-res{args.n_res}-ch{args.ch_opt}-trans-ep-{epoch+1}-fold-{fold_id+1}.pth')
                torch.save(controlnet.state_dict(), savepath)

                os.makedirs(args.figure_dir, exist_ok=True)
                figure_path = os.path.join(args.figure_dir, f'loss_curve-res{args.n_res}-ch{args.ch_opt}-trans-fold-{fold_id+1}.png')
                utils.plot_loss(figure_path, fold_id, epoch, train_loss_list, valid_loss_list)