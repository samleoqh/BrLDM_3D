import os
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from monai import transforms
from monai.utils import set_determinism
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from tqdm import tqdm

from brlp import const
from brlp import utils
from brlp import networks
from brlp import (
    get_dataset_from_pd,
    sample_using_diffusion
)
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, CosineAnnealingLR, PolynomialLR

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    # _dict['context'] = torch.tensor([ _dict[c] for c in const.CONDITIONING_VARIABLES ]).unsqueeze(0)
    treatment_cond = _dict["treatment"]
    ages = _dict["age"]
    ses_time = _dict["time"]
    _dict['context'] = torch.tensor([
            ages/100, #0.60,  # ############# age 
            (ses_time-0)/1500.,    # session time 
            treatment_cond, # 0.4,   # ############# diagnosis/ treatment
            0.567, # (mean) cerebral cortex 
            0.539, # (mean) hippocampus
            0.578, # (mean) amygdala
            0.558, # (mean) cerebral white matter
            0.60 , # variable size lateral ventricles  # 0.60
        ]).unsqueeze(0)
     
    return _dict


def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    scale_factor
):
    """
    Visualize the generation on tensorboard
    """

    for tag_i, size in enumerate([ 'CRT', 'TMZ' ]):

        context = torch.tensor([[
            (torch.randint(30, 80, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age 
            # 0, #(torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
            (torch.randint(0, 1500, (1,)) -0) / 1500, 
            # (torch.randint(1, 3,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
            0.20 * (tag_i+1),  # treatment
            0.567, # (mean) cerebral cortex 
            0.539, # (mean) hippocampus
            0.578, # (mean) amygdala
            0.558, # (mean) cerebral white matter
            # 0.30 * (tag_i+1), # variable size lateral ventricles  # 0.60
            0.60, 
        ]])

        image = sample_using_diffusion(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            context=context,
            device=DEVICE, 
            scale_factor=scale_factor
        )

        utils.tb_display_generation(
            writer=writer, 
            step=epoch, 
            tag=f'{mode}/{size}_ventricles',
            image=image
        )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',  required=True, type=str)
    parser.add_argument('--cache_dir',  required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt',  required=True, type=str)
    parser.add_argument('--diff_ckpt',   default=None, type=str)
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=500,     type=int)
    parser.add_argument('--batch_size',  default=16,    type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    # parser.add_argument('--lr',          default=2.5e-4,  type=float)
    args = parser.parse_args()
    
    npz_reader = NumpyReader(npz_keys=['data'])
    
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys=['latent_path'], names=['latent']),
        transforms.LoadImageD(keys=['latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['latent'], channel_dim=0), 
        
        # transforms.RandFlipd(keys=['latent'], spatial_axis=[0], prob=0.5),
        # transforms.RandFlipd(keys=['latent'], spatial_axis=[1], prob=0.5),
        # transforms.RandFlipd(keys=['latent'], spatial_axis=[2], prob=0.5),
        
        transforms.DivisiblePadD(keys=['latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates)
    ])
    
    transforms_fn_val = transforms.Compose([
        transforms.CopyItemsD(keys=['latent_path'], names=['latent']),
        transforms.LoadImageD(keys=['latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['latent'], channel_dim=0), 
        
        # transforms.RandFlipd(keys=['latent'], spatial_axis=[0], prob=0.5),
        # transforms.RandFlipd(keys=['latent'], spatial_axis=[1], prob=0.5),
        # transforms.RandFlipd(keys=['latent'], spatial_axis=[2], prob=0.5),
        
        transforms.DivisiblePadD(keys=['latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates)
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[ dataset_df.split == 'train' ]
    valid_df = dataset_df[ dataset_df.split == 'valid' ]
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn_val, args.cache_dir)
    
    trainset_repeated = ConcatDataset([trainset] * 20)
    
    validset_repeated = ConcatDataset([validset] * 8)

    train_loader = DataLoader(dataset=trainset_repeated, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True,
                              pin_memory=True)
    
    valid_loader = DataLoader(dataset=validset_repeated, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              persistent_workers=True, 
                              pin_memory=True)
    
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE)
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule='scaled_linear_beta', 
        beta_start=0.0015, 
        beta_end=0.0205
    )
    
    # for name, param in diffusion.named_parameters():
    #     for layer in ['conv_in', 'down_blocks', 'middle_block']:
    #         if layer in name:
    #             param.requires_grad=False

    inferer = DiffusionInferer(scheduler=scheduler)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(diffusion.parameters(), lr=args.lr, 
                                # momentum=0.99,
                                # nesterov=True,
                                # weight_decay=3e-5)
    lr_scheduler = PolynomialLR(optimizer, total_iters=100000, power=0.9)
    scaler = GradScaler()
    
    with torch.no_grad():
        with autocast(enabled=True):
            z = trainset[0]['latent']
            
    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")


    writer = SummaryWriter()
    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }
    datasets        = { 'train': trainset, 'valid': validset }
    best_loss = 1.
    best_tr_loss = 1.
    
    for epoch in range(args.n_epochs):
        
        for mode in loaders.keys():
            
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in progress_bar:
                            
                # with autocast(enabled=True):  
                with torch.autocast('cuda', enabled=True):  
                        
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    latents = batch['latent'].to(DEVICE) * scale_factor
                    context = batch['context'].to(DEVICE)
                    n = latents.shape[0]
                                                            
                    with torch.set_grad_enabled(mode == 'train'):
                        
                        noise = torch.randn_like(latents).to(DEVICE)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()

                        noise_pred = inferer(
                            inputs=latents, 
                            diffusion_model=diffusion, 
                            noise=noise, 
                            timesteps=timesteps,
                            condition=context,
                            mode='crossattn'
                        )

                        loss = F.mse_loss(noise.float(), noise_pred.float() )

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    
                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1
        
            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)
            
            if best_tr_loss > epoch_loss and mode == 'train': 
                best_tr_loss = epoch_loss
                savepath = os.path.join(args.output_dir, f'unet-ep-{epoch}-tr_{best_tr_loss:.3f}.pth')
                torch.save(diffusion.state_dict(), savepath)

            # visualize results
            images_to_tensorboard(
                writer=writer, 
                epoch=epoch, 
                mode=mode, 
                autoencoder=autoencoder, 
                diffusion=diffusion, 
                scale_factor=scale_factor
            )

        # save the model
        if best_loss > epoch_loss:  
            best_loss = epoch_loss              
            savepath = os.path.join(args.output_dir, f'unet-ep-{epoch}-val_{best_loss:.3f}.pth')
            torch.save(diffusion.state_dict(), savepath)