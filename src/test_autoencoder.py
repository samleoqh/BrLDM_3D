import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from monai.utils import set_determinism
import nibabel as nib

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM



from brlp import const
from brlp import utils
from brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator,
    get_dataset_from_pd  
)


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    args = parser.parse_args()


    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[ dataset_df.split == 'train' ]
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=8, 
                              batch_size=1, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)

    autoencoder   = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    
    # saver = transforms.SaveImage(
    #     output_dir=tempdir,
    #     output_ext=".nii.gz",
    #     output_dtype=np.uint8,
    #     resample=False,
    #     squeeze_end_dims=True,
    #     writer="NibabelWriter",
    # )

    ssim = SSIM().to(DEVICE)

    l1_loss_fn = L1Loss()

    loss = []
    for epoch in range(1):
        
        autoencoder.eval()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            with torch.no_grad():

                images = batch["image"].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                reco = torch.clamp(reconstruction,min=0)
                images = torch.clamp(images,min=0)
                # new_image = nib.Nifti1Image(images.cpu().numpy()[0,0, ...]*255, affine=np.eye(4))
                new_recon = nib.Nifti1Image(reco.cpu().numpy()[0,0,...]*255, affine=np.eye(4))
                # nib.save(new_image, os.path.join(args.cache_dir, f'{step}.nii.gz'))
                nib.save(new_recon, os.path.join(args.cache_dir, f'{step}_reco_noft.nii.gz'))

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                ssm = ssim(reconstruction.float(), images.float())
                print(f'ssim: {ssm.item()}')
                # print(f"{rec_loss.item()}")
                loss.append(rec_loss.item())
            
            if step < 2: continue
            else:break
        print(sum(loss)/len(loss))

        