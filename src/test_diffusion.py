import os
import argparse

import torch
import torch.nn.functional as F

from brlp import utils
from brlp import networks
from brlp import (
    get_dataset_from_pd,
    sample_using_diffusion
)


from monai.utils import set_determinism
import nibabel as nib
import numpy as np

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




context = torch.tensor([[
    0.6,  # age 
    # 0, #(torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
    0 / 1500, # 0 - 100
    # (torch.randint(1, 3,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
    0.20 ,  # treatment CRT
    0.567, # (mean) cerebral cortex 
    0.539, # (mean) hippocampus
    0.578, # (mean) amygdala
    0.558, # (mean) cerebral white matter
    # 0.30 * (tag_i+1), # variable size lateral ventricles  # 0.60
    0.60, 
]])

context2 = torch.tensor([[
    0.6,  # age 
    # 0, #(torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
    1200 / 1500, # 0 - 100
    # (torch.randint(1, 3,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
    0.40 ,  # treatment CRT
    0.567, # (mean) cerebral cortex 
    0.539, # (mean) hippocampus
    0.578, # (mean) amygdala
    0.558, # (mean) cerebral white matter
    # 0.30 * (tag_i+1), # variable size lateral ventricles  # 0.60
    0.60, 
]])

def get_contexts(s_step=0, num_ctx = 1000, n_steps = 10):
    ctx_list = []
    for i in range(s_step, num_ctx, n_steps):
        if i < 50:
            ctx_list.append(
                torch.tensor([[
                    0.6,  # age 
                    # 0, #(torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
                    i / 1500., # 0 - 100
                    # (torch.randint(1, 3,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
                    0.20 ,  # treatment CRT
                    0.567, # (mean) cerebral cortex 
                    0.539, # (mean) hippocampus
                    0.578, # (mean) amygdala
                    0.558, # (mean) cerebral white matter
                    # 0.30 * (tag_i+1), # variable size lateral ventricles  # 0.60
                    0.60, 
                ]])
            )
        else:
            ctx_list.append(
                torch.tensor([[
                    0.6,  # age 
                    # 0, #(torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
                    i / 1500., # 0 - 100
                    # (torch.randint(1, 3,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
                    0.40 ,  # treatment CRT
                    0.567, # (mean) cerebral cortex 
                    0.539, # (mean) hippocampus
                    0.578, # (mean) amygdala
                    0.558, # (mean) cerebral white matter
                    # 0.30 * (tag_i+1), # variable size lateral ventricles  # 0.60
                    0.60, 
                ]])
                )
    return ctx_list

if __name__ == '__main__':
    torch.manual_seed(101)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt',  required=True, type=str)
    parser.add_argument('--diff_ckpt',   default=True, type=str)
    args = parser.parse_args()
    
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE).eval()
    z = torch.randn((3, 16, 20, 16)).unsqueeze(0).to(DEVICE)
    
    ctx_list = get_contexts(50, 1200, 20)
    
    for i, context in enumerate(ctx_list):
    
        image = sample_using_diffusion(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            context=context,
            device=DEVICE, 
            scale_factor=1.,
            z = z
        )
        
        image = torch.clamp(image, min=0.)
        image = nib.Nifti1Image(image.cpu().numpy()*255, affine=np.eye(4))
        # nib.save(new_image, os.path.join(args.cache_dir, f'{step}.nii.gz'))
        nib.save(image,  f'diff_generated_CT_day_{i}.nii.gz')
    
    # image2 = sample_using_diffusion(
    #     autoencoder=autoencoder, 
    #     diffusion=diffusion, 
    #     context=context2,
    #     device=DEVICE, 
    #     scale_factor=1.,
    #     z = z
    # )
    
    # image = torch.clamp(image, min=0.)
    # image2 = torch.clamp(image2, min=0.)
    # # image = image - image2
    # # print(image.min()*255, image.max()*255)
    # # print(image.shape)
    # image = nib.Nifti1Image(image.cpu().numpy()*255, affine=np.eye(4))
    # # nib.save(new_image, os.path.join(args.cache_dir, f'{step}.nii.gz'))
    # nib.save(image,  'diff_generated_CRT_day0s11.nii.gz')
    # image2 = nib.Nifti1Image(image2.cpu().numpy()*255, affine=np.eye(4))
    # # nib.save(new_image, os.path.join(args.cache_dir, f'{step}.nii.gz'))
    # nib.save(image2,  'diff_generated_TMZ_day1200s11.nii.gz')
