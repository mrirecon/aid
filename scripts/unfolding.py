# Description: This file contains the code for the temporal unfolding of the diffusion model.
# The code is based on the guided diffusion codebase.

import numpy as np

from guided_diffusion.script_util import (
    causal_model_and_diffusion_defaults,
    create_causal_model_and_diffusion,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    match_to_dict,
)

import utils

from functools import partial
import torch

from fastmri_recon import ddim_diffusion, K_crop, normalize

import mri_utils
import h5py
import argparse
import debugpy
import os


def cond_func(x, measurement, mask, temporal=True):
    if temporal:
        x = x[:, 0, ...]
    x = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())[:None]
    x = mri_utils._ifft(mri_utils._fft(x)*mask)
    grad = x - measurement
    grad = torch.view_as_real(grad).permute(0,3,1,2)
    return grad[:,None, ...] if temporal else grad


def main_recon(config, model_path, kspace, acc, temporal=True, size=320, device="cuda:0", steps=1000, step_size=1., cond_iters=4, scale=0.3, outdir='./results', N=1):
    os.makedirs(outdir, exist_ok=True)

    savecfl = lambda name, arr: utils.writecfl(os.path.join(outdir, name), arr)

    if temporal:
        init_config = match_to_dict(config, causal_model_and_diffusion_defaults())
        model, diffusion = create_causal_model_and_diffusion(**init_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device)
        window = torch.zeros([1, config['seq_length'], 2, size, size]).to(device)
    else:
        init_config = match_to_dict(config, model_and_diffusion_defaults())
        model, diffusion = create_model_and_diffusion(**init_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device)

    betas = torch.from_numpy(diffusion.betas).float().to(device)

    idx = 4
    kspace_a = kspace[idx]
    kspace_a = K_crop(kspace_a, (size, size))
    csm0 = utils.bart(1, 'caldir 30', kspace_a[np.newaxis, ...])[0]
    img0 = np.sum(mri_utils.ifft2c(kspace_a, axes=(0,1))*np.conj(csm0), axis=-1)

    prev_img = normalize(img0[np.newaxis, ...], device)[None]
    if temporal:
        window[:,0] = prev_img


    kspace_b = kspace[idx+1]
    kspace_b = K_crop(kspace_b, (size, size))
    csm1 = utils.bart(1, 'caldir 30', kspace_b[np.newaxis, ...])[0]
    img1 = np.sum(mri_utils.ifft2c(kspace_b, axes=(0,1))*np.conj(csm1), axis=-1)

    kspace_single = mri_utils.fft2c(img1, axes=(0,1))
    mask = mri_utils.equal_mask(size, size, acc)
    und_kspace = kspace_single * mask

    zero_filled = mri_utils._ifft(torch.from_numpy(und_kspace.astype(np.complex64))).to(device)
    zero_filled = zero_filled[None] / zero_filled.abs().max()

    mask_t  = torch.from_numpy(abs(mask).astype(np.float32))[None].to(device)
    grad_op = partial(cond_func, measurement=zero_filled*scale, mask=mask_t, temporal=temporal)

    if temporal:
        x = torch.randn([N,1,2,320,320]).to(device)
        x = ddim_diffusion(x, model, betas, steps, step_size, cond_iters, grad_op, temporal=True, x0=window, idx_t=0)
    else:
        x = torch.randn([N,2,320,320]).to(device)
        x = ddim_diffusion(x, model, betas, steps, step_size, cond_iters, grad_op)
    if temporal:
        x = x[:,0,...]
    savecfl('recon', utils.float2cplx(x.cpu().numpy().transpose(0,2,3,1).squeeze()))
    savecfl('zero_filled', zero_filled.cpu().numpy().squeeze())
    savecfl('cond_prev', img0)
    savecfl('ref', img1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=int, required=False, default='0', help="GPU device")
    parser.add_argument("--h5path", type=str, required=True, help="Path to h5 file containing kspace data")
    parser.add_argument("--steps", type=int, required=False, default=1000, help="Number of steps to run the diffusion")
    parser.add_argument("--step_size", type=float, required=False, default=1., help="Step size for k-space consistency")
    parser.add_argument("--cond_iters", type=int, required=False, default=4, help="Number of iterations for k-space consistency")
    parser.add_argument("--scale", type=float, required=False, default=0.3, help="Scale for k-space consistency")
    parser.add_argument("--acc", type=int, required=False, default=2, help="Acceleration factor")
    parser.add_argument("--outdir", type=str, required=False, default='.', help="Output directory")
    parser.add_argument("--nontemporal", action='store_false', help="Use temporal model")
    parser.add_argument("--N", type=int, required=False, default=1, help="Number of samples")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    device = "cuda:%d" % args.device
    data = h5py.File(args.h5path)
    kspace_a = np.array(data['kspace']).transpose(0, 2, 3, 1)

    nz, nx, ny, nc = kspace_a.shape
    assert ny == 320, "Only 320x320 images supported"

    debug = False
    if debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")  
        debugpy.wait_for_client()
        print("Debugpy connected")
        debugpy.breakpoint()

    main_recon(config, args.model_path, kspace_a, args.acc, args.nontemporal, 320, device=device, steps=args.steps, cond_iters=args.cond_iters,
               step_size=args.step_size, scale=args.scale, outdir=args.outdir, N=args.N)
