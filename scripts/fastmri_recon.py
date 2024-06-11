import numpy as np

from guided_diffusion import dist_util
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

from tqdm import tqdm
import ddnm
import mri_utils
import h5py
import argparse
import debugpy
import os

## cond diffusion
@torch.no_grad()
def ddim_diffusion(xt, model, b, T, step_size, cond_iters, cond_func=None, temporal=False, x0=None, idx_t=None):
    skip = 1000 / T
    n = xt.size(0)
    losses = []

    times = ddnm.get_schedule_jump(T, 1, 1)
    time_pairs = list(zip(times[:-1], times[1:]))        

    pbar = tqdm(time_pairs)
    pbar_labels = ['loss', 'mean', 'min', 'max']
    
    for i, j in pbar:
        i, j = i*skip, j*skip
        if j<0: j=-1 

        t       = (torch.ones(n) * i).to(xt.device)
        next_t  = (torch.ones(n) * j).to(xt.device)
        at      = ddnm.compute_alpha(b, t.long())
        at_next = ddnm.compute_alpha(b, next_t.long())
        if n > 1 and temporal:
            at = at[..., None]
            at_next = at_next[..., None]
        sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
        a       = at_next.sqrt()[0, 0, 0, 0]
        
        if temporal:
            if n > 1:
                #et = []
                #for idx in range(n):
                 #   et.append(model.forward_sample(torch.cat([x0, xt[idx:idx+1]], dim=1), t[0:1], idx_t=idx_t)[:, :, :2])
                #et = torch.cat(et, dim=0)
                cond_x0 = torch.cat([x0 for _ in range(n)], dim=0)
                et = model.forward_sample(torch.cat([cond_x0, xt], dim=1), t, idx_t=idx_t)[:, :, :2]
            else:
                et = model.forward_sample(torch.concat([x0,xt],dim=1), t, idx_t=idx_t)[:, :, :2]
        else:
            et = model(xt, t)[:, :2]

        
        xt = (1/at.sqrt()) * (xt - et * (1 - at).sqrt())

        if cond_func is not None:
            for _ in range(cond_iters):
                meas_grad = cond_func(xt)
                xt = xt - meas_grad * step_size
        xt = at_next.sqrt() * xt + torch.randn_like(xt) * sigma_t

        if cond_func is not None:
            metrics = [(meas_grad).norm(), (xt).abs().mean(), (xt).abs().min(), (xt).abs().max()]
            ddnm.update_pbar_desc(pbar, metrics, pbar_labels)

    return xt

def cond_func(x, measurement, mask, coilsen, temporal):
    if temporal:
        x = x[0]
    x = torch.view_as_complex(x.permute(0,2,3,1).contiguous())[:,None]
    
    x = torch.sum(mri_utils._ifft(mri_utils._fft(x * coilsen) * mask) * torch.conj(coilsen), axis=1)
    
    grad = x - measurement
    grad = torch.view_as_real(grad).permute(0,3,1,2)
    return grad[None] if temporal else grad

def get_k_cond(zero_filled, mask, coilsen, scale=0.3, temporal=True):
    # get the conditional function to enforce k-space consistency
    grad_params = {'measurement': zero_filled*scale, 'mask': mask, 'coilsen': coilsen, 'temporal': temporal}
    grad_op     = partial(cond_func, **grad_params)
    return grad_op

def K_crop(kspace, shape, axis=(0,1)):
    # Crop k-space data to a desired image shape
    img    = mri_utils.ifft2c(kspace,axes=axis)
    img = mri_utils.crop(img, shape)
    kspace = mri_utils.fft2c(img,axes=axis)
    return kspace

def normalize(imgs, device):
    """
     Normalize complex images to have unit magnitude and represent it with real and imaginary parts on the axis 1
     imgs: complex images with shape (batch, 2, height, width)
     device: torch device
    """
    if isinstance(imgs, torch.Tensor):
        imgs = imgs[0]
        imgs = torch.view_as_complex(imgs.permute(0,2,3,1).contiguous())
        imgs = imgs.cpu().numpy()

    imgs = imgs / np.max(imgs)
    arr = mri_utils.cplx2float(imgs)
    arr = np.transpose(arr, [0, 3, 1, 2]).astype(np.float32) #(1,2,320,320)    
    arr = torch.from_numpy(arr).to(device)
    return arr

def main_recon(config, model_path, mask, kspace_a, temporal=True, size=320, warm_start=None, device='cuda:0', steps=1000, step_size=1., cond_iters=4, scale=0.3, outdir='./results', tune=False):
    
    os.makedirs(outdir, exist_ok=True)
    savecfl = lambda name, arr: utils.writecfl(os.path.join(outdir, name), arr)
    
    
    # initialize model
    if temporal:
        init_config = match_to_dict(config, causal_model_and_diffusion_defaults())
        model, diffusion = create_causal_model_and_diffusion(**init_config)
        model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
        model = model.to(device)
        window = torch.zeros([1,config['seq_length'],2,320,320]).to(device)
    else:
        init_config = match_to_dict(config, model_and_diffusion_defaults())
        model, diffusion = create_model_and_diffusion(**init_config)
        model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
        model = model.to(device)

    betas = torch.from_numpy(diffusion.betas).float().to(device)
    recon = []
    ref = []
    zero_filleds = []

    if temporal:
        kspace_0 = kspace_a[0]
        kspace_0 = K_crop(kspace_0, (size, size))
        csm0 = utils.bart(1, 'caldir 30', kspace_0[np.newaxis, ...])[0]
        img0 = np.sum(mri_utils.ifft2c(kspace_0, axes=(0,1)) * np.conj(csm0), axis=-1)
        ref.append(img0)
        prev_img = normalize(img0[np.newaxis, ...], device)[None]
        kspace_a = kspace_a[1:]

        warmed = False
        if warm_start is None:
            window[:,0] = prev_img    
        else:
            savecfl('warm_start', prev_img.cpu().numpy().squeeze())
            warmed = True
            window[:,...] = warm_start
            window = torch.cat([window[:, 1:, ...], prev_img], dim=1)

    def prepare_und_kspace(kspace, mask):
        csm = utils.bart(1, 'caldir 30', kspace[np.newaxis, ...])[0]
        img = np.sum(mri_utils.ifft2c(kspace, axes=(0,1)) * np.conj(csm), axis=-1)

        und_ksp = kspace * mask[..., np.newaxis]

        und_ksp = torch.from_numpy(und_ksp.astype(np.complex64)).permute(2,0,1)[None].to(device)
        mask_t    = torch.from_numpy(abs(mask).astype(np.float32))[None][None].to(device)
        coilsen = torch.from_numpy(csm.astype(np.complex64)).permute(2,0,1)[None].to(device)

        zero_filled = torch.sum((mri_utils._ifft(und_ksp) * torch.conj(coilsen)),dim=1)
        zero_filled = zero_filled / zero_filled.abs().max()

        return zero_filled, mask_t, coilsen, img

    for idx, iter_k in enumerate(kspace_a):
        kspace = K_crop(iter_k, (size, size))
        zero_filled, mask_t, coilsen, img = prepare_und_kspace(kspace, mask)
        zero_filleds.append(zero_filled)
        grad_op = get_k_cond(zero_filled, mask_t, coilsen, scale, temporal)

        if temporal:
            x = torch.randn([1,1,2,320,320]).to(device)

            if not warmed:
                pos = idx if idx < config['seq_length'] else -1
            else:
                pos = -1 # think about the first idx
            x = ddim_diffusion(x, model, betas, steps, step_size, cond_iters, grad_op, temporal=True, x0=window, idx_t=pos)
            
            if not warmed:
                #prev_img = torch.cat([prev_img, normalize(x, device)[None]], dim=1)
                prev_img = torch.cat([prev_img, x], dim=1)
                window[:,0:idx+2] = prev_img
                if idx+2 == config['seq_length']:
                    warmed = True
            else:
                #window = torch.cat([window[:, 1:, ...], normalize(x, device)[None]], dim=1)
                window = torch.cat([window[:, 1:, ...], x], dim=1)
        else:
            x = torch.randn([1,2,320,320]).to(device)
            x = ddim_diffusion(x, model, betas, steps, step_size, cond_iters, grad_op)

        recon.append(x)
        ref.append(img)

        if tune:
            break

    recon_tmp = np.stack([r[0].cpu().numpy().squeeze() for r in recon], axis=0).transpose(0, 2, 3, 1)
    savecfl('recon', utils.float2cplx(recon_tmp))
    savecfl('ref', np.stack([r for r in ref], axis=0))
    savecfl('zero_filled', np.stack([zf[0].cpu().numpy().squeeze() for zf in zero_filleds], axis=0))
    savecfl('mask', mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--nontemporal", action='store_false', help="Use temporal model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=int, required=False, default='0', help="GPU device")
    parser.add_argument("--acc", type=int, required=False, help="Acceleration factor")
    parser.add_argument("--random", action='store_true', help="equal spaced mask")
    parser.add_argument("--acs", type=int, required=False, default=0, help="Number of ACS lines")
    parser.add_argument("--use-poison", action='store_true', help="Use poisson sampling")
    parser.add_argument("--a_x", type=float, required=False, default=1.5, help="Acceleration factor in x")
    parser.add_argument("--a_y", type=float, required=False, default=1.5, help="Acceleration factor in y")
    parser.add_argument("--h5path_a", type=str, required=True, help="Path to h5 file containing kspace data")
    parser.add_argument("--h5path_b", type=str, required=False, help="Path to h5 file containing kspace data for warm start")
    parser.add_argument("--steps", type=int, required=False, default=1000, help="Number of steps to run the diffusion")
    parser.add_argument("--step_size", type=float, required=False, default=1., help="Step size for k-space consistency")
    parser.add_argument("--cond_iters", type=int, required=False, default=4, help="Number of iterations for k-space consistency")
    parser.add_argument("--scale", type=float, required=False, default=0.3, help="Scale for k-space consistency")
    parser.add_argument("--outdir", type=str, required=False, default='.', help="Output directory")
    parser.add_argument("--tune", action='store_true', help="Tune the reconstruction")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    device = "cuda:%d" % args.device
    data = h5py.File(args.h5path_a)
    kspace_a = np.array(data['kspace']).transpose(0, 2, 3, 1)

    nz, nx, ny, nc = kspace_a.shape
    assert ny == 320, "Only 320x320 images supported"

    if args.h5path_b:
        print("Warm start provided")
        data = h5py.File(args.h5path_b)
        kspace_b = np.array(data['kspace']).transpose(2,3,1,0)
        coilsens = np.zeros_like(kspace_b, dtype='complex64')
        for i in range(kspace_b.shape[-1]):
            s_kspace = kspace_b[..., i]
            coilsens[..., i] = utils.bart(1, 'caldir 30', s_kspace[np.newaxis, ...])
        prev_img = np.sum(mri_utils.ifft2c(kspace_b, axes=(0,1)) * np.conj(coilsens), axis=2)
        prev_img = prev_img[ny//2:nx-ny//2, ...]
        prev_img = prev_img/np.max(prev_img, axis=(0,1), keepdims=True)
        prev_img = prev_img[:,:,-config['seq_length']:]
        prev_img = utils.cplx2float(prev_img)
        prev_img = torch.from_numpy(prev_img).permute(2,3,0,1).to(device)

    else:
        print("No warm start provided")
        prev_img = None

    debug = False
    if debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")  
        debugpy.wait_for_client()
        print("Debugpy connected")
        debugpy.breakpoint()

    if not args.use_poison:
        if not args.random:
            mask = mri_utils.equal_mask(ny, ny, args.acc)
        else:
            mask = mri_utils.random_mask(ny, ny, args.acc)

        if args.acs != 0:
            mask[:, ny//2-args.acs//2:ny//2+args.acs//2] = 1
    else:
        mask = utils.bart(1, 'poisson  -Y %d -Z %d -y %f -z %f -v -C %d' % (ny, ny, args.a_x, args.a_y, args.acs))[0]
    
    main_recon(config, args.model_path, mask, kspace_a, args.nontemporal, 320, warm_start=prev_img,
               device=device, steps=args.steps, cond_iters=args.cond_iters,
               step_size=args.step_size, scale=args.scale, outdir=args.outdir, tune=args.tune)