import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    causal_model_and_diffusion_defaults,
    create_causal_model_and_diffusion,
    match_to_dict,
)

import utils

from functools import partial
import argparse

import ddnm
import debugpy

import einops

def main(config, model_path, use_ddim=False, skip=10, extra_steps=0, logdir=None, devices=0, debug=False, data_dir="", vq_config="", vq_ckpt="", first_stage=False):

    dist_util.setup_dist(devices)
    logger.configure(logdir)

    if debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")  
        debugpy.wait_for_client()
        print("Debugpy connected")
        debugpy.breakpoint()

    if use_ddim:
        logger.log("Using DDIM")
    else:
        logger.log("Using DDPM")
    init_config = match_to_dict(config, causal_model_and_diffusion_defaults())

    if first_stage:
        assert not config['concat_cond']

    if logdir is not None:
        savecfl = lambda name, arr: utils.writecfl(logdir + '/' + name, arr)
    else:
        savecfl = utils.writecfl

    if config['latent']:
        if config['vae']:
            from diffusers.models import AutoencoderKL
            init_config['image_size'] = init_config['image_size'] // 8
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(dist_util.dev())
            @th.no_grad()
            def decode(x, batch_size=1):
                x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
                samples = vae.decode(x/0.18215).sample
                samples = einops.rearrange(samples, '(b t) c h w -> b t c h w', b=batch_size)
                return samples
            @th.no_grad()
            def encode(x, batch_size=1):
                x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
                samples = vae.encode(x).latent_dist.sample().mul_(0.18215)
                samples = einops.rearrange(samples, '(b t) c h w -> b t c h w', b=batch_size)
                return samples
        else:
            from guided_diffusion.train_util import get_vqmodel
            init_config['image_size'] = init_config['image_size'] // 4
            vq = get_vqmodel(vq_config, vq_ckpt).to(dist_util.dev())
            @th.no_grad()
            def decode(x, batch_size=1):
                x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
                xrec = vq.decode(x/0.18215)
                samples = einops.rearrange(xrec, '(b t) c h w -> b t c h w', b=batch_size)
                return samples
            
            @th.no_grad()
            def encode(x, batch_size=1):
                x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
                z, _, [_, _, indices] = vq.encode(x)
                z = z * 0.18215
                z = einops.rearrange(z, '(b t) c h w -> b t c h w', b=batch_size)
                return z


    model, diffusion = create_causal_model_and_diffusion(**init_config)
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if init_config['use_fp16']:
        model.convert_to_fp16()
    model.eval()


    batch_size = 1
    input_shape = [batch_size, init_config['seq_length'], config['in_channels'], init_config["image_size"], init_config["image_size"]]
    clip_denoised = True
    if first_stage:
        model_kwargs = {'multistage': 1}
    else:
        model_kwargs = {}

    if not use_ddim:
        sample_fn = partial(diffusion.p_sample_loop, 
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            temporal=True,
            progress=True,
            noisy_x0=config['noisy_x0'],
        )
    else:
        sample_fn = partial(ddnm.ddnm_retrospective, betas=utils.to_tensor(diffusion.betas, dist_util.dev()), T=1000//skip, mag=config['mag'])

    if not use_ddim:
        t_sample_fn = partial(diffusion.p_sample_temporal_loop, 
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            extra_steps=extra_steps,
            noisy_x0=config['noisy_x0'],
        )
    else:
        t_sample_fn = partial(ddnm.ddnm_prospective_loop, betas=utils.to_tensor(diffusion.betas, dist_util.dev()), T=1000//skip, extra_steps=extra_steps, mag=config['mag'])

    data = utils.CustomDataset(data=[data_dir],
                    image_size=config['image_size'],
                    mag=config['mag'],
                    dataset=config['dataset'])

    seq = next(iter(data)).to(dist_util.dev())[None, :config['seq_length']+1, ...]
    
    if config['latent']:
        seq = (seq - 0.5) / 0.5
        tmp = th.concat([seq, seq, seq], 2)
        seq = encode(tmp, batch_size=1)
    x0 = seq[:, :-1, ...]

    if first_stage:
        logger.log("First stage sampling ....")
    else:
        logger.log("Retrospective sampling ....")

    sample = sample_fn(model, input_shape, x0=x0)  # this is the prediction for the next image and it is supposed to match x0 from dataset, retrospetively

    if not first_stage:
        if config['latent']:
            savecfl('latent_retro', sample.cpu())
            savecfl('latent_seq', seq.cpu())
            sample = decode(sample)
            sample = (sample + 1) / 2
            seq = decode(seq)
            seq = (seq + 1) / 2

        generate_x1 = sample.cpu().numpy()
        if not config['mag']:
            generate_x1 = np.transpose(generate_x1, (0, 1, 3, 4, 2))
            savecfl('sample', utils.float2cplx(generate_x1))
            savecfl('seq', utils.float2cplx(np.transpose(seq.cpu().numpy(), (0, 1, 3, 4, 2))))
        else:
            savecfl('sample', generate_x1)
            savecfl('seq', seq.cpu().numpy())
    else:
        if config['latent']:
            savecfl('latent_first', sample.cpu())
            sample = decode(sample)
            sample = (sample + 1) / 2
            
        generate_x1 = sample.cpu().numpy()
        if not config['mag']:
            generate_x1 = np.transpose(generate_x1, (0, 1, 3, 4, 2))
            savecfl('sample_first', utils.float2cplx(generate_x1))
        else:
            savecfl('sample_first', generate_x1)

    if not first_stage:
        logger.log("Prospective sampling warm start....")
        input_shape[1] = 1
        sample = t_sample_fn(x0, model, input_shape, seq_length=customized['seq_length'], warm_start=True) # this is the prediction for the next image, prospectively
        if config['latent']:
            sample = decode(sample)
            sample = (sample + 1) / 2
        generate_x1 = sample.cpu().numpy()
        if not config['mag']:
            generate_x1 = np.transpose(generate_x1, (0, 1, 3, 4, 2))
            savecfl('sample_t', utils.float2cplx(generate_x1))        
        else:
            savecfl('sample_t', generate_x1)

        logger.log("Prospective sampling cold start....")
        input_shape[1] = 1
        sample = t_sample_fn(th.zeros_like(x0), model, input_shape, seq_length=customized['seq_length'], warm_start=False) # this is the prediction for the next image, prospectively
        if config['latent']:
            sample = decode(sample)
            sample = (sample + 1) / 2
        generate_x1 = sample.cpu().numpy()
        if not config['mag']:
            generate_x1 = np.transpose(generate_x1, (0, 1, 3, 4, 2))
            savecfl('sample_tc', utils.float2cplx(generate_x1))
        else:
            savecfl('sample_tc', generate_x1)

    dist.barrier()
    logger.log("Sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--use_ddim', action='store_true', help='Use ddim')
    parser.add_argument('--skip', type=int, default=1, help='Skip')
    parser.add_argument('--extra_steps', type=int, default=0, help='Extra steps')
    parser.add_argument('--logdir', type=str, help='Path to the log directory')
    parser.add_argument('--datadir', type=str, default="", help='Path to the log directory')
    parser.add_argument('--devices', type=int, default='0', help='Devices')
    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--vq_config', type=str, default="", help='')
    parser.add_argument('--vq_ckpt', type=str, default="", help='')
    parser.add_argument('--first_stage', action='store_true', help='First stage')
    args = parser.parse_args()
    
    customized = utils.load_config(args.config)
    path = args.model
    use_ddim = args.use_ddim
    skip = args.skip
    
    main(customized, path, use_ddim, skip, args.extra_steps, args.logdir, args.devices, args.debug, args.datadir, args.vq_config, args.vq_ckpt, args.first_stage)

