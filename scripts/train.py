"""
Train a diffusion model on images temporal-spatially.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    causal_model_and_diffusion_defaults,
    create_causal_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import utils
import debugpy

debug = False # when set to True, run the program in single process for debugging
def main():
    args = create_argparser().parse_args()
    assert args.multistage >= 0 and args.multistage <= 2, "multistage must be 0, 1, or 2"

    dist_util.setup_dist(args.devices)
    logger.configure(args.logdir)
    if debug:
        debugpy.listen(5678)
        debugpy.wait_for_client()
        print("debugpy connected")
        debugpy.breakpoint()
    logger.log("creating model and diffusion...")

    image_size = args.image_size
    if args.latent:
        if args.vae:
            args.image_size = image_size // 8
        else:
            args.image_size = image_size // 4
    model, diffusion = create_causal_model_and_diffusion(
        **args_to_dict(args, causal_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = utils.load_data(data_dir=args.data_dir, 
                           batch_size=args.batch_size,
                           image_size=image_size,
                           seq_length=args.seq_length,
                           mag=args.mag,
                           dataset=args.dataset,
                           num_workers=args.dataworkers)
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        latent=args.latent,
        vae=args.vae,
        vq_config=args.vq_config,
        vq_ckpt=args.vq_ckpt,
    ).run_loop(temporal=True, noisy_x0=args.noisy_x0, multistage=args.multistage)


def create_argparser():
    defaults = causal_model_and_diffusion_defaults()
    customized = dict(
        image_size=128,
        data_dir="/scratch/gluo/fastMRI/vols",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=6,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        in_channels=1,
        seq_length=10,
        logdir='/home/gluo/temporal/logs',
        learn_sigma=True,
        dataset='fastmri',
        mag=True,
        dataworkers=2,
        devices=0,
        noisy_x0=False,
        latent=False,
        vae=True,
        vq_config="/home/gluo/temporal/2024-04-18T19-10-06_momo_v1_vqgan/configs/2024-04-18T19-10-06-project.yaml",
        vq_ckpt="/home/gluo/temporal/2024-04-18T19-10-06_momo_v1_vqgan/testtube/version_0/checkpoints/epoch=402-step=68912.ckpt",
        multistage=0, # 0: one-stage training, 1: first stage for two-stage training, 2: second stage for two-stage training
        concat_cond=True,
    )
    defaults.update(customized)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
