#!/bin/bash

logdir=/home/gluo/temporal/logs
expname=fastmri_320_cplx
datadir=/scratch/gluo/fastMRI/vols
image_size=320

first_stage() {
    torchrun --nproc_per_node=2 --nnodes=1 train.py \
    --image_size=${image_size} \
    --dataset=fastmri \
    --in_channels=2 \
    --data_dir=${datadir} \
    --logdir=${logdir}/first_stage_${expname} \
    --use_checkpoint=False \
    --learn_sigma=True \
    --mag=False \
    --dataworkers=5 \
    --batch_size=3 \
    --seq_length=10 \
    --multistage=1 \
    --concat_cond=False
}

second_stage() {
    torchrun --nproc_per_node=2 --nnodes=1 --master_port=23336 train.py \
    --image_size=${image_size} \
    --dataset=fastmri \
    --data_dir=${datadir} \
    --logdir=${logdir}/second_stage_${expname} \
    --in_channels=2 \
    --learn_sigma=True \
    --mag=False \
    --dataworkers=5 \
    --batch_size=1 \
    --seq_length=10 \
    --multistage=2 \
    --concat_cond=False \
    --resume_checkpoint=${logdir}/first_stage_${expname}/model150000.pt \
    --use_checkpoint=False \
    --device=2
}

one_stage() {
    torchrun --nproc_per_node=1 --nnodes=1 --master_port=2335 train.py \
    --image_size=${image_size} \
    --dataset=fastmri \
    --data_dir=${datadir} \
    --logdir=${logdir}/one_stage_${expname} \
    --concat_cond=False \
    --learn_sigma=True \
    --mag=False \
    --dataworkers=5 \
    --batch_size=2 \
    --in_channels=2 \
    --seq_length=10 \
    --multistage=0
}

normal() {
    torchrun --nproc_per_node=4 --nnodes=1 image_train.py \
    --image_size=${image_size} \
    --dataset=fastmri \
    --data_dir=${datadir} \
    --logdir=${logdir}/normal_fastmri_320_cplx \
    --learn_sigma=True \
    --mag=False \
    --dataworkers=5 \
    --batch_size=10 \
    --in_channels=2 \
    --attention_resolutions=20,10
}

# Uncomment the function you want to run
# first_stage
# second_stage
# one_stage
# normal
