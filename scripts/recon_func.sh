#!/bin/bash
#SBATCH -p kisski
#SBATCH --nodes=1
#SBATCH --job-name=sampling
#SBATCH --output=/user/luo9/u11161/slurm_log/%x_%j.out

# Training setup
All_ADDR=($(scontrol show hostnames $SLURM_JOB_NODELIST))
GPUS_PER_NODE=4
nodes_array=($All_ADDR)
head_node=${nodes_array[0]}

root_path=/projects/extern/kisski/kisski_aid/dir.project

echo "CUDA_VISIBLE_DEVICES"=$CUDA_VISIBLE_DEVICES
echo "MASTER_ADDR"=$MASTER_ADDR
echo "GPU_NUM"=$(($GPUS_PER_NODE*$SLURM_NNODES))
home=/user/luo9/u11161/temporal-diffusion/temporal
module load cuda
module load miniconda3

job4()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.3 --acc=4 --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_4 
}

job5()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.3 --acc=8 --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_8 
}

job6()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.2 --acc=12 --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_12
}


### cold start, equal spaced mask, with ACS lines ####

job7()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=4 --acs=8 --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_4_acs
}

job8()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=8 --acs=8 --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_8_acs
}

job9()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=12 --acs=8 --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_12_acs
}


### cold start, non equal spaced mask, without ACS lines ####

job13()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.3 --acc=4 --random --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_random_4
}

job14()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.2 --acc=8 --random --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_random_8 
}

job15()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.15 --acc=12 --random --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_random_12
}




### cold start, non equal spaced mask, with ACS lines ####

job16()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=8 --acs=8 --random --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_random_4_acs
}

job17()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=8 --acs=8 --random --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_random_8_acs
}

job18()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/fastmri_320_cplx/model440000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=12 --acs=8 --random --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/coldstart_random_12_acs
}



#### experiments without temporal consistency ####

job19()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.3 --acc=4 --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_4 
}


job20()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.3 --acc=8 --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_8 
}

job21()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.2 --acc=12 --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_12 
}



job22()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=4 --acs=8 --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_4_acs 
}

job23()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=8 --acs=8 --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_8_acs 
}

job24()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=12 --acs=8 --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_12_acs 
}



job25()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.3 --acc=4 --random --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_random_4
}

job26()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.2 --acc=8 --random --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_random_8 
}

job27()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=0.15 --acc=12 --random --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_random_12
}



job28()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=4 --acs=8 --random --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_random_4_acs 
}


job29()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=8 --acs=8 --random --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_random_8_acs 
}

job30()
{
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$4 fastmri_recon.py --config=$home/configs/normal_fastmri_320_cplx.yaml \
    --model_path=$root_path/temporal/logs/normal_fastmri_320_cplx/model170000.pt \
    --h5path_a=$2 \
    --step_size=1 --cond_iters=4 --scale=1 --acc=12 --acs=8 --random --nontemporal --device=$1  --steps=1000 \
    --outdir=$root_path/temporal/results/$3/nontemporal_random_12_acs 
}

run_jobs_array=(true false false false false false)

job_sets=(
  "job19 job22 job23 job24"
  "job4 job5 job6 job13"
  "job7 job8 job9 job14"
  "job16 job17 job18 job15"
  "job25 job26 job27 job21"
  "job28 job29 job30 job20"
)

FILE=${FILE:-default_value_1}

for i in "${!run_jobs_array[@]}"; do
  run_jobs="${run_jobs_array[$i]}"
  if [ "$run_jobs" = true ]; then

    file=$root_path/fastmri/val/$FILE
    

    job_set=(${job_sets[$i]})
    ${job_set[0]} 0 $h5dir/$file $(basename "$file" .h5) 2901 &
    ${job_set[1]} 1 $h5dir/$file $(basename "$file" .h5) 2902 &
    ${job_set[2]} 2 $h5dir/$file $(basename "$file" .h5) 2903 &
    ${job_set[3]} 3 $h5dir/$file $(basename "$file" .h5) 2904 
    wait
  fi
  
done
wait
echo "END TIME: $(date)"