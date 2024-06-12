set -e

download_pretrained(){
    echo "Downloading pretrained models"
    mkdir -p $1
    cd $1

    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/AID/fastmri_320_cplx.pt
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/AID/normal_fastmri_320_cplx.pt
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/AID/cardiac_latent.pt
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/image_seq.cfl
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/image_seq.hdr
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/cardiac_seq.cfl
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/cardiac_seq.hdr
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/file_brain_AXT1POST_200_6002026.h5
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/bart
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Autoencoder/vqvae/vq_cardiac_d4.ckpt
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Autoencoder/vqvae/vq_cardiac_d4.yaml

    chmod +x bart
    export BART_PATH=$PWD/bart
    cd -
}

unfolding(){
    echo "Running unfolding"
    python unfolding.py --config=configs/fastmri_320_cplx.yaml \
    --model_path=$3/fastmri_320_cplx.pt \
    --h5path=$1/$2.h5 \
    --step_size=1 --cond_iters=5 --scale=0.5 --steps=1000 --acc=2 --N=10 \
    --outdir=$3/example_unfolding_ar

    python unfolding.py --config=configs/normal_fastmri_320_cplx.yaml \
    --model_path=$3/normal_fastmri_320_cplx.pt \
    --h5path=$1/$2.h5 \
    --step_size=1 --cond_iters=5 --scale=0.5 --nontemporal --steps=1000 --acc=2 --N=10 \
    --outdir=$3/example_unfolding_normal
}

reconstruction(){
    echo "Running volume reconstruction"
    torchrun --nproc_per_node=1 --nnodes=1 fastmri_recon.py --config=configs/fastmri_320_cplx.yaml \
    --model_path=$3/fastmri_320_cplx.pt \
    --h5path_a=$1/$2.h5 \
    --step_size=1 --cond_iters=4 --scale=0.3 --acc=4 --steps=1000 \
    --outdir=$3/example_reconstruction_ar
}

sampling_brain()
{
    echo "Running sampling"
    torchrun --nnodes 1 --nproc-per-node 1 sample.py --config configs/fastmri_320_cplx.yaml \
    --model $1/fastmri_320_cplx.pt --logdir $1/example_samples --extra_steps=20 \
    --datadir $2
}

sampling_cardiac()
{
    torchrun --nnodes 1 --nproc-per-node 1 sample.py \
    --config configs/cardiac_latent.yaml --model $1/cardiac_latent.pt \
    --logdir $1/example_cardiac \
    --extra_steps=0 --datadir $2 \
    --vq_ckpt $1/vq_cardiac_d4.ckpt --vq_config $1/vq_cardiac_d4.yaml
}

logdir=logs
filename=file_brain_AXT1POST_200_6002026

download_pretrained $logdir
sampling_brain $logdir $logdir/image_seq.cfl
unfolding $logdir $filename $logdir
reconstruction $logdir $filename $logdir
sampling_cardiac $logdir $logdir/cardiac_seq