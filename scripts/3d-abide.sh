set -e

download_pretrained(){
    echo "Downloading pretrained models"
    mkdir -p $1
    cd $1

    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/AID/abide.pt
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/abide.cfl
    wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/abide.hdr
    
    chmod +x bart
    export BART_PATH=$PWD/bart
    cd -
}



sampling_abide()
{
    echo "Running sampling"
    torchrun --nnodes 1 --nproc-per-node 1 sample.py --config configs/3d-abide.yaml \
    --model $1/abide.pt --logdir $1/3d-abide_samples --extra_steps=50 \
    --datadir $2
}


logdir=logs

download_pretrained $logdir
sampling_abide $logdir $logdir/abide.cfl
