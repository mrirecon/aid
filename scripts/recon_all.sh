file_sets=(
    "file_brain_AXT1POST_200_6002026.h5"
    "file_brain_AXT1POST_201_6002686.h5"
    "file_brain_AXT1POST_201_6002687.h5"
    "file_brain_AXT1POST_201_6002705.h5"
    "file_brain_AXT1POST_201_6002739.h5"
    "file_brain_AXT1POST_201_6002741.h5"
    "file_brain_AXT1POST_201_6002743.h5"
    "file_brain_AXT1POST_201_6002785.h5"
    "file_brain_AXT1POST_201_6002806.h5"
    "file_brain_AXT1POST_203_6000874.h5"
)   


file_sets2=(
    "file_brain_AXT1PRE_200_6002087.h5"
    "file_brain_AXT1PRE_200_6002101.h5"
    "file_brain_AXT1PRE_200_6002108.h5"
    "file_brain_AXT1PRE_200_6002120.h5"
    "file_brain_AXT1PRE_200_6002195.h5"
    "file_brain_AXT1PRE_200_6002352.h5"
    "file_brain_AXT1PRE_200_6002391.h5"
    "file_brain_AXT1PRE_203_6000754.h5"
    "file_brain_AXT1PRE_205_6000118.h5"
    "file_brain_AXT1PRE_210_6001876.h5"
)

for file in "${file_sets2[@]}"; do
sbatch --export=FILE=$file recon_func.sh
done