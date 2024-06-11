import os
import numpy as np
import h5py
import argparse


from tqdm import tqdm
from numpy.fft import fftshift, ifftshift, ifft2

import utils

def write_rss(filepath, save_folder, file_id):

    try:
        fs = h5py.File(filepath, "r")
        kspace = fs['kspace']
        contrast = os.path.basename(filepath).split('_')[2]
    except:
        print(filepath+' is corrupted')
        return file_id
    
    kspace = np.transpose(kspace, [2,3,1,0])
    coil_imgs = fftshift(ifft2(ifftshift(kspace), axes=(0,1)))

    coilsens = np.zeros_like(coil_imgs, dtype='complex64')
    
    
    for i in range(kspace.shape[-1]):
        s_kspace = kspace[..., i]
        coilsens[..., i] = utils.bart(1, 'caldir 30', s_kspace[np.newaxis, ...])

    vol = np.squeeze(np.sum(coil_imgs*np.conj(coilsens), axis=2))
    shape = vol.shape[0:2]
    out_shape = (shape[0], shape[0]//2)
    if shape != out_shape:
        vol = utils.bart(1, 'resize -c 0 %d 1 %d'%(out_shape[0], out_shape[1]), vol)
    savename = os.path.join(save_folder, utils.getname("fastMRI", contrast, "cplx", file_id))
    utils.writecfl(savename, vol)

    fs.close()
    file_id = file_id+1

    return file_id



def main(data_folder, save_folder, start_id):
    try:
        files_list = utils.check_out("find %s -type f -name *.h5"%data_folder)
    except:
        print("No files found")
        return
    file_id = start_id
    for i in tqdm(range(len(files_list))):
        file_id = write_rss(files_list[i], save_folder, file_id)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_folder', metavar='path', default='/scratch/gluo/compressed/fastMRI/multicoil_train', help='')
    parser.add_argument('--save_folder', metavar='path', default='/scratch/gluo/fastMRI', help='')
    parser.add_argument('--start_id', type=int, metavar='int', default=1000000, help='')
    args = parser.parse_args()
    
    main(args.data_folder, args.save_folder, args.start_id)

