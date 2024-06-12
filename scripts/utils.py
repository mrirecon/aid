import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import subprocess

import yaml
import torchvision.transforms.functional as F
import os
from torch import fft as torch_fft
import tempfile

import debugpy

BART_PATH = os.environ.get('BART_PATH')
if BART_PATH is None:
    raise ValueError("BART_PATH not set")

#### MRI UTILS ####

def to_tensor(x, device='cuda'):
    if isinstance(x, torch.Tensor):
        return x
    else:
        dtype = torch.float32 if x.dtype == np.float32 or x.dtype==np.float64 or x.dt else torch.complex64
        return torch.tensor(x, dtype=dtype, device=device)

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.detach().cpu().numpy()

def ifft(x, device='cuda'):
    
    x = to_tensor(x, device)
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1))
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x
    

def fft(x, device='cuda'):
    x = to_tensor(x, device)
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1))
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x

def conj(x, device='cuda'):
    return torch.conj(to_tensor(x, device=device))

def normalize(x):
    return x / torch.max(torch.abs(x))

def bart(nargout, cmd, *args, return_str=False):
    """
    Call bart from the system command line.

    Args:
        nargout (int): The number of output arguments expected from the command.
        cmd (str): The command to be executed by bart.
        *args: Variable number of input arguments for the command.
        return_str (bool, optional): Whether to return the output as a string. Defaults to False.

    Returns:
        list or str: The output of the command. If nargout is 1, returns a single element list.
                     If return_str is True, returns the output as a string.

    Raises:
        Exception: If the command exits with an error.

    Usage:
        bart(<nargout>, <command>, <arguments...>)
    """
    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguments...>)")
        return None

    name = tempfile.NamedTemporaryFile().name

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]
    in_str = ' '.join(infiles)

    for idx in range(nargin):
        writecfl(infiles[idx], args[idx])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]
    out_str = ' '.join(outfiles)

    shell_str = BART_PATH + ' ' + cmd + ' ' + in_str + ' ' + out_str
    print(shell_str)
    if not return_str:
        ERR = os.system(shell_str)
    else:
        try:
            strs = subprocess.check_output(shell_str, shell=True).decode()
            return strs
        except:
            ERR = True

    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    output = []
    for idx in range(nargout):
        elm = outfiles[idx]
        if not ERR:
            output.append(readcfl(elm))
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    if ERR:
        print("Make sure bart is properly installed")
        raise Exception("Command exited with an error.")

    if nargout == 1:
        output = output[0]

    return output

#### DATA UTILS ####

class CustomDataLoader(DataLoader):
    # when want to get slices out of the volume, set slice=True and vol_batch_size=1
    # when want to get batch of volumes and the volumes has 10 slices, set slice=False and volume_size=10 drop_last=True
    def __init__(self, dataset, slice=True, volume_size=1, batch_size=1, vol_batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        super(CustomDataLoader, self).__init__(dataset, vol_batch_size, shuffle, sampler,
                                               batch_sampler, num_workers, collate_fn,
                                               pin_memory, drop_last, timeout,
                                               worker_init_fn)
        self.real_batch_size = batch_size
        self.slice = slice
        self.volume_size = volume_size

    @staticmethod
    def check_length(holder):
        if isinstance(holder, (list)):
            if len(holder) == 0:
                return 0
            else:
                arr = holder[0]
                return arr.shape[0]
        else:
            if holder.numel() == 0:
                return 0
            else:
                return holder.shape[0]
            
    # Override the __iter__ method to customize batch generation
    def __iter__(self):
        # Get the iterator from the parent DataLoader
        self._iterator = super(CustomDataLoader, self).__iter__()

        # Custom behavior before iterating over batches
        holder = []
        # Iterate over batches
        for i, data in enumerate(self._iterator):
            # Custom behavior for each batch
            
            if i == 0:
                holder = data
            else:
                if isinstance(holder, (list)):
                    for k in range(len(holder)):
                        if isinstance(holder[k], torch.Tensor):
                            holder[k] = torch.cat([holder[k], data[k]], 0)
                        elif isinstance(holder[k], list):
                            holder[k] = holder[k] + data[k]
                        else:
                            raise TypeError("Unsupported type to batch: {}".format(type(holder[k])))
                else:
                    holder = torch.cat([holder, data], 0)
            
            needed = self.real_batch_size if self.slice else self.real_batch_size * self.volume_size
            while CustomDataLoader.check_length(holder) > needed:
                batch, holder = CustomDataLoader.aggregate_batch(holder, self.real_batch_size if self.slice else self.real_batch_size * self.volume_size)
                if not self.slice:
                    if isinstance(batch, (list)):
                        for i, b in enumerate(batch):
                            batch[i] = torch.stack(torch.split(b, self.volume_size, 0))
                    else:
                        batch = torch.stack(torch.split(batch, self.volume_size, 0))
                yield batch
                del batch
            
        if not self.drop_last and CustomDataLoader.check_length(holder) > 0:
            batch, holder = CustomDataLoader.aggregate_batch(holder, self.real_batch_size if self.slice else self.real_batch_size * self.volume_size)
            assert CustomDataLoader.check_length(holder) == 0
            yield batch


    @staticmethod
    def mk_batch(torch_tensor, batch_size):
        if  isinstance(torch_tensor, torch.Tensor):
            return torch_tensor[:batch_size, ...], torch_tensor[batch_size:, ...]
        elif isinstance(torch_tensor, (list)):
            return torch_tensor[:batch_size], torch_tensor[batch_size:]
        else:
            raise TypeError("Unsupported type to batch: {}".format(type(torch_tensor)))
    
    @staticmethod
    def aggregate_batch(data_holder, batch_size):
        first_dp = data_holder
        if isinstance(first_dp, (list, tuple)):
            result = [[], []]
            for k in range(len(first_dp)):
                used, remained = CustomDataLoader.mk_batch(first_dp[k], batch_size)
                result[0].append(used)
                result[1].append(remained)
        elif isinstance(first_dp, dict):
            result = {}
            for key in first_dp.keys():
                result[key] = CustomDataLoader.mk_batch(first_dp[key], batch_size)
        else:
            return CustomDataLoader.mk_batch(data_holder, batch_size)
        return result

class ComplexResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Separate real and imaginary parts
        real = img[..., 0]
        imag = img[..., 1]

        resized_real = []
        resized_imag = []
    
        # Resize real and imaginary parts separately with NEAREST interpolation
        for i in range(img.size(0)):  # Loop through batch dimension
            resized_real.append(F.resize(real[i].unsqueeze(0), self.size, interpolation=transforms.InterpolationMode.NEAREST).squeeze(0))
            resized_imag.append(F.resize(imag[i].unsqueeze(0), self.size, interpolation=transforms.InterpolationMode.NEAREST).squeeze(0))

        # Stack resized real and imaginary parts
        resized_real = torch.stack(resized_real, dim=0)
        resized_imag = torch.stack(resized_imag, dim=0)

        # Combine resized real and imaginary parts
        resized_complex_image = torch.stack([resized_real, resized_imag], dim=1)

        return resized_complex_image

def trans_lambda(arr, mag=True):
    x, y, z = arr.shape
    offset = (x-y) // 2
    if mag:
        return arr[offset:x-offset, ...]
    else:
        return np.transpose(cplx2float(arr[offset:x-offset, ...]), (2,0,1,3))

class ComplexToTensor(object):
    def __call__(self, img):
        # Split real and imaginary parts
        real = img[..., 0]
        imag = img[..., 1]

        # Convert real and imaginary parts to tensors
        real_tensor = torch.tensor(real, dtype=torch.float32)
        imag_tensor = torch.tensor(imag, dtype=torch.float32)

        # Stack real and imaginary parts
        return torch.stack([real_tensor, imag_tensor], dim=-1)

def get_transform_fastmri(size, mag=True):
    return transforms.Compose([
    transforms.Lambda(lambda image: trans_lambda(image, mag=mag)),
    transforms.ToTensor() if mag else ComplexToTensor(),
    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST) if mag
    else ComplexResize((size, size)), 
    transforms.RandomHorizontalFlip()])

def get_transform_abide(size):
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST), 
    transforms.RandomHorizontalFlip()])

def get_transform_ssafary(size):
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST) if size != 256 else transforms.Lambda(lambda x: x),
    transforms.RandomHorizontalFlip()])

def collate_fn(batch):
    # you get a list of tensors that contain the volumes.
    # when set vol_batch_size=1, you get a list of a tensor that contains the slices of the volume
    # when set vol_batch_size=2, you get a list of a list of two tensors that contains the slices of the two volume
    # return x0 and x1
    arr = torch.concat(batch, 0)
    arr = arr.permute(0, 2, 3, 1)
    return [arr[0:-1], arr[1:]]

def collate_fn2(batch):
    arr = torch.concat(batch, 0)
    #arr = arr.permute(0, 2, 3, 1)
    return arr


def float2cplx(float_in):
    if isinstance(float_in, torch.Tensor):
        return torch.view_as_complex(float_in)
    return np.array(float_in[...,0]+1.0j*float_in[...,1], dtype='complex64')

def cplx2float(cplx_in):
    if isinstance(cplx_in, torch.Tensor):
        return torch.view_as_real(cplx_in)
    return np.array(np.stack((cplx_in.real, cplx_in.imag), axis=-1), dtype='float32')

def reader_fastmri(path, mag=True):
    arr = readcfl(path[:-4])
    #maxi =  np.max(np.abs(arr), axis=(0,1), keepdims=True)
    maxi =  np.max(np.abs(arr))
    if mag:
        arr = np.abs(arr)/maxi
    else:
        arr = arr/maxi
    return arr

def reader_abide(path):
    arr = readcfl(path[:-4])
    arr = arr[..., 50:-30]
    maxi =  np.max(np.abs(arr))
    return np.abs(arr)/maxi 

def reader_ssafary(path, mag=True):
    arr = readcfl(path[:-4])
    maxi =  np.max(np.abs(arr))
    #maxi =  np.max(np.abs(arr), axis=(0,1), keepdims=True)
    if mag:
        arr = np.abs(arr)/maxi
    else:
        arr = arr/maxi
    return arr

class CustomDataset(Dataset):
    def __init__(self, data, image_size, mag=True, dataset='fastMRI'):
        self.data = data
        self.image_size = image_size
        self.mag = mag
        self.dataset = dataset
        if self.dataset == 'fastmri':
            self.transform = get_transform_fastmri(image_size, mag=mag)
            self.reader = partial(reader_fastmri, mag=mag)
        elif self.dataset == 'abide':
            assert self.mag == True
            self.transform = get_transform_abide(image_size)
            self.reader = reader_abide
        elif self.dataset == 'ssafary':
            self.transform = get_transform_ssafary(image_size)
            self.reader = partial(reader_ssafary, mag=mag)
        else:
            raise ValueError('dataset not supported')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]

        data = self.reader(data)
        data = self.transform(data)
        if self.mag:
            data = data[:, None, ...]
        return data

class Co3DDataset(Dataset):
    
    def __init__(self, dir="/scratch/gluo/co3d", image_size=256, category="apple"):
        from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
        JsonIndexDatasetMapProviderV2)
        from pytorch3d.implicitron.tools.config import expand_args_fields
        from omegaconf import DictConfig      

        self.image_size = image_size
        expand_args_fields(JsonIndexDatasetMapProviderV2)
        dataset_map = JsonIndexDatasetMapProviderV2(
            dataset_root=dir,
            category=category,
            subset_name='fewview_train',
            test_on_train=False,
            only_test_set=False,
            load_eval_batches=False,
            dataset_JsonIndexDataset_args=DictConfig(
                {"remove_empty_masks": True, "load_point_clouds": False})).get_dataset_map()
        
        self.dataset_map = dataset_map
        self.dataset = dataset_map["train"]

        self.sequence_names = list(self.dataset.seq_annots.keys())
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor()
        ])


    def __len__(self):
        # return the number of sequences
        return len(self.sequence_names)

    def __getitem__(self, index):
        # load the sequence
        sequence_name = self.sequence_names[index]
        frames = [x[2] for x in list(self.dataset.sequence_frames_in_order(sequence_name))]

        images = []
        for frame in frames:
            images.append(self.transform(self.dataset[frame].image_rgb))
        images = torch.stack(images)
        return images

def subplot(ax, img, title, cmap, interpolation, vmin, vmax):
    ax.imshow(img, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

plot_params = {'cmap': 'gray', 'interpolation': 'none', 'vmin': 0}
axplot      = partial(subplot, **plot_params)

def fabs(x):
    
    x = to_numpy(x)
        
    if x.dtype is np.float32:
        return x
    else:
        return np.abs(x)
                      

def plot_grid(grid_x, grid_y, images, size=20, titles=None, vmax=1.):
    
    fig, axss = plt.subplots(grid_x, grid_y, figsize=(size, size), gridspec_kw={'width_ratios': [1  for _ in range(grid_y)]})
    
    for i in range(grid_x):
        for j in range(grid_y):
            if i==0:
                if titles is not None:
                    strs=titles[j]
                else:
                    strs='x_%d'%j
            else:
                strs=''
            if grid_x==1:
                if grid_y==1:
                    axplot(axss, fabs(images), title=strs, vmax=vmax)
                else:
                    axplot(axss[j], fabs(images[j]), title=strs, vmax=vmax)
            else:
                axplot(axss[i,j], fabs(images[i*grid_y+j]), title=strs, vmax=vmax)
    plt.tight_layout(pad=0.)

def readcfl(name):
    """
    Read a cfl file and return the data as a NumPy array.

    Parameters:
    name (str): The name of the cfl file (without the extension).

    Returns:
    numpy.ndarray: The data stored in the cfl file, reshaped according to the dimensions specified in the corresponding .hdr file.

    """
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F')

def writecfl(name, array):
    """
    Write a NumPy array to a file in the .cfl format.

    Parameters:
    name (str): The base name of the output file.
    array (ndarray): The NumPy array to be written.

    Returns:
    None
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
        h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()

def check_out(cmd, split=True):
    """ utility to check_out terminal command and return the output"""

    strs = subprocess.check_output(cmd, shell=True).decode()

    if split:
        split_strs = strs.split('\n')[:-1]
    return split_strs

def load_config(path):
    """
    load configuration defined with yaml file
    """
    with open(path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

def save_config(x,path):
    with open(os.path.join(path, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(x, yaml_file, default_flow_style=False, sort_keys=False)


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    seq_length=1,
    mag=True,
    dataset='fastmri',
    num_workers=2,
    slice=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = check_out("find %s -type f -name \"*.cfl\" "%data_dir)
    if dataset != 'co3d':
        dataset = CustomDataset(all_files, image_size, mag=mag, dataset=dataset)
    else:
        dataset = Co3DDataset(dir=data_dir, image_size=image_size, category='apple')
    loader = CustomDataLoader(dataset, batch_size=batch_size, slice=slice, volume_size=1 if slice else seq_length+1, shuffle=True, num_workers=num_workers,
                                      collate_fn=collate_fn2, drop_last=True)
    while True:
        yield from loader

if __name__ == "__main__":
    dir = "/scratch/gluo/ssa_fary_card/vols_256"
    data = load_data(data_dir=dir, batch_size=10, image_size=320, seq_length=16, mag=True, dataset='ssafary')
    a = next(data)
    print(a.shape)
    writecfl('test', a.numpy())

    co3d = load_data(data_dir="/scratch/gluo/co3d", batch_size=10, image_size=256, seq_length=10, mag=True, dataset='co3d', num_workers=5)
    a = next(co3d)
    print(a.shape)
    writecfl('test', a.numpy())