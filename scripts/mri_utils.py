from numpy.fft import fft, fft2 as fft2d, ifft2 as ifft2d, ifft, ifftshift, fftshift
import numpy as np

def fftc(x, axis=- 1):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x), axis=axis, norm="ortho"))


def ifftc(x, axis=- 1):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x), axis=axis, norm="ortho"))


def fft2c(x, axes=(- 2, - 1)):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    #axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2d(ifftshift(x), axes=axes, norm="ortho"))
    return res


def ifft2c(x, axes=(- 2, - 1)):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    #axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2d(ifftshift(x), axes=axes, norm="ortho"))
    return res

def sos(x, axis=- 1):
    '''
    root mean sum of squares, default on first dim
    '''
    res = np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    return res
    
def rsos(x, axis=0):
    '''
    root mean sum of squares, default on first dim
    '''
    res = np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    return res

def zpad(array_in, outshape):
    import math
    #out = np.zeros(outshape, dtype=array_in.dtype)
    oldshape = array_in.shape
    assert len(oldshape)==len(outshape)
    #kspdata = np.array(kspdata)
    pad_list=[]
    for iold, iout in zip(oldshape, outshape):
        left = math.floor((iout-iold)/2)
        right = math.ceil((iout-iold)/2)
        pad_list.append((left, right))

    zfill = np.pad(array_in, pad_list, 'constant')
    return zfill

def crop(img, bounding):
    import operator
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices].copy()

import torch.fft as torch_fft
def _ifft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x

def _fft(x):
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x

def float2cplx(float_in):
    return np.array(float_in[...,0]+1.0j*float_in[...,1], dtype='complex64')

def cplx2float(cplx_in):
    return np.array(np.stack((cplx_in.real, cplx_in.imag), axis=-1), dtype='float32')

def equal_mask(nx, ny, factor):
    mask = np.zeros((nx, ny), dtype=np.float32)
    mask[:, ::factor] = 1
    return mask


def random_mask(nx, ny, factor, seed=1234):
    np.random.seed(seed)
    mask = np.zeros([nx,ny],dtype=np.complex_)
    lines = np.random.choice(range(0, ny), ny//factor, replace=False)
    mask[:,lines] = 1
    return mask

def custom_probability_density(x, start, end, rho=1):
    mid = (start + end) / 2
    slope = 1 / (mid - start)
    flat_zone = 1
    
    if x > mid - flat_zone or x < mid + flat_zone:
        return slope * 1
    else:
        return slope * abs(mid - x)**(1/rho)

def generate_random_numbers(start, end, count):
    probabilities = [custom_probability_density(x, 0, end, rho=2) for x in range(0, end)]
    probabilities /= np.sum(probabilities)
    random_seed = 42
    np.random.seed(random_seed)
    numbers = np.arange(start, end)
    return np.random.choice(numbers, count, p=probabilities, replace=False)

def high_mask(nx, ny, factor):
    mask = np.zeros([nx,ny],dtype=np.complex_)
    lines = generate_random_numbers(0, ny, ny//factor)
    mask[:,lines] = 1
    return mask
