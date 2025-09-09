import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Function to load the file list from the text file
def load_file_list(text_file):
    with open(text_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Function to group files by (patient, date)
def group_files(files):
    from collections import defaultdict
    groups = defaultdict(dict)
    for file in files:
        if not file.endswith('.npy'):
            continue
        parts = file.split('_')
        if len(parts) != 3:
            continue
        patient, date, modality = parts[0], parts[1], parts[2].split('.')[0]
        groups[(patient, date)][modality] = file
    return groups

class ACRINDataset(Dataset):
    def __init__(self, data_dir, text_file, transform=None):
        """
        Args:
            data_dir (str): Directory where .npy files are stored.
            text_file (str): Path to the text file listing .npy files (e.g., 'acrin.txt').
            transform (callable, optional): Transform to apply to volumes.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.groups = group_files(load_file_list(text_file))
        self.samples = list(self.groups.keys())
        self.modalities = ['T1', 'T1gd', 'T2', 'FLAIR', 'FMISO']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient, date = self.samples[idx]
        mods = self.groups[(patient, date)]
        
        if not all(mod in mods for mod in self.modalities):
            raise ValueError(f"Sample {patient}_{date} missing modalities: {self.modalities}")

        volumes = []
        for mod in self.modalities:
            path = os.path.join(self.data_dir, mods[mod])
            volume = torch.from_numpy(np.load(path)).float()  # Shape: (height, width, depth)
            volumes.append(volume)

        stacked = torch.stack(volumes, dim=0)  # Shape: (num_modalities, height, width, depth)
        if self.transform:
            stacked = self.transform(stacked)

        return stacked

class CustomDataLoader(DataLoader):
    """
    Custom DataLoader that supports two modes:
    - Slice mode (slice=True): returns batches of slices
    - Volume mode (slice=False): returns batches of volumes,
      where each volume has `volume_size` slices
    """

    def __init__(self, dataset, slice=True, volume_size=1, batch_size=1, slice_dim=-1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        super().__init__(
            dataset,
            batch_size=1,  # fetch one at a time, batching is manual
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        self.real_batch_size = batch_size
        self.slice = slice
        self.volume_size = volume_size
        self.slice_dim = slice_dim  # Dimension to slice along (0, 1, 2, or -1 for last dimension)

    def _check_length(self, x: torch.Tensor) -> int:
        """Return the size of the slicing dimension."""
        if self.slice_dim < 0:
            dim = x.ndim + self.slice_dim
        else:
            dim = self.slice_dim
        return x.shape[dim] if x.numel() > 0 else 0

    def _split_batch(self, x: torch.Tensor, n: int):
        """Split tensor along the slicing dimension into (head[:n], tail[n:])."""
        if self.slice_dim < 0:
            dim = x.ndim + self.slice_dim
        else:
            dim = self.slice_dim
        
        # Use torch.narrow to slice along the specified dimension
        head = torch.narrow(x, dim, 0, n)
        tail = torch.narrow(x, dim, n, x.shape[dim] - n)

        return head, tail

    @staticmethod
    def permute_to_shape(x: torch.Tensor, sliced_dim: int, dest_dim: int) -> torch.Tensor:
        """
        # this permute is buggy when shape is (5,256,5,256)
        Permute tensor so that the dimension of size `sliced_dim` 
        is moved to position `dest_dim`.

        Example:
            x: shape (5, 256, 10, 256)
            permute_to_shape(x, sliced_dim=10, dest_dim=1) -> (5, 10, 256, 256)
        """
        
        dim_index = x.shape.index(sliced_dim)
        perm_order = list(range(len(x.shape)))
        moving_dim = perm_order.pop(dim_index)
        perm_order.insert(dest_dim, moving_dim)
        return x.permute(*perm_order)

    def __iter__(self):
        buffer = None
        for data in super().__iter__():
            data = data[0]
            buffer = data if buffer is None else torch.cat([buffer, data], dim=2)

            needed = self.real_batch_size if self.slice else self.real_batch_size * self.volume_size

            while self._check_length(buffer) >= needed:
                batch, buffer = self._split_batch(buffer, needed)

                if not self.slice: 
                    batch = torch.stack(torch.split(batch, self.volume_size, dim=self.slice_dim))
                    batch = self.permute_to_shape(batch, self.volume_size, 1)
                else:
                    batch = self.permute_to_shape(batch, self.real_batch_size, 0)

                yield batch

        if not self.drop_last and self._check_length(buffer) > 0:
            needed = self.real_batch_size if self.slice else self.real_batch_size * self.volume_size
            batch, buffer = self._split_batch(buffer, needed)
            assert self._check_length(buffer) == 0

            if not self.slice:
                batch = torch.stack(torch.split(batch, self.volume_size, dim=0))

            yield batch

class SeqDataLoader(DataLoader):
    """
    Custom DataLoader that supports two modes:
    - Slice mode (slice=True): returns batches of slices
    - Volume mode (slice=False): returns batches of volumes,
      where each volume has `volume_size` slices
    """

    def __init__(self, dataset, seq_length=1, batch_size=1, slice_dim=-1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        super().__init__(
            dataset,
            batch_size=1,  # fetch one at a time, batching is manual
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        self.real_batch_size = batch_size
        self.seq_length = seq_length
        self.slice_dim = slice_dim  # Dimension to slice along (0, 1, 2, or -1 for last dimension)
        self.needed = self.real_batch_size * self.seq_length

    def _check_length(self, x: torch.Tensor) -> int:
        """Return the size of the slicing dimension."""
        if self.slice_dim < 0:
            dim = x.ndim + self.slice_dim
        else:
            dim = self.slice_dim
        return x.shape[dim] if x.numel() > 0 else 0

    def _split_batch(self, x: torch.Tensor, n: int):
        """Split tensor along the slicing dimension into (head[:n], tail[n:])."""
        if self.slice_dim < 0:
            dim = x.ndim + self.slice_dim
        else:
            dim = self.slice_dim

        head = torch.narrow(x, dim, 0, n)
        tail = torch.narrow(x, dim, n, x.shape[dim] - n)

        return head, tail

    @staticmethod
    def permute_to_shape(x: torch.Tensor, sliced_dim: int, dest_dim: int) -> torch.Tensor:
        """
        Permute tensor so that the dimension of size `sliced_dim` 
        is moved to position `dest_dim`.

        Example:
            x: shape (5, 256, 10, 256)
            permute_to_shape(x, sliced_dim=10, dest_dim=1) -> (5, 10, 256, 256)
        """
        
        dim_index = x.shape.index(sliced_dim)
        perm_order = list(range(len(x.shape)))
        moving_dim = perm_order.pop(dim_index)
        perm_order.insert(dest_dim, moving_dim)
        return x.permute(*perm_order)

    def __iter__(self):
        buffer = None
        for data in super().__iter__():
            data = data[0]
            T, X, Y, Z = data.shape
            data = data.permute(1, 2, 0, 3)
            data = data.reshape(1, X, T*Y, Z)
            buffer = data if buffer is None else torch.cat([buffer, data], dim=2)

            while self._check_length(buffer) >= self.needed:
                batch, buffer = self._split_batch(buffer, self.needed)
                batch = torch.stack(torch.split(batch, self.seq_length, dim=self.slice_dim))
                batch = self.permute_to_shape(batch, self.seq_length, 1)

                yield batch

        if not self.drop_last and self._check_length(buffer) > 0:
            batch, buffer = self._split_batch(buffer, self.needed)
            assert self._check_length(buffer) == 0
            batch = torch.stack(torch.split(batch, self.seq_length, dim=0))
            yield batch


def load_data(
    data_dir,
    batch_size,
    seq_length=1,
    dataset='acrin',
    filelist="path/to/filelist",
    slice=False,
):
    dataset = ACRINDataset(data_dir, text_file=filelist, transform=None)
    loader = CustomDataLoader(dataset, slice=slice, volume_size= 1 if slice else seq_length, batch_size=batch_size, slice_dim=-2)
    while True:
        yield from loader

def load_data2(
    data_dir,
    batch_size,
    seq_length=1,
    dataset='acrin',
    filelist="path/to/filelist",
):
    dataset = ACRINDataset(data_dir, text_file=filelist, transform=None)
    loader = SeqDataLoader(dataset, seq_length, batch_size=batch_size, slice_dim=-2)
    while True:
        yield from loader

if __name__ == '__main__':
    data_dir='/run/user/1001/gvfs/smb-share:server=nasusers.crp-sante.healthnet.lu,share=braine/Datasets/ACRIN/npy'
    #dataset = ACRINDataset(data_dir, text_file='/home/gluo/Documents/aid/scripts/filelist', transform=None)
    #loader = CustomDataLoader(dataset, slice=True, volume_size=1 , batch_size=10, slice_dim=-2)

    
    data = load_data(data_dir, batch_size=50, seq_length=1, dataset='acrin', filelist='/home/gluo/Documents/aid/scripts/filelist', slice=True)

    for i in range(1):
        a = next(data)
        #a = a/torch.amax(torch.abs(a), dim=(0,2,3), keepdim=True)
        print(a.shape)
        np.save('test', a.numpy())

    data = load_data2(data_dir, batch_size=5, seq_length=100, dataset='acrin', filelist='/home/gluo/Documents/aid/scripts/filelist')

    for i in range(1):
        a = next(data)
        #a = a/torch.amax(torch.abs(a), dim=(0,2,3), keepdim=True)
        print(a.shape)
        np.save('test', a.numpy())