"""
Helpers for distributed training.
"""

import io
import os

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

def setup_dist(shift=0):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")
    os.environ["LOCAL_RANK"] = str((int(os.environ["LOCAL_RANK"])+shift) % GPUS_PER_NODE)
    th.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def dev(shift=0):
    if th.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            return th.device(f"cuda:{local_rank}")
        else:
            return th.device(f"cuda:{shift % GPUS_PER_NODE}")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

