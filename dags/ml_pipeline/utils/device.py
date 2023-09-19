import torch

# Get cpu, gpu or mps device for training.
device = (
    "cuda" # programming model from NVIDIA for parallel compute operations on GPUs
    if torch.cuda.is_available()
    else "mps" # high performance GPU training for MacOS devices
    if torch.backends.mps.is_available()
    else "cpu" # regular CPU training
)