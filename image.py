import torch


def discretize(sample):
    """Convert a sample from 0-1 to 0-255."""
    return (sample * 255).to(torch.int32)
