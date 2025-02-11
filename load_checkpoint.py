import torch

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    print("Checkpoint loaded successfully")
    return checkpoint

checkpoint_path = 'saved_models/checkpoint.pth'
load_checkpoint(checkpoint_path)