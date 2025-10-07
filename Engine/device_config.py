# device_config.py

#import torch

# Default: CPU
#device = torch.device('cpu')

# Later (when torch-directml works on 3.13):
# try:
#     import torch_directml
#     device = torch_directml.device()
#     print(" Using AMD GPU via DirectML")
# except ImportError:
#     device = torch.device('cpu')
#     print(" DirectML not found. Falling back to CPU.")

#print(f" Using device: {device}")

# device_config.py
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Using device: {device}")

