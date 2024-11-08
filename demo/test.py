import torch

checkpoint = torch.load("bgl_transformer.pth")
print(checkpoint.keys())