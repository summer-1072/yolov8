import torch
grid_y, grid_x = torch.meshgrid(torch.arange(end=5) + 0.5,
                                torch.arange(end=5) + 0.5, indexing='ij')

print(torch.__version__ >= '1.10.0')