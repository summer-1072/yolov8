import torch
import numpy as np

y = torch.tensor([0, 0, 1])
x = torch.tensor([0, 1, 1])

matches = torch.tensor([[0, 0, 0.25], [0, 1, 0.75], [1, 1, 0.5]]).numpy()

print('matches', matches)
matches = matches[matches[:, 2].argsort()[::-1]]
matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
matches = matches[matches[:, 2].argsort()[::-1]]
matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

print('matches', matches)