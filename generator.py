import torch
import torch.nn as nn
import torchvision.models as models

class Generator(nn.Module):
    def __init__(self, total_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(total_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1600),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.model(z)