import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFluidModel(nn.Module):
    """
    A simple feedforward model to demonstrate the fluid visualization concepts.
    Extend or replace with your architecture as needed.
    """
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleFluidModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten input if necessary (like for MNIST)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
