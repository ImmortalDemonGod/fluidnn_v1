import torch
import torch.nn.functional as F
from .utils import FluidLogger

class FluidTrainer:
    """
    Manages model training and invokes fluid-like visualization logic.
    Tracks losses, gradients, and turbulence over training steps.
    """

    def __init__(self, model, optimizer, device='cpu'):
        """
        Args:
            model: PyTorch model (nn.Module)
            optimizer: PyTorch optimizer
            device (str): 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        # For capturing metrics
        self.logger = FluidLogger()

    def train_step(self, x, y):
        """
        Single training step: forward pass, backward pass, optimizer step.
        Captures and logs relevant states for fluid-like visualization.
        """
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        # Forward
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log metrics (loss, gradients, etc.)
        self.logger.log_step(self.model, loss.item())

        return loss.item()

    def train_loop(self, data_loader, epochs=1):
        """
        Loops over the dataset for a specified number of epochs.
        Logs data for visualization after each epoch.
        """
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(data_loader):
                batch_loss = self.train_step(x, y)
                epoch_loss += batch_loss

            avg_loss = epoch_loss / len(data_loader)
            print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

            # Could add "epoch_end" logging or calls to advanced visualizers here
