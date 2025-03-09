import torch
import numpy as np

class FluidLogger:
    """
    Records training metrics for fluid-like visualization.
    Maintains lists of:
    - losses
    - gradient snapshots
    - turbulence scores (angular changes in gradients)
    """

    def __init__(self):
        self.loss_history = []
        self.gradient_history = []
        self.turbulence_history = []

    def log_step(self, model, loss_value):
        """
        Logs one training step's loss and gradients.
        """
        self.loss_history.append(loss_value)

        # Capture current gradients
        current_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                current_grads[name] = param.grad.detach().cpu().clone()

        # Append gradient snapshot
        self.gradient_history.append(current_grads)

        # Calculate turbulence if we have at least 2 steps
        if len(self.gradient_history) > 1:
            self.turbulence_history.append(self._calculate_turbulence())

    def _calculate_turbulence(self):
        """
        Measures angular difference between current and previous step's gradients.
        Returns a float representing turbulence intensity.
        """
        curr = self.gradient_history[-1]
        prev = self.gradient_history[-2]

        angles = []
        for key in curr:
            if key in prev:
                c = curr[key].flatten()
                p = prev[key].flatten()
                if torch.norm(c) > 1e-12 and torch.norm(p) > 1e-12:
                    cosine = torch.dot(c, p) / (torch.norm(c) * torch.norm(p))
                    # Clip for numerical stability
                    cosine = torch.clamp(cosine, -1.0, 1.0)
                    angle = torch.acos(cosine).item()
                    angles.append(angle)
        if angles:
            return float(np.mean(angles))
        return 0.0
