import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fluidnn_v1.core.model import SimpleFluidModel
from fluidnn_v1.core.trainer import FluidTrainer
from fluidnn_v1.visualization.basic import plot_loss_curve, plot_gradient_vector_field
from fluidnn_v1.visualization.turbulence import plot_turbulence, diagnose_turbulence
from fluidnn_v1.visualization.interactive import animate_loss_turbulence

def main():
    """
    Demonstrates how to:
    1) Construct a model
    2) Train with FluidTrainer
    3) Plot results (loss, gradient vector fields, turbulence)
    4) Optionally show animation
    """

    # 1. Create synthetic dataset
    X = np.random.randn(2000, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=(2000,))
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Instantiate model & trainer
    model = SimpleFluidModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = FluidTrainer(model, optimizer)

    # 3. Train for a few epochs
    trainer.train_loop(data_loader, epochs=2)

    # 4. Retrieve logs
    logger = trainer.logger

    # 5. Visualize Loss
    plot_loss_curve(logger.loss_history)

    # 6. Visualize a sample gradient vector field (just the last step)
    if logger.gradient_history:
        last_grads = logger.gradient_history[-1]
        plot_gradient_vector_field(last_grads, title_suffix="(Last Step)")

    # 7. Visualize Turbulence
    plot_turbulence(logger.turbulence_history)
    print("[Turbulence Diagnosis]", diagnose_turbulence(logger.turbulence_history))

    # 8. Optional: Animate the training over steps
    _ = animate_loss_turbulence(logger.loss_history, logger.turbulence_history, interval=400)

if __name__ == "__main__":
    main()
