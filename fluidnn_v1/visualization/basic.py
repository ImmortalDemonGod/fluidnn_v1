import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(loss_history, filename='loss_curve.png'):
    """
    Plots a simple loss curve over training steps.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color='blue', label='Training Loss')
    plt.title('Loss Curve')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Loss curve saved to {filename}")

def plot_gradient_vector_field(grad_dict, title_suffix=""):
    """
    Plots a simple vector field for a 2D gradient matrix (e.g., linear layer).
    If multiple params exist, only shows the first one as a demo for simplicity.
    """
    if not grad_dict:
        print("No gradients to visualize.")
        return

    # Grab the first item from the dictionary
    first_key = next(iter(grad_dict))
    grad = grad_dict[first_key]
    if len(grad.shape) != 2:
        print(f"Skipping vector field: gradient shape {grad.shape} not 2D.")
        return

    # 2D shape, can attempt quiver plot
    fig, ax = plt.subplots(figsize=(6, 5))
    rows, cols = grad.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    U = grad.numpy()
    V = np.zeros_like(U)
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    ax.set_aspect('equal')
    ax.set_title(f"Gradient Vector Field {title_suffix}")
    ax.invert_yaxis()
    plt.savefig('gradient_field.png')
    plt.close()
    print("Gradient field saved to gradient_field.png")
