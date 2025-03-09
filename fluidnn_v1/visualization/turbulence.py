import matplotlib.pyplot as plt

def plot_turbulence(turbulence_history, filename='turbulence.png'):
    """
    Visualizes turbulence (average gradient angle) over training steps.
    """
    if not turbulence_history:
        print("No turbulence data available.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(turbulence_history, color='red', label='Turbulence')
    plt.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.7, label='Moderate')
    plt.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='High')
    plt.title('Turbulence Over Training')
    plt.xlabel('Training Step')
    plt.ylabel('Average Gradient Angle')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Turbulence plot saved to {filename}")

def diagnose_turbulence(turbulence_history):
    """
    Simple function that prints a textual diagnosis of training stability based on turbulence.
    """
    if not turbulence_history:
        return "No turbulence data collected yet."

    avg_turb = sum(turbulence_history) / len(turbulence_history)
    if avg_turb < 0.3:
        return f"Turbulence is stable (avg={avg_turb:.2f}). Training seems smooth."
    elif avg_turb < 0.7:
        return f"Turbulence is moderate (avg={avg_turb:.2f}). Potential small oscillations."
    else:
        return f"Turbulence is high (avg={avg_turb:.2f}). Training may be unstable."
