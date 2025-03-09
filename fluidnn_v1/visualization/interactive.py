import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_loss_turbulence(loss_history, turbulence_history, interval=500):
    """
    Creates an animation that updates both the loss curve and turbulence over time.
    interval (ms) sets how frequently frames update.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_xlim(0, len(loss_history))
    ax1.set_ylim(min(loss_history, default=0), max(loss_history, default=1) + 0.1)
    ax1.set_title("Live Loss Curve")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")

    ax2.set_xlim(0, len(turbulence_history))
    ax2.set_ylim(min(turbulence_history, default=0), max(turbulence_history, default=1) + 0.2)
    ax2.set_title("Live Turbulence")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Angle")

    line_loss, = ax1.plot([], [], color='blue')
    line_turb, = ax2.plot([], [], color='red')

    def init():
        line_loss.set_data([], [])
        line_turb.set_data([], [])
        return line_loss, line_turb

    def update(frame):
        # Up to index 'frame' for partial real-time effect
        xdata = list(range(frame))
        line_loss.set_data(xdata, loss_history[:frame])
        line_turb.set_data(xdata, turbulence_history[:frame])
        return line_loss, line_turb

    ani = animation.FuncAnimation(
        fig, update, frames=range(len(loss_history)), 
        init_func=init, interval=interval, blit=True
    )
    plt.tight_layout()
    plt.show()
    return ani
