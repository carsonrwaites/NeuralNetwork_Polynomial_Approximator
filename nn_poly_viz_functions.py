import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_model_pred(model, x, x_scale, y, y_std, y_mean, device):
    X_tensor = torch.from_numpy(x_scale.copy()).reshape(len(x_scale), 1, 1).to(device)
    with torch.no_grad():
        reg = model(X_tensor)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.plot(x, reg.detach().cpu().numpy().reshape(1000, 1)*y_std + y_mean,
            linestyle='--',
            color='orange')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['True Function', 'NN Approximation'])
    plt.show()

def snapshot_callback(model, device, x_vis_scale, y_std, y_mean, snapshots):
    model.eval()
    x_vis = torch.tensor(x_vis_scale)
    x_vis = x_vis.unsqueeze(1).to(device)
    with torch.no_grad():
        y_pred = model(x_vis).cpu().numpy().squeeze() * y_std + y_mean
    snapshots.append(y_pred)
    model.train()

def plot_training_surface(x_vis, y_true, snapshots, snapshot_epochs, model_info=None):
    # Build meshgrid for surface plot
    X, E = np.meshgrid(x_vis, snapshot_epochs)
    # snapshots is already (n_epochs, n_x) so it maps directly

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the predicted surface across training
    ax.plot_surface(X, E, snapshots, cmap='viridis', alpha=0.8)

    # Overlay the true polynomial as a flat surface at the final epoch
    Y_true_surface = np.tile(y_true, (len(snapshot_epochs), 1))
    ax.plot_surface(X, E, Y_true_surface, alpha=0.1, color='red')

    ax.set_xlabel('x')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('y')
    ax.set_title('Model predictions across training')

    plt.tight_layout()
    plt.show()

def plot_interactive_snapshots_old(x_vis, y_true, snapshots, snapshot_epochs, model_info):
    # Build one frame per snapshot
    frames = []
    snapshot_epochs = snapshot_epochs[0]
    for i, epoch in enumerate(snapshot_epochs):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_vis, y=snapshots[i], mode='lines', name='Model', line=dict(color='blue')),
                go.Scatter(x=x_vis, y=y_true, mode='lines', name=f'$f(x) = {model_info['expr']}$', line=dict(color='red', dash='dash'))
            ],
            name=str(epoch)
        ))

    # Initial figure state (epoch 0)
    fig = go.Figure(
        data=[
            go.Scatter(x=x_vis, y=snapshots[0], mode='lines', name='Model', line=dict(color='blue')),
            go.Scatter(x=x_vis, y=y_true, mode='lines', name=f'$f(x) = {model_info['expr']}$', line=dict(color='red', dash='dash'))
        ],
        frames=frames
    )

    # Slider steps
    steps = []
    for epoch in snapshot_epochs:
        steps.append(dict(
            method='animate',
            args=[[str(epoch)], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
            label=str(epoch)
        ))

    fig.update_layout(
        title=f'Model fit across training epochs | Layers: {model_info['num_layers']} | Nodes: {model_info['layer_size']} | Activation: {model_info['activation']}',
        xaxis_title='x',
        yaxis_title='y',
        sliders=[dict(
            currentvalue=dict(prefix='Epoch: '),
            steps=steps
        )],
        yaxis=dict(range=[y_true.min() - (y_true.max() - y_true.min()) * 0.1, y_true.max() + (y_true.max() - y_true.min()) * 0.1])  # fix y axis so curve doesn't jump around
        )

    fig.show()

def plot_interactive_snapshots(x_vis, y_true, snapshots, train_info_snapshots, model_info):
    from plotly.subplots import make_subplots

    snapshot_epochs = train_info_snapshots[0]
    all_epochs = train_info_snapshots[0]
    train_losses = train_info_snapshots[1]
    test_losses = train_info_snapshots[2]

    y_range = [y_true.min() - (y_true.max() - y_true.min()) * 0.1,
               y_true.max() + (y_true.max() - y_true.min()) * 0.1]
    loss_range = [min(train_losses.min(), test_losses.min()) * 0.5,
                  max(train_losses.max(), test_losses.max()) * 1.5]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=('Model Fit', 'Training & Test Loss'),
        vertical_spacing=0.12
    )

    # --- Static traces (never change with slider) ---

    # True polynomial — top panel
    fig.add_trace(go.Scatter(
        x=x_vis, y=y_true, mode='lines',
        name=f'$f(x) = {model_info["expr"]}$',
        line=dict(color='red', dash='dash')
    ), row=1, col=1)

    # Train and test loss curves — bottom panel
    fig.add_trace(go.Scatter(
        x=all_epochs, y=train_losses, mode='lines',
        name='Train loss', line=dict(color='royalblue')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=all_epochs, y=test_losses, mode='lines',
        name='Test loss', line=dict(color='tomato')
    ), row=2, col=1)

    # --- Initial dynamic traces (updated by slider) ---

    # Model prediction — top panel
    fig.add_trace(go.Scatter(
        x=x_vis, y=snapshots[0], mode='lines',
        name='Model', line=dict(color='blue')
    ), row=1, col=1)

    # Vertical epoch marker — bottom panel
    fig.add_trace(go.Scatter(
        x=[snapshot_epochs[0], snapshot_epochs[0]],
        y=[loss_range[0], loss_range[1]],
        mode='lines',
        name='current_epoch_marker',  # named for reliable reference
        line=dict(color='gray', dash='dot', width=1),
        showlegend=False
    ), row=2, col=1)

    # --- Frames ---
    n_static = 3  # true polynomial, train loss, test loss
    dynamic_indices = [n_static, n_static + 1]  # model prediction, vertical line

    frames = []
    for i, epoch in enumerate(snapshot_epochs):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_vis, y=snapshots[i], mode='lines',
                           line=dict(color='blue')),
                go.Scatter(x=[epoch, epoch], y=[loss_range[0], loss_range[1]],
                           mode='lines', line=dict(color='gray', dash='dot', width=1))
            ],
            traces=dynamic_indices,
            name=str(epoch)
        ))

    fig.frames = frames

    # --- Slider ---
    steps = []
    for epoch in snapshot_epochs:
        steps.append(dict(
            method='animate',
            args=[[str(epoch)], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
            label=str(int(epoch))
        ))

    fig.update_layout(
        title=f'Layers: {model_info["num_layers"]} | Nodes: {model_info["layer_size"]} | Activation: {model_info["activation"]}',
        sliders=[dict(currentvalue=dict(prefix='Epoch: '), steps=steps)],
        yaxis=dict(range=y_range),
        xaxis2=dict(range=[all_epochs.min(), all_epochs.max()]),
        yaxis2=dict(type='log', range=[np.log10(loss_range[0]), np.log10(loss_range[1])], fixedrange=True),
        xaxis2_title='Epoch',
        yaxis2_title='MSE Loss (log)',
        height=1000
    )

    fig.show()


def plot_animated_gif(snapshots, snapshot_epochs, x_vis, y_true, model_info, filename="training.gif", fps=15):

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vis, y_true, 'r--', label=f'$f(x) = {model_info['expr']}$')
    line, = ax.plot(x_vis, snapshots[0], 'b-', label='Model')
    epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    fig.suptitle('Model fit across training epochs', fontsize=16)

    # Model information in title
    ax.set_title(f"Layers: {model_info['num_layers']} | Nodes: {model_info['layer_size']} | Activation: {model_info['activation']}",
                 fontsize=9)
    ax.set_ylim(y_true.min() - (y_true.max() - y_true.min()) * 0.1,
                y_true.max() + (y_true.max() - y_true.min()) * 0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.legend()

    def update(frame):
        line.set_ydata(snapshots[frame])
        epoch_text.set_text(f'Epoch: {snapshot_epochs[frame]}')
        return line, epoch_text

    ani = FuncAnimation(fig, update, frames=len(snapshots), interval=1000//fps, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Saved to {filename}")