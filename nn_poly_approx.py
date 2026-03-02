import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from nn_poly_viz_functions import *
x = sp.Symbol('x')

# Polynomial to learn
expr =  x**6 - 12*x**4 + 3*x**3 + 20*x**2 - 5*x - 8

# Model parameters
layer_size = 12
num_layers = 2
activation = nn.Tanh()  # nn.Tanh(), nn.ReLU(), nn.Sigmoid()

# Model training parameters
epochs = 3000  # Maximum of epochs to train for
stop_loss = 0.0001  # What MSE to break training loop at
stag_max = 30  # How many epochs to let run without lowering test loss

class PolynomialDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Transform:
    def __init__(self, data):
        self.t_mean = data.mean()
        self.t_std = data.std()

    def transform(self, data):
        return (data - self.t_mean) / self.t_std

    def inverse_transform(self, data):
        return data * self.t_std + self.t_mean

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

X = np.linspace(-4, 4, 1000).astype(np.float32)
latex_str = sp.latex(expr)
print(f"Training on polynomial: {latex_str}")
f = sp.lambdify(x, expr, "numpy")
Y = f(X)

X_mean, X_std = X.mean(), X.std()
Y_mean, Y_std = Y.mean(), Y.std()
X_scale = (X - X_mean) / X_std
Y_scale = (Y - Y_mean) / Y_std

dataset = PolynomialDataset(X_scale, Y_scale)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device\n")

model_info = {'layer_size': layer_size, 'num_layers': num_layers, 'activation': activation, 'expr': latex_str}

class NeuralNetwork(nn.Module):
    def __init__(self, layer_size=32, num_layers=2, activation=nn.Tanh()):
        super().__init__()
        layers = [nn.Linear(1, layer_size), activation]
        for _ in range(num_layers-1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(activation)
        layers.append(nn.Linear(layer_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = NeuralNetwork(layer_size=layer_size,
                      num_layers=num_layers,
                      activation=activation).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
snapshot_every = 10
progress_every = 50

pred_snapshots = []
train_info_snapshots = [[], [], []]  # epoch number, train loss, test loss

min_loss = 1000  # To keep track of minimum loss
stag_counter = 0  # To keep track of plateauing test loss

for epoch in range(epochs+1):
    train_loss = train(model, train_loader, optimizer, loss_fn, device)
    test_loss = test(model, test_loader, loss_fn, device)

    if epoch % progress_every == 0:
        print(f"Epoch {epoch}: train loss = {train_loss:.6f}, test loss = {test_loss:.6f}")

    if test_loss < min_loss:
        min_loss = test_loss
        stag_counter = 0
    else:
        stag_counter += 1

    if min_loss < stop_loss or stag_counter >= stag_max:
        snapshot_callback(model, device, X_scale, Y_std, Y_mean, pred_snapshots)
        train_info_snapshots[0].append(epoch)
        train_info_snapshots[1].append(train_loss)
        train_info_snapshots[2].append(test_loss)
        break

    # Moved this after the stopping criteria to avoid duplicate snapshots if terminating on a x10
    if epoch % snapshot_every == 0:
        snapshot_callback(model, device, X_scale, Y_std, Y_mean, pred_snapshots)
        train_info_snapshots[0].append(epoch)
        train_info_snapshots[1].append(train_loss)
        train_info_snapshots[2].append(test_loss)

pred_snapshots = np.array(pred_snapshots)
train_info_snapshots = np.array(train_info_snapshots)

pd.DataFrame(train_info_snapshots).to_csv('train_info_snapshots.csv')
pd.DataFrame(pred_snapshots).to_csv('pred_snapshots.csv')

## See final model fit
#plot_model_pred(model, X, X_scale, Y, Y_std, Y_mean, device)

## See 3-dimensional training surface
#plot_training_surface(X, Y, pred_snapshots, train_info_snapshots)

## See 2-dimensional fit with epoch slider
plot_interactive_snapshots(X, Y, pred_snapshots, train_info_snapshots, model_info)

## Export gif of model fit
#plot_animated_gif(pred_snapshots, train_info_snapshots, X, Y, model_info, filename=f"Visualizations/poly_approx4.gif", fps=15)
