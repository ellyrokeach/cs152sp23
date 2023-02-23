# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: cs152
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, Module

import matplotlib.pyplot as plt
import numpy as np


# %% [markdown]
# # Prepare The Dataset

# %%
def demo_curve(X: torch.Tensor, noise: float) -> torch.Tensor:
    return torch.sin(X) + torch.randn(X.shape) * noise


class SinusoidDataset(Dataset):
    def __init__(self, num_samples=1000, num_features=1, num_targets=1, noise=0.1):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_targets = num_targets
        self.noise = noise

        self.X = torch.rand(num_samples, num_features) * 2 * torch.pi - torch.pi
        self.y = demo_curve(self.X, noise)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# %%
# Dataset specifications
N = 100
nx = 1
ny = 1

noise = 0.1
data_split = [0.8, 0.2]

# Create a fake dataset and split it into training and validation partitions
full_dataset = SinusoidDataset(N, nx, ny, noise)
train_dataset, valid_dataset = random_split(full_dataset, data_split)

plt.plot(full_dataset.X, full_dataset.y, "o")


# %% [markdown]
# # Design A Neural Network

# %%
class TwoLayerNetwork(Module):
    def __init__(self, num_features, num_hidden, num_targets, activation):
        super(TwoLayerNetwork, self).__init__()
        self.activation = activation
        self.layer1 = Linear(num_features, num_hidden)
        self.layer2 = Linear(num_hidden, num_targets)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


# %% [markdown]
# # Train With (Batch) Gradient Descent

# %%
# Hyperparameters
n1 = 10
activation = F.relu

batch_size = len(train_dataset)
num_epochs = 10
learning_rate = 0.2

# Create data loaders for the training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

# Create the model, loss function, and optimizer
model = TwoLayerNetwork(nx, n1, ny, activation)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

train_losses = []
valid_losses = []

# Run batch gradient descent
for _ in range(num_epochs):

    #
    # Put the model in training mode and update the parameters
    #

    model.train()

    # Grab the entire dataset as a single batch
    X, y = next(iter(train_loader))

    yhat = model(y)

    loss = criterion(yhat, y)
    train_losses.append(loss.detach().item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #
    # Put the model in evaluation mode and compute the validation loss
    #

    model.eval()

    with torch.no_grad():
        X, y = next(iter(valid_loader))
        yhat = model(y)
        loss = criterion(yhat, y)
        valid_losses.append(loss.detach().item())

_, axes = plt.subplots(1, 2, figsize=(12, 4))

test_X = torch.linspace(-np.pi, np.pi, 100)
test_y = demo_curve(test_X, noise=noise)

axes[0].plot(full_dataset.X, full_dataset.y, "o", label="Data")
axes[0].plot(test_X, test_y, label="Model Output")
axes[0].set_title("Model Fit")
axes[0].legend()

axes[1].plot(train_losses, label="Training Loss")
axes[1].plot(valid_losses, label="Validation Loss")
axes[1].set_title("Epoch VS Loss")
axes[1].legend()

# %% [markdown]
# # Train With Stochastic Gradient Descent

# %%
# Hyperparameters
n1 = 10
activation = F.relu

batch_size = 1
num_epochs = 10
learning_rate = 0.2

# Create data loaders for the training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

# Create the model, loss function, and optimizer
model = TwoLayerNetwork(nx, n1, ny, activation)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

train_losses = []
valid_losses = []

# Run batch gradient descent
for _ in range(num_epochs):

    #
    # Put the model in training mode and update the parameters
    #

    model.train()

    # Grab the entire dataset as a single batch
    for X, y in train_loader:
        yhat = model(y)
        loss = criterion(yhat, y)
        train_losses.append(loss.detach().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #
    # Put the model in evaluation mode and compute the validation loss
    #

    model.eval()

    with torch.no_grad():
        X, y = next(iter(valid_loader))
        yhat = model(y)
        loss = criterion(yhat, y)
        valid_losses.append(loss.detach().item())

_, axes = plt.subplots(1, 2, figsize=(12, 4))

test_X = torch.linspace(-np.pi, np.pi, 100)
test_y = demo_curve(test_X, noise=noise)

axes[0].plot(full_dataset.X, full_dataset.y, "o", label="Data")
axes[0].plot(test_X, test_y, label="Model Output")
axes[0].set_title("Model Fit")
axes[0].legend()

axes[1].plot(
    torch.linspace(1, num_epochs, len(train_losses)),
    train_losses,
    label="Training Loss",
)
axes[1].plot(
    torch.linspace(1, num_epochs, len(valid_losses)),
    valid_losses,
    label="Validation Loss",
)
axes[1].set_title("Epoch VS Loss")
axes[1].legend()

# %% [markdown]
# # Train With Minibatch Stochastic Gradient Descent

# %%
# Hyperparameters
n1 = 10
activation = F.relu

batch_size = 16
num_epochs = 10
learning_rate = 0.2

# Create data loaders for the training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

# Create the model, loss function, and optimizer
model = TwoLayerNetwork(nx, n1, ny, activation)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

train_losses = []
valid_losses = []

# Run batch gradient descent
for _ in range(num_epochs):

    #
    # Put the model in training mode and update the parameters
    #

    model.train()

    # Grab the entire dataset as a single batch
    for X, y in train_loader:
        yhat = model(y)
        loss = criterion(yhat, y)
        train_losses.append(loss.detach().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #
    # Put the model in evaluation mode and compute the validation loss
    #

    model.eval()

    with torch.no_grad():
        X, y = next(iter(valid_loader))
        yhat = model(y)
        loss = criterion(yhat, y)
        valid_losses.append(loss.detach().item())

_, axes = plt.subplots(1, 2, figsize=(12, 4))

test_X = torch.linspace(-np.pi, np.pi, 100)
test_y = demo_curve(test_X, noise=noise)

axes[0].plot(full_dataset.X, full_dataset.y, "o", label="Data")
axes[0].plot(test_X, test_y, label="Model Output")
axes[0].set_title("Model Fit")
axes[0].legend()

axes[1].plot(
    torch.linspace(1, num_epochs, len(train_losses)),
    train_losses,
    label="Training Loss",
)
axes[1].plot(
    torch.linspace(1, num_epochs, len(valid_losses)),
    valid_losses,
    label="Validation Loss",
)
axes[1].set_title("Epoch VS Loss")
axes[1].legend()

# %% [markdown]
# # Manually Handling Minibatches
#
# Here is my original version of minibatch SGD. I am leaving it here so that you can see how one might manually create batches instead of relying on the dataset+dataloader approach.
#
# ```python
#
# # Hyperparameters
# n1 = 10
# activation = F.relu
#
# batch_size = 16
# num_epochs = 10
# learning_rate = 0.2
#
# num_batches = x.shape[0] // batch_size
#
# model = TwoLayerNetwork(nx, n1, ny, activation)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# criterion = torch.nn.MSELoss()
#
# losses = []
# for _ in range(num_epochs):
#     shuffled_indices = torch.randperm(x.shape[0])
#
#     for batch in range(num_batches):
#
#         xb = x[shuffled_indices[batch*batch_size:batch*batch_size+batch_size]]
#         yb = y[shuffled_indices[batch*batch_size:batch*batch_size+batch_size]]
#
#         yhatb = model(xb)
#         lossb = criterion(yhatb, yb)
#         losses.append(lossb.detach().item())
#
#         optimizer.zero_grad()
#         lossb.backward()
#         optimizer.step()
# ```

# %%
