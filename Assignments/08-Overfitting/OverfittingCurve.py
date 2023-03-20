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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Overfitting a Curve
#
# In this assignment, you will play around with the various models and corresponding parameters. 
#
# ## Questions to Answer
#
# Things to try:
#
# - **Before you run any code**, make some predictions. What do you expect to see for the different models?
#     + linear
#     + quadratic
#     + cubic
#     + n-degree polynomial
#     + ordinary least squares
#     + neural network
# - Now run the notebook. What surprised you? What matched your expectations?
# - Now report on your results with the following:
#     + Changing the number of degrees in the polynomial model.
#     + Using a non-zero weight decay.
#     + Changing the number of layers in the neural network model.
#     + Changing the number of training samples.
# - Finally, open the `OverfittingFashionMNIST.ipynb` and see if you can get the neural network to overfit the data (get the bad thing to happen).

# %% [markdown]
# ## Imports

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from torchsummary import summary

from fastprogress.fastprogress import progress_bar

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="talk")


# %% [markdown]
# ## Create Fake Training Data

# %%
class CubicDataset(Dataset):
    def __init__(self, num_samples: int, input_range: tuple[float, float]):

        # Internal function to generate fake data
        def fake_y(x, add_noise=False):
            y = 10 * x ** 3 - 5 * x
            return y + torch.randn_like(y) * 0.5 if add_noise else y

        self.num_samples = num_samples
        self.input_range = input_range

        min_x, max_x = input_range

        # True curve for plotting purposes
        true_N = 100
        self.true_X = torch.linspace(min_x, max_x, true_N).reshape(-1, 1)
        self.true_y = fake_y(self.true_X)

        self.X = torch.rand(self.num_samples).reshape(-1, 1) * (max_x - min_x) + min_x
        self.y = fake_y(self.X, add_noise=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def plot(self, model=None, losses=None, poly_deg=None):

        # Plot loss curves if given
        train_losses = losses[0] if losses else None
        valid_losses = losses[1] if losses else None
        plot_losses = train_losses != None and len(train_losses) > 1

        _, axes = plt.subplots(1, 2, figsize=(16, 8)) if plot_losses else plt.subplots(1, 1, figsize=(8, 8))
        ax1: plt.Axes = axes[0] if plot_losses else axes
        ax2: plt.Axes | None = axes[1] if plot_losses else None
 
        ax1.plot(self.X, self.y, "o", label="Noisy Samples")
        ax1.plot(self.true_X, self.true_y, label="Baseline Curve")

        # Plot the model's learned regression function
        if model:
            x = self.true_X.unsqueeze(-1)
            x = x.pow(torch.arange(poly_deg + 1)) if poly_deg else x

            with torch.no_grad():
                pred_y = model(x)

            ax1.plot(self.true_X, pred_y.squeeze(), label="Learned Model")

        ax1.set_xlim(self.input_range)
        ax1.set_ylim(-5, 5)
        ax1.legend()
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # Plot training and validation losses
        if plot_losses and ax2:
            ax2.plot(torch.linspace(1, num_epochs, len(train_losses)), train_losses, label="Training")
            ax2.plot(torch.linspace(1, num_epochs, len(valid_losses)), valid_losses, label="Validation")
            ax2.legend()
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")


# %%
# Number of samples/examples
N = 25
train_valid_split = [0.8, 0.2]
batch_size = N // 4

# Range of training data input
MIN_X, MAX_X = -1, 1

cubic_dataset = CubicDataset(N, (MIN_X, MAX_X))
cubic_dataset.plot()

train_dataset, valid_dataset = random_split(cubic_dataset, train_valid_split)


# %% [markdown]
# ## Training Loop

# %%
def train_model(learning_rate, num_epochs, weight_decay, model, params):
    # Torch utils
    criterion = nn.MSELoss()
    optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

    train_losses = []
    valid_losses = []

    # Training loop
    for _ in progress_bar(range(num_epochs)):

        # Model can be an nn.Module or a function
        if isinstance(model, nn.Module):
            model.train()
        
        for X, y in train_loader:
            yhat = model(X)

            loss = criterion(yhat, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if isinstance(model, nn.Module):
            model.eval()

        with torch.no_grad():
            for X, y in valid_loader:
                yhat = model(X)
                loss = criterion(yhat, y)
                valid_losses.append(loss.item())

    return train_losses, valid_losses


# %% [markdown]
# ## Train a Linear Model Using Batch Gradient Descent

# %%
# Hyperparameters
learning_rate = 0.1
num_epochs = 64
weight_decay = 0

# Model parameters
m = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Place parameters into a sequence for torch.optim
params = (b, m)

# Create simple linear model
def model(X):
    return m * X + b


losses = train_model(learning_rate, num_epochs, weight_decay, model, params)
cubic_dataset.plot(model, losses)

# %% [markdown]
# ## Train a Quadratic Model Using Batch Gradient Descent

# %%
# Hyperparameters
learning_rate = 0.1
num_epochs = 64
weight_decay = 0

# Model parameters
w2 = torch.randn(1, requires_grad=True)
w1 = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Place parameters into a sequence for torch.optim
params = (b, w1, w2)

# Create simple quadratic model
def model(X):
    return b + w1 * X + w2 * X ** 2


losses = train_model(learning_rate, num_epochs, weight_decay, model, params)
cubic_dataset.plot(model, losses)

# %% [markdown]
# ## Train a Cubic Model Using Batch Gradient Descent

# %%
# Hyperparameters
learning_rate = 0.1
num_epochs = 64
weight_decay = 0

# Model parameters
w3 = torch.randn(1, requires_grad=True)
w2 = torch.randn(1, requires_grad=True)
w1 = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Place parameters into a sequence for torch.optim
params = (b, w1, w2, w3)

# Create simple cubic model
def model(X):
    return b + w1 * X + w2 * X ** 2 + w3 * X ** 3


losses = train_model(learning_rate, num_epochs, weight_decay, model, params)
cubic_dataset.plot(model, losses)

# %% [markdown]
# ## Train a Polynomial Model Using Batch Gradient Descent

# %%
# Hyperparameters
learning_rate = 0.1
num_epochs = 64
weight_decay = 0

# Model parameters
degrees = 50  # 3, 4, 16, 32, 64, 128
powers = torch.arange(degrees + 1)
params = torch.randn(degrees + 1, requires_grad=True)

# Create simple cubic model
def model(X):
    X_polynomials = X.pow(powers)
    return X_polynomials @ params


losses = train_model(learning_rate, num_epochs, weight_decay, model, [params])
cubic_dataset.plot(model, losses, poly_deg=degrees)

# %% [markdown]
# ## Compute Polynomial Model Using Ordinary Least Squares

# %%
train_X = torch.tensor([x for x, _ in train_dataset])
train_y = torch.tensor([y for _, y in train_dataset])
train_X_polynomial = train_X.unsqueeze(-1).pow(powers)

# Compute "optimal" parameters
params = ((train_X_polynomial.T @ train_X_polynomial).inverse() @ train_X_polynomial.T) @ train_y

def model(X):
    return X @ params


# Compute loss
mse = nn.functional.mse_loss(train_X_polynomial @ params, train_y)
cubic_dataset.plot(model, losses=None, poly_deg=degrees)

# %%
params.abs().mean()


# %% [markdown]
# ## Train Neural Network Model Using Batch Gradient Descent

# %%
class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNetwork, self).__init__()

        # The hidden layers include:
        # 1. a linear component (computing Z) and
        # 2. a non-linear comonent (computing A)
        hidden_layers = [
            nn.Sequential(nn.Linear(nlminus1, nl), nn.ReLU())
            for nl, nlminus1 in zip(layer_sizes[1:-1], layer_sizes)
        ]

        # For regression we should use a linear output layer
        output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        # Group all layers into the sequential container
        all_layers = hidden_layers + [output_layer]
        self.layers = nn.Sequential(*all_layers)

    def forward(self, X):
        return self.layers(X)


# %%
# Hyperparameters
learning_rate = 0.01
num_epochs = 1000
weight_decay = 0

layer_sizes = (1, 100, 100, 100, 1)

model = NeuralNetwork(layer_sizes)
summary(model)

X = train_X.unsqueeze(-1)

losses = train_model(learning_rate, num_epochs, weight_decay, model, model.parameters())
cubic_dataset.plot(model, losses)

# %%
for param in model.parameters():
    print(param.abs().mean().item())

# %%
# # !jupytext --sync OverfittingCurve.ipynb

# %%
