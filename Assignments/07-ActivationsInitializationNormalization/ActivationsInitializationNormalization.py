# -*- coding: utf-8 -*-
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
# # Activations, Initialization, and Normalization
#
# This notebook is all about the range of numbers. Specifically, the range of
#
# - input features (pixels, words, distances, stock data, audio waves, etc.),
# - activation function inputs (weighted sum of outputs from previous layer)
# - activation function outputs (sigmoid, relu, etc.),
# - parameters (weights, biases, etc.), and
# - parameter gradients.
#
# Here are a couple of reminders that you might find helpful.
#
# Each neuron implements these two equations:
#
# $$
# \begin{align}
# Z^{[l]} &= A^{[l-1]} W^{[l]T} + \mathbf{b}^{[l]}\\
# A^{[l]} &= g^{[l]}(Z^{[l]})
# \end{align}
# $$
#
# - $Z^{[l]}$ is the linear output of layer $l$ (e.g., the output of a `nn.Linear` module)
# - $A^{[l-1]}$ is the activation output for layer $l-1$
# - $W^{[l]}$ is a parameter matrix for layer $l$ called "weights"
# - $\mathbf{b}^{[l]}$ is a parameter vector for layer $l$ called "bias"
# - $A^{[l]}$ is the activations for layer $l$ (outputs of an activation function, e.g., `nn.Sigmoid`)
# - $g^{[l]}(\cdot)$ is the activation function for layer $l$ (e.g., sigmoid)
#
# Here are a couple of activation function examples. Pay close attention to the range of the input values (input values are on the x-axis).
#
# ![Sigmoid Activation Function](https://singlepages.github.io/NeuralNetworks/img/Sigmoid.png)
#
# ![ReLU Activation Function](https://singlepages.github.io/NeuralNetworks/img/ReLU.png)

# %% [markdown]
# ## Questions to Answer
#
# These questions will also appear on gradescope.
#
# 1. What terms directly impact the output of an activation function?
# 1. What is the purpose of the bias term in a single neuron?
# 1. What is the purpose of an activation function (what happens when we remove all activation functions)?
# 1. Where is the "interesting" / "useful" range for most activation functions?
# 1. Why are deeper networks generally more useful than shallower networks?
# 1. What happens to gradients in deeper networks?
# 1. Why is it an issue for input features to be in the range from 25 to 35?
# 1. What is a "good" range for input features?
# 1. Can we "normalize" values between layers?
# 1. What is the goal when initializing network parameters?
#
# You can use the code below to help find (or confirm) answers to these conceptual questions.

# %% [markdown]
# ## Imports

# %%
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

from torchsummary import summary

from fastprogress.fastprogress import progress_bar

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

from enum import Enum

jtplot.style(context="talk")


# %% [markdown]
# ## Experimental Parameters and Hyperparameters
#
# I recommend the following process for running quick experiments:
#
# 1. Change a value in the cell below.
# 2. Run the entire notebook (or at least from here down).
# 3. Examine the output plots.
#
# You can also edit and add any code that you find useful.

# %%
class InitMethod(Enum):
    """A list of initialization methods for the weights of a neural network."""
    Zeros = 0
    Ones = 1
    Large = 2
    Uniform = 3
    Normal = 4
    Normal2 = 5
    Xavier = 6
    Kaiming = 7


# %%
# Number of samples/examples
N = 1000
train_valid_split = [0.8, 0.2]
batch_size = 32

# Range of training data input (try something shifted away from 0)
input_range = (-3, 3)

# Noise factor for training data input (what happens if noise is too high?)
input_noise = 0.1

# Horizontal shift of the training data input (useful for demonstrating input normalization)
input_shift = 1

# Noise factor for training data output (what happens if noise is too high?)
output_noise = 0.1

# Vertically shift the sinusoidal output (useful for demonstrating use of bias)
output_shift = 0.5

# Neural network activation function
#   Options: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
#   You should also try nn.Identity
activation_module = nn.ReLU

# Neural network output neuron bias (should the output layer include bias?)
output_bias = True

# Neural network architecture (useful for comparing width and depth)
#   neuron : layer_sizes = []                      # No hidden layers
#   wider  : layer_sizes = [100]                   # One hidden layer with 100 neurons
#   deep   : layer_sizes = [8] * 3                 # Three hidden layers with 8 neurons each
#   deeper : layer_sizes = [80, 90, 80, 70, 80, 5] # Six hidden layers with 80, 90, 80, 70, 80, 5 neurons
neurons_per_hidden_layer = [8]

# Neural network parameter initialization method (see the enumeration above)
initialization_method = InitMethod.Kaiming

# Number of training epochs (useful for examining problematic gradients, set to 1)
num_epochs = 20


# %% [markdown]
# ## Synthetic Dataset

# %%
class SinusoidDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        input_range: tuple[float, float],
        input_shift: float,
        input_noise: float,
        output_shift: float,
        output_noise: float,
    ):
        self.num_samples = num_samples

        # Sinusoidal data without noise
        self.X_no_noise = torch.linspace(*input_range, num_samples).reshape(-1, 1) + input_shift
        self.y_no_noise = torch.sin(self.X_no_noise) + output_shift

        # Sinusoidal data with noise
        self.X = self.X_no_noise + torch.randn(self.X_no_noise.shape) * input_noise
        self.y = torch.sin(self.X) + torch.randn(self.X.shape) * output_noise + output_shift

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def plot(self):
        plt.plot(self.X, self.y, "o", label="Noisy Training Data")
        plt.plot(self.X_no_noise, self.y_no_noise, label="Baseline Sinusoid")
        plt.legend()


# %%
sinusoid_dataset = SinusoidDataset(N, input_range, input_shift, input_noise, output_shift, output_noise)
sinusoid_dataset.plot()
train_dataset, valid_dataset = random_split(sinusoid_dataset, train_valid_split)


# %% [markdown]
# ## Fully-Connected Neural Network With Linear Output

# %%
class NeuralNetwork(nn.Module):
    def __init__(self, neurons_per_hidden_Layer: list[int], output_bias: bool, act_mod: nn.Module):
        super(NeuralNetwork, self).__init__()

        # Internal function for creating a linear layer with an activation
        def layer(nlm1: int, nl: int) -> nn.Module:
            linear = nn.Linear(nlm1, nl)
            # Optionally group the linear layer with an activation function
            return nn.Sequential(linear, act_mod()) if act_mod else linear

        # Add the input and output layers
        neurons_per_layer = [1] + neurons_per_hidden_Layer + [1]

        # Hidden layers
        hidden_layers = []
        if len(neurons_per_hidden_Layer) > 0:
            hidden_layers = [
                layer(nlminus1, nl)
                for nlminus1, nl in zip(neurons_per_layer[:-2], neurons_per_layer[1:-1])
            ]

        # Output layer
        output_layer = nn.Linear(neurons_per_layer[-2], neurons_per_layer[-1], bias=output_bias)

        # Group all layers into the sequential container
        all_layers = hidden_layers + [output_layer]
        self.layers = nn.Sequential(*all_layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


def initialize_parameters(layer: nn.Module):
    # Ignore any non-parameter layers (e.g., activation functions)
    if type(layer) == nn.Linear:
        print("Initializing", layer)

        with torch.no_grad():

            # We can always initialize bias to zero
            if layer.bias is not None:
                layer.bias.fill_(0.0)

            # Initialize the weights using the specified method
            match initialization_method:
                case InitMethod.Zeros:
                    layer.weight.fill_(0.0)

                case InitMethod.Ones:
                    layer.weight.fill_(1.0)

                case InitMethod.Large:
                    layer.weight.set_(torch.rand_like(layer.weight) * 10.0)

                case InitMethod.Uniform:
                    layer.weight.uniform_()

                case InitMethod.Normal:
                    layer.weight.normal_()

                case InitMethod.Normal2:
                    fan_out = torch.sqrt(torch.tensor(layer.weight.shape[0]))
                    layer.weight.normal_() * (1 / fan_out)

                case InitMethod.Xavier:
                    nn.init.xavier_uniform_(layer.weight)

                case InitMethod.Kaiming:
                    nn.init.kaiming_normal_(layer.weight)
                
                case _:
                    print(f"'{initialization_method}' is not a valid initialization method")


def report_mean_stdev(tensor: torch.Tensor, label: str, indent="  "):
    std, mean = torch.std_mean(tensor)
    print(f"{indent}{label} Mean  = {mean.item():.3f}")
    print(f"{indent}{label} Stdev = {std.item():.3f}\n")


def report_layer_info(l: int, layer: nn.Module, A=None):
    print(f"---------------- Layer {l} ---------------")
    print(layer, "\n")
    if A is not None:
        report_mean_stdev(A, "Layer Input")

    if type(layer) == nn.Sequential or type(layer) == nn.Linear:
        W = layer.weight if type(layer) == nn.Linear else layer[0].weight
        b = layer.bias if type(layer) == nn.Linear else layer[0].bias
        report_mean_stdev(W, "Weights")
        if b is not None:
            report_mean_stdev(b, "Bias")
        if W.grad is not None:
            report_mean_stdev(W.grad.abs(), "Weights gradient")
        if b is not None and b.grad is not None:
            report_mean_stdev(b.grad.abs(), "Bias gradient")
    print()


# %% [markdown]
# ## Model Creation

# %%
# Create the model
model = NeuralNetwork(neurons_per_hidden_layer, output_bias, activation_module)
summary(model)

# Use a custom initializer for layer parameters
print()
model.apply(initialize_parameters)

# Report on initial parameter values
print()
for l, layer in enumerate(model.layers):
    report_layer_info(l + 1, layer)

# %% [markdown]
# ## Training Loop

# %%
# Train the model and report on the final loss value
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Create the training and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

train_losses = []
valid_losses = []

for epoch in progress_bar(range(num_epochs)):

    model.train()
    for X, y in train_loader:
        yhat = model(X)

        loss = criterion(yhat, y)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        for X, y in valid_loader:
            yhat = model(X)
            loss = criterion(yhat, y)
            valid_losses.append(loss.item())

print(f"Final loss: {train_losses[-1]:.6f}")

# Plot training results
_, (ax1, ax2) = plt.subplots(2, 1)

# Plot model predictions
yhat = model(sinusoid_dataset.X)
ax1.plot(sinusoid_dataset.X, sinusoid_dataset.y, "o", label="Target")
ax1.plot(sinusoid_dataset.X, yhat.detach(), "o", label="Prediction")
ax1.legend()

# Plot training and validation losses
ax2.plot(torch.linspace(1, num_epochs, len(train_losses)), train_losses, label="Training")
ax2.plot(torch.linspace(1, num_epochs, len(valid_losses)), valid_losses, label="Validation")
ax2.legend();


# %% [markdown]
# ## Examine Hidden Calculations
#
# Let's look at the outputs of the neurons that feed into the output neuron (this will just be in the inputs if you create a single neuron model).

# %%
# A "hook" so that we can save the hidden values
def capture_layer_input(module: nn.Module, layer_in, layer_out) -> None:
    global final_layer_input
    final_layer_input = layer_in[0].detach()


# Register hook to capture input to final layer
final_layer_input = None
final_layer = model.layers[-1]
final_layer.register_forward_hook(capture_layer_input)

# Grab parameters for the final layer
WL = final_layer.weight.detach()
bL = final_layer.bias.item() if final_layer.bias is not None else 0.0

# Plot the baseline sinusoid
plt.plot(sinusoid_dataset.X_no_noise, sinusoid_dataset.y_no_noise, "--", label="Baseline Sinusoid")

# Plot the output of the final layer
yhat = model(sinusoid_dataset.X_no_noise)  # Activate hook
plt.plot(sinusoid_dataset.X_no_noise, yhat.detach(), "o", label="Prediction")

# Compare with hand-computed final layer output
manually_computed_output = final_layer_input @ WL.T + bL
plt.plot(sinusoid_dataset.X_no_noise, manually_computed_output, "o", markersize=5, label="Combined Activations")

# Plot each input to the final layer
individual_activations = final_layer_input * WL
plt.plot(sinusoid_dataset.X_no_noise, individual_activations, label="Individual Activations")

_ = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# %% [markdown]
# ## Examine Layer Inputs, Parameters, and Gradients
#
# We can get a good idea for inter-layer activations and gradients by printing them off.

# %%
with torch.no_grad():
    # Let's call the input to each layer (including the first layer) "A"
    A = X

    # Print the mean and standard deviation for the input (aim for mean 0 and stdev 1)
    report_layer_info(0, "Input Features", A)

    # Do the same for each layer
    for l, layer in enumerate(model.layers):
        A = layer(A)
        report_layer_info(l + 1, layer, A)

# %%
