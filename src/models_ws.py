"""
models.py

This module defines a simple neural network model using PyTorch. It includes the definition of a 
basic feedforward neural network class, `SimpleNet`, and a utility function, `create_model`, to 
instantiate the model with specified dimensions.

External Dependencies:
- torch: PyTorch library for tensor computations and neural networks.
- torch.nn: Submodule of PyTorch for building neural network layers.
- torch.nn.functional: Submodule of PyTorch providing functional interfaces for neural network layers.

Classes:
- SimpleNet: A simple feedforward neural network with one hidden layer.

Functions:
- create_model: Utility function to create an instance of SimpleNet with default or specified dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer transforming input to hidden layer.
        fc2 (nn.Linear): The second fully connected layer transforming hidden layer to output.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the SimpleNet model with specified layer sizes.

        Parameters:
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer.
        """
        super(SimpleNet, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Parameters:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        # Apply the first layer and ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the second layer
        x = self.fc2(x)
        return x

def create_model(input_size=10, hidden_size=5, output_size=1):
    """
    Utility function to create a SimpleNet model with specified or default dimensions.

    Parameters:
        input_size (int, optional): The size of the input layer. Defaults to 10.
        hidden_size (int, optional): The size of the hidden layer. Defaults to 5.
        output_size (int, optional): The size of the output layer. Defaults to 1.

    Returns:
        SimpleNet: An instance of the SimpleNet model.
    """
    # Instantiate the SimpleNet model with given dimensions
    model = SimpleNet(input_size, hidden_size, output_size)
    return model