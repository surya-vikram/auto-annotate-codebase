"""
main.py

This script demonstrates basic operations using numpy and PyTorch. It performs numerical
computations on a numpy array and utilizes a simple neural network model created with PyTorch.

Key Functionalities:
- Generates a random numpy array and calculates its mean.
- Normalizes the numpy array to a 0-1 range.
- Creates and utilizes a simple feedforward neural network model.

External Dependencies:
- numpy: For numerical operations on arrays.
- torch: For creating and operating on neural network models.
- utils (calculate_mean, normalize_array): Utility functions for array operations.
- models (create_model): Function to instantiate a neural network model.
"""

import numpy as np
import torch

from utils import calculate_mean, normalize_array
from models import create_model

def main():
    """
    Main function to execute the script's core functionalities.

    Steps:
    1. Generate a random numpy array and calculate its mean.
    2. Normalize the numpy array and display the first 5 values.
    3. Create a PyTorch model and generate a random input tensor.
    4. Pass the input tensor through the model and print the output.
    """
    # Step 1: Create a random numpy array and calculate its mean
    np_array = np.random.rand(100)  # Generate an array of 100 random numbers between 0 and 1
    mean_value = calculate_mean(np_array)  # Calculate the mean using the imported utility function
    print(f"Mean of the numpy array: {mean_value}")

    # Step 2: Normalize the numpy array and display first 5 values
    normalized_array = normalize_array(np_array)  # Normalize the array to a 0-1 range
    print("First 5 normalized values:", normalized_array[:5])  # Display the first 5 normalized values

    # Step 3: Create a PyTorch model
    input_size = 10
    hidden_size = 5
    output_size = 1
    model = create_model(input_size, hidden_size, output_size)  # Instantiate the model using the imported function

    # Step 4: Generate a random input tensor with a batch size of 3
    input_tensor = torch.randn(3, input_size)  # Create a random tensor with shape (3, input_size)
    output_tensor = model(input_tensor)  # Pass the input tensor through the model to get the output
    print("Model output:", output_tensor)  # Print the model's output

if __name__ == "__main__":
    main()