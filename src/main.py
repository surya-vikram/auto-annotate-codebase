import numpy as np
import torch

from utils import calculate_mean, normalize_array
from models import create_model


def main():
    # Create a random numpy array and calculate its mean
    np_array = np.random.rand(100)
    mean_value = calculate_mean(np_array)
    print(f"Mean of the numpy array: {mean_value}")

    # Normalize the numpy array and display first 5 values
    normalized_array = normalize_array(np_array)
    print("First 5 normalized values:", normalized_array[:5])

    # Create a PyTorch model
    input_size = 10
    hidden_size = 5
    output_size = 1
    model = create_model(input_size, hidden_size, output_size)

    # Generate a random input tensor with a batch size of 3
    input_tensor = torch.randn(3, input_size)
    output_tensor = model(input_tensor)
    print("Model output:", output_tensor)


if __name__ == "__main__":
    main()
