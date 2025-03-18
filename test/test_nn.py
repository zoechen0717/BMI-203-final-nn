# TODO: import dependencies and write unit tests below

import numpy as np
import pytest
from nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

# Create a test neural network
def create_test_nn():
    return NeuralNetwork(
        nn_arch=[{'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01, seed=42, batch_size=32, epochs=100, loss_function="binary_cross_entropy"
    )

def create_test_nn_mse():
    return NeuralNetwork(
        nn_arch=[{'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01, seed=42, batch_size=32, epochs=100, loss_function="mean_squared_error"
    )

def test_single_forward():
    nn = create_test_nn()

    # Define test inputs
    W_curr = np.array([[0.5, -0.2, 0.3, 0.1]])  # Weights
    b_curr = np.array([[0.1]])  # Bias
    A_prev = np.array([[1, 2, 3, 4]])  # Input
    activation = "relu"

    # Call _single_forward
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, activation)

    # Expected outputs
    expected_Z = np.array([[1.5]])
    expected_A = np.array([[1.5]])  # Since relu(1.5) = 1.5

    # Assertions to check correctness
    assert np.allclose(Z_curr, expected_Z), f"Expected Z_curr {expected_Z}, but got {Z_curr}"
    assert np.allclose(A_curr, expected_A), f"Expected A_curr {expected_A}, but got {A_curr}"

def test_forward():
    nn = create_test_nn()
    x = np.array([[1, 2, 3, 4]])
    y, cache = nn.forward(x)
    assert isinstance(y, np.ndarray)
    assert isinstance(cache, dict)

def test_single_backprop():
    nn = create_test_nn()
    x = np.array([[1, 2, 3, 4]])
    y, cache = nn.forward(x)
    dA = np.ones_like(y)
    dA_prev, dW, db = nn._single_backprop(nn._param_dict['W1'], nn._param_dict['b1'], cache['Z1'], cache['A0'], dA, 'sigmoid')
    assert isinstance(dA_prev, np.ndarray)
    assert isinstance(dW, np.ndarray)
    assert isinstance(db, np.ndarray)

def test_predict():
    nn = create_test_nn()
    x = np.array([[1, 2, 3, 4]])
    y = nn.predict(x)
    assert isinstance(y, np.ndarray)

def test_binary_cross_entropy():
    nn = create_test_nn()
    y_true = np.array([1])
    y_pred = np.array([0.5])
    loss = nn._binary_cross_entropy(y_true, y_pred)
    assert np.isclose(loss, 0.6931471805599453)

def test_binary_cross_entropy_backprop():
    nn = create_test_nn()
    y_true = np.array([1])
    y_pred = np.array([0.5])
    dy = nn._binary_cross_entropy_backprop(y_true, y_pred)
    assert isinstance(dy, np.ndarray)  # assert that dy is a numpy array

def test_mean_squared_error():
    nn = create_test_nn_mse()
    y_true = np.array([1])
    y_pred = np.array([0.5])
    loss = nn._mean_squared_error(y_true, y_pred)
    assert np.isclose(loss, 0.25)

def test_mean_squared_error_backprop():
    nn = create_test_nn_mse()
    y_true = np.array([[1]])  # Reshaped to (1,1)
    y_pred = np.array([[0.5]])
    dy = nn._mean_squared_error_backprop(y_true, y_pred)
    assert isinstance(dy, np.ndarray)


def test_sample_seqs():
    seqs = ['AAAA', 'TTTT', 'CCCC', 'GGGG', 'ATAT', 'TATA', 'ACAC', 'GTGT']
    labels = [1, 1, 1, 0, 0, 0, 0, 0]
    sample_seqs_out, sample_labels_out = sample_seqs(seqs, labels)
    assert len(sample_seqs_out) == len(sample_labels_out)
    assert len(sample_seqs_out) in [4, 5, 6, 7, 8] # depending on the random sampling
    assert sum(sample_labels_out) == 3

def test_one_hot_encode_seqs():
    # Test the one-hot encoding function
    seqs = ['AAAA', 'TTTT', 'CCCC', 'GGGG']
    encodings = one_hot_encode_seqs(seqs)
    expected = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ]).flatten()
    assert np.array_equal(encodings.flatten(), expected)
