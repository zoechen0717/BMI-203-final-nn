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
    x = np.array([1, 2, 3, 4])
    y = nn.forward(x)
    assert y.shape == (1,)

def test_binary_cross_entropy():
    nn = create_test_nn()
    y_true = np.array([1])
    y_pred = np.array([0.5])
    loss = nn.binary_cross_entropy(y_true, y_pred)
    assert np.isclose(loss, 0.6931471805599453)

def test_binary_cross_entropy_backprop():
    nn = create_test_nn()
    y_true = np.array([1])
    y_pred = np.array([0.5])
    dy = nn.binary_cross_entropy_backprop(y_true, y_pred)
    assert isinstance(dy, np.ndarray)  # assert that dy is a numpy array

def test_mean_squared_error():
    nn = create_test_nn_mse()
    y_true = np.array([1])
    y_pred = np.array([0.5])
    loss = nn.mean_squared_error(y_true, y_pred)
    assert np.isclose(loss, 0.25)

def test_mean_squared_error_backprop():
    nn = create_test_nn_mse()
    y_true = np.array([1])
    y_pred = np.array([0.5])
    dy = nn.mean_squared_error_backprop(y_true, y_pred)
    assert isinstance(dy, np.ndarray)
    assert dy.shape == y_true.shape
    assert np.isclose(dy[0], -0.5)


def test_sample_seqs():
    seqs = ['AAAA', 'TTTT', 'CCCC', 'GGGG', 'ATAT', 'TATA', 'ACAC', 'GTGT']
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    sample_seqs_out, sample_labels_out = sample_seqs(seqs, labels)
    assert len(sample_seqs_out) == len(sample_labels_out)
    assert len(sample_seqs_out) == 4 # 4 positive and 4 negative samples

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
