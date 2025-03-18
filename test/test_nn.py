# TODO: import dependencies and write unit tests below

import numpy as np
from nn.nn import NeuralNetwork, binary_cross_entropy, binary_cross_entropy_backprop, mean_squared_error, mean_squared_error_backprop
from nn.preprocess import sample_seqs
from nn.preprocess import one_hot_encode_seqs

def test_single_forward():
    # Test the forward pass of a single layer neural network
    nn = NeuralNetwork(4, 1)
    nn.weights = np.array([[1], [2], [3], [4]])
    nn.biases = np.array([1])
    x = np.array([1, 2, 3, 4])
    y = nn.forward(x)
    assert y == 31

def test_forward():
    # Test the forward pass of a multi-layer neural network
    nn = NeuralNetwork(4, 1)
    nn.weights = np.array([[1], [2], [3], [4]])
    nn.biases = np.array([1])
    x = np.array([1, 2, 3, 4])
    y = nn.forward(x)
    assert y == 31

def test_single_backprop():
    # Test the backward pass of a single
    nn = NeuralNetwork(4, 1)
    nn.weights = np.array([[1], [2], [3], [4]])
    nn.biases = np.array([1])
    x = np.array([1, 2, 3, 4])
    y = nn.forward(x)
    dy = 1
    dx = nn.backprop(dy)
    assert isinstance(dx, np.ndarray)

def test_predict():
    # Test the predict function of the neural network
    nn = NeuralNetwork(4, 1)
    nn.weights = np.array([[1], [2], [3], [4]])
    nn.biases = np.array([1])
    x = np.array([1, 2, 3, 4])
    y = nn.predict(x)
    assert y == 31  

def test_binary_cross_entropy():
    # Test the binary cross entropy loss function
    y_true = np.array([1])
    y_pred = np.array([0.5])
    loss = binary_cross_entropy(y_true, y_pred)
    assert np.isclose(loss, 0.6931471805599453)

def test_binary_cross_entropy_backprop():
    # Test the binary cross entropy loss function
    y_true = np.array([1])
    y_pred = np.array([0.5])
    dy = binary_cross_entropy_backprop(y_true, y_pred)
    assert np.isclose(dy, -2.0)

def test_mean_squared_error():
    # Test the mean squared error loss function
    y_true = np.array([1])
    y_pred = np.array([0.5])
    loss = mean_squared_error(y_true, y_pred)
    assert np.isclose(loss, 0.25)

def test_mean_squared_error_backprop():
    # Test the mean squared error loss function
    y_true = np.array([1])
    y_pred = np.array([0.5])
    dy = mean_squared_error_backprop(y_true, y_pred)
    assert np.isclose(dy, -1.0)

def test_sample_seqs():
    # Test the sampling function
    seqs = ['AAAA', 'TTTT', 'CCCC', 'GGGG', 'ATAT', 'TATA', 'ACAC', 'GTGT']
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    sample_seqs_out, sample_labels_out = sample_seqs(seqs, labels)
    assert len(sample_seqs) == len(sample_labels_out)
    assert len(sample_seqs) == 4
    assert sum(sample_labels_out) == 2

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
