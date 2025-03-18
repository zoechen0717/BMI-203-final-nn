# TODO: import dependencies and write unit tests below

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nn import NeuralNetwork
from nn import binary_cross_entropy
from nn import binary_cross_entropy_backprop
from nn import mean_squared_error
from nn import mean_squared_error_backprop
from nn import sample_seqs
from nn import one_hot_encode_seqs

def test_single_forward():
    # Test the forward pass of a single layer neural network
    nn = NeuralNetwork(4, 1)
    nn.weights = np.array([[1], [2], [3], [4]])
    nn.biases = np.array([1])
    x = np.array([1, 2, 3, 4])
    y = nn.forward(x)
    assert y == 30

def test_forward():
    # Test the forward pass of a multi-layer neural network
    nn = NeuralNetwork(4, 1)
    nn.weights = np.array([[1], [2], [3], [4]])
    nn.biases = np.array([1])
    x = np.array([1, 2, 3, 4])
    y = nn.forward(x)
    assert y == 30

def test_single_backprop():
    # Test the backward pass of a single
    nn = NeuralNetwork(4, 1)
    nn.weights = np.array([[1], [2], [3], [4]])
    nn.biases = np.array([1])
    x = np.array([1, 2, 3, 4])
    y = nn.forward(x)
    dy = 1
    dx = nn.backprop(dy)
    assert dx == 1

def test_predict():
    # Test the predict function of the neural network
    nn = NeuralNetwork(4, 1)
    nn.weights = np.array([[1], [2], [3], [4]])
    nn.biases = np.array([1])
    x = np.array([1, 2, 3, 4])
    y = nn.predict(x)
    assert y == 30  

def test_binary_cross_entropy():
    # Test the binary cross entropy loss function
    y_true = np.array([1])
    y_pred = np.array([0.5])
    loss = binary_cross_entropy(y_true, y_pred)
    assert loss == 0.6931471805599453

def test_binary_cross_entropy_backprop():
    # Test the binary cross entropy loss function
    y_true = np.array([1])
    y_pred = np.array([0.5])
    dy = binary_cross_entropy_backprop(y_true, y_pred)
    assert dy == -2.0

def test_mean_squared_error():
    # Test the mean squared error loss function
    y_true = np.array([1])
    y_pred = np.array([0.5])
    loss = mean_squared_error(y_true, y_pred)
    assert loss == 0.25 

def test_mean_squared_error_backprop():
    # Test the mean squared error loss function
    y_true = np.array([1])
    y_pred = np.array([0.5])
    dy = mean_squared_error_backprop(y_true, y_pred)
    assert dy == -1.0

def test_sample_seqs():
    # Test the sampling function
    seqs = ['
    'AAAA', 'TTTT', 'CCCC', 'GGGG', 'ATAT', 'TATA', 'ACAC', 'GTGT']
    labels = [True, True, True, True, False, False, False, False]
    sample_seqs, sample_labels = sample_seqs(seqs, labels)
    assert len(sample_seqs) == len(sample_labels)
    assert len(sample_seqs) == 4
    assert sum(sample_labels) == 2
    assert sum(sample_labels) == 2

def test_one_hot_encode_seqs():
    # Test the one-hot encoding function
    seqs = ['AAAA', 'TTTT', 'CCCC', 'GGGG']
    encodings = one_hot_encode_seqs(seqs)
    assert len(encodings) == 16
    assert encodings[0] == 1
    assert encodings[1] == 0
    assert encodings[2] == 0
    assert encodings[3] == 0
    assert encodings[4] == 0
    assert encodings[5] == 1
    assert encodings[6] == 0
    assert encodings[7] == 0
    assert encodings[8] == 0
    assert encodings[9] == 0
    assert encodings[10] == 0
    assert encodings[11] == 1
    assert encodings[12] == 0
    assert encodings[13] == 0
    assert encodings[14] == 0
    assert encodings[15] == 1  
