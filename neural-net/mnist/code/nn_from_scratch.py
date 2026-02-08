# from __future__ import annotations
import numpy as np
from read_mnist import MnistDataloader
from pathlib import Path
# import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import List

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

class NeuralNetwork:
    def __init__(self, input_size:int=784, hidden_layers:List[int]=[32, 32], output_size:int=10):
        """_summary_
        Initialises network weights (randomly) and biases (zero)
        
        Args:
            input_size (int): Number of neurons in the input layer. E.g for MNIST, you will have a neuron for each pixel (784)
            hidden_layers (List[int]): Number of neurons per hidden layer. E.g [16, 16] means two hidden layers of 16 neurons each
            output_size (int): Number of neurons in the output layer. E.g for MNIST you are classifying an image as a digit from 0-9, meaning 10 output neurons.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        
        # input to hidden layers network
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        
        # hidden layer network
        for i in range(len(hidden_layers)-1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
            self.biases.append(np.zeros((1, hidden_layers[i+1])))
        
        # hidden layer to output network
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))
        
    def relu(self, unscaled_vals: FloatArray) -> FloatArray:
        return np.maximum(0, unscaled_vals)

    def softmax(self, output_activations: FloatArray) -> FloatArray:
        shifted = output_activations - np.max(output_activations, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
    def feed_forward(self, inputs: FloatArray) -> FloatArray:
        layers = [inputs]
        
        for i in range(len(self.weights)):
            # stepping it out for process clarity 
            z = np.matmul(layers[-1], self.weights[i])+self.biases[i]
            activations = self.relu(z)
            layers.append(activations)

        return self.softmax(layers[-1])

def main() -> None:
    # data filepaths
    DATA_DIR = Path(__file__).resolve().parents[1] / 'dataset'
    training_images_filepath = DATA_DIR / 'train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = DATA_DIR / 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_filepath = DATA_DIR / 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = DATA_DIR / 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    
    # load data
    data_loader = MnistDataloader(training_images_filepath,
                                  training_labels_filepath,
                                  test_images_filepath,
                                  test_labels_filepath)  
    # x = images, y = labels
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    
    x0 = np.array(x_train[0]).flatten()
    
    network = NeuralNetwork()
    print(network.feed_forward(x0))
    
if __name__ == '__main__':
    main()
    