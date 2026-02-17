# from __future__ import annotations
import numpy as np
from read_mnist import MnistDataloader
from pathlib import Path
# import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import List, Tuple

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

class NeuralNetwork:
    def __init__(self, input_size:int=784, hidden_layers:List[int]=[64, 64], output_size:int=10):
        """
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
        
        # np.random.seed(0) # for testing
        
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
    
    def relu_deriv(self, dA: FloatArray, Z:FloatArray) -> FloatArray:
        # ReLU'(Z) = 1 if Z > 0, or
        #            0 if Z <= 0
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ

    def softmax(self, output_activations: FloatArray) -> FloatArray:
        """Converts output logits into a probability distribution for interpretation.

        Args:
            output_activations (FloatArray): Final activation values for the final layer of neurons.

        Returns:
            FloatArray: Probability distribution describing certainty of classification. Sums to 1. 
                E.g for one sample if output[5] == 0.9, the network is 90% certain that the input digit was a 5. 
        """
        shifted = output_activations - np.max(output_activations, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
    def feed_forward(self, inputs: FloatArray) -> FloatArray:
        """
        Forward feed section of neural network loop. 
        Feeds input values in and propagates through the layers, returning the final output. 

        Args:
            inputs (FloatArray): Array of network inputs. Each row is a separate sample (n x 784 matrix)

        Returns:
            FloatArray: Output probability distributions describing digit classification as per softmax(). (n x 10 matrix)
        """
        layers = [inputs] # list of neuron activation values (after relu), aka As
        pre_acts = [] # pre-activations (before relu), aka Zs
        
        for i in range(len(self.weights)):
            # stepping it out for process clarity 
            z = np.matmul(layers[-1], self.weights[i])+self.biases[i]
            pre_acts.append(z)
            
            # dont ReLU the final layer
            if i < len(self.weights) - 1:
                activations = self.relu(z)
            else:
                activations = z
                
            layers.append(activations)
            
        cache = {"activations": layers, "pre-activations": pre_acts}

        return self.softmax(layers[-1]), cache
    
    def batch_cross_entropy(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Calculates the categorical cross entropy loss function for sample batches.
        
        Note: this function is not actually used for model training as the log and
        exponentiation cancel out when combining softmax with cross entropy giving:
        dZ = y_pred - y_true
        
        Args:
            y_pred (FloatArray): Predicted digit classification (probability distribution)
            y_true (FloatArray): Actual digit classification (one-hot array as per one_hot() below)

        Returns:
            loss (float): Categorical cross-entropy loss per sample
        """
        loss = float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))
        return loss

    def backprop(self, y_pred: FloatArray, y_true: FloatArray, cache) -> Tuple[FloatArray, FloatArray]:
        """
        Calculates loss gradient parameters from output to input layers. 
        Returns the partial derivatives of the loss function wrt the weights
        and biases. 

        Args:
            y_pred (FloatArray): Predicted probability distribution
            y_true (FloatArray): Actual probability distribution (one-hot vectors)
            cache (Dict): Stores neuron activation and pre-activation values

        Returns:
            dW (FloatArray): Partial derivative of loss function wrt weights
            db (FloatArray): Partial derivative of loss function wrt biases
        """
        acts = cache["activations"]
        pre_acts = cache["pre-activations"]
        N = y_pred.shape[0]
        
        # note: some notation taken from literature: W = weights, b = bias, 
        # Z = pre-activation value, A = activation value, N = batch_size
        
        dW = [None] * len(self.weights)
        db = [None] * len(self.weights)
        
        # output layer: softmax & cross-entropy gradient
        dZ = (y_pred - y_true) / N # (N x 10)

        for i in reversed(range(len(self.weights))):
            A_prev = acts[i] # (N x layer_in)
            dW[i] = np.matmul(A_prev.T, dZ) # (layer_in x layer_out)
            db[i] = np.sum(dZ, axis=0, keepdims=True)
            
            if i > 0:
                dA_prev = np.matmul(dZ, self.weights[i].T) # (N x layer_in)
                dZ = self.relu_deriv(dA_prev, pre_acts[i-1])
        return dW, db
    
    def sgd_step(self, dW: FloatArray, db: FloatArray, lr: float):
        """
        Stochastic gradient descent step. Adjusts weights and biases to minimise
        loss function. 

        Args:
            dW (FloatArray): partial derivative of loss function wrt weights
            db (FloatArray): partial derivative of loss function wrt biases
            lr (float): learning rate to control step size
        """
        for i in range(len(self.weights)):
            self.weights[i] -= lr * dW[i]
            self.biases[i] -= lr * db[i]

def one_hot(y:IntArray, num_classes: int=10) -> FloatArray:
    """
    Cross-entropy expects a one-hot target.
    
    Args:
        y (int): The correct/desired digit class (0-9)

    Returns:
        FloatArray: One-hot encoded array where the y-th element is a 1. 
    """
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def accuracy(y_pred: FloatArray, y: IntArray) -> float:
    """
    Calculates accuracy of the model based on dataset. 

    Args:
        y_pred (FloatArray): Predicted probability distribution
        y (IntArray): Actual probability distribution

    Returns:
        float: Accuracy
    """
    return float(np.mean(np.argmax(y_pred, axis=1) == y))

def train(network: NeuralNetwork, X: FloatArray, y:IntArray,
          epochs: int=10, batch_size: int=100, lr: float=0.1):
    """
    Trains the neural network on some batch dataset. 

    Args:
        network (NeuralNetwork): The neural network to train
        X (FloatArray): Input dataset (pixel values of hand-drawn digits)
        y (IntArray): Data labels
        epochs (int, optional): Number of times the original dataset is iterated over during training. Defaults to 10.
        batch_size (int, optional): Computationally intensive to calculate loss gradient over entire dataset per sgd step, so we use smaller batches instead. Defaults to 100.
        lr (float, optional): Learning rate. Defaults to 0.1.
    """
    # normalise input values to [0, 1]
    X = X.astype(np.float64) / 255.0
    X = X.reshape(X.shape[0], -1) # (N, 784)
    y_oh = one_hot(y, 10)
    
    N = X.shape[0] # num training samples
    
    for epoch in range(epochs):
        idx = np.random.permutation(N) # shuffle training data
        Xs, ys, yohs = X[idx], y[idx], y_oh[idx]
        
        epoch_loss = 0.0
        
        for start in range(0, N, batch_size):
            end = start + batch_size
            X_batch = Xs[start:end]
            y_batch = ys[start:end]
            y_oh_batch = yohs[start:end]
            
            y_pred, cache = network.feed_forward(X_batch)
            # loss = network.batch_cross_entropy(y_pred, y_oh_batch)
            dW, db = network.backprop(y_pred, y_oh_batch, cache)
            network.sgd_step(dW, db, lr)
            
            # epoch_loss += loss * X_batch.shape[0]
        
        # epoch_loss /= N
        
        # quick eval on training subset
        y_pred_train, _ = network.feed_forward(X[:2000])
        acc = accuracy(y_pred_train, y[:2000])
        
        print(f"training epoch {epoch+1}/{epochs}    train_acc~={acc:.3f}")

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
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # initialise neural net and train
    network = NeuralNetwork()
    train(network, x_train, y_train)
    
    # run testing dataset through neural net and print accuracy
    y_pred_test, _ = network.feed_forward(x_test.reshape(x_test.shape[0], -1).astype(np.float64)/255.0)
    print(f"testing set             train_acc~={accuracy(y_pred_test, y_test):.3f}")
    
if __name__ == '__main__':
    main()
    