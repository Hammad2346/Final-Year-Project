import numpy as np
import pandas as pd
import json
from tqdm import tqdm  

sizesOfLayers=[784,128,64,10]

class Neuron:
    def __init__(self, numberOfInputs):
        self.weights = np.random.randn(numberOfInputs) * np.sqrt(1.0 / numberOfInputs)
        self.bias = np.random.randn() * np.sqrt(1.0 / numberOfInputs)
        self.input = None

        self.neuronValue = 0
        self.unactivated_sum = 0

        self.delta = 0
        self.alpha = 0.01

    def calculate(self, inputs):
        self.input = inputs
        self.unactivated_sum = self.weights.dot(inputs) + self.bias
        self.neuronValue = self.activation(self.unactivated_sum)
        return self.neuronValue

    def activation(self, x):
        return np.maximum(self.alpha * x, x)  

    def derivative(self):
        return 1.0 if self.unactivated_sum > 0 else self.alpha

class HiddenLayer:
    def __init__(self, inputSize, outputSize):
        self.neurons = [Neuron(inputSize) for _ in range(outputSize)]

    def getValues(self, inputs):
        return np.array([neuron.calculate(inputs) for neuron in self.neurons])

class HiddenLayers:
    def __init__(self, sizes):
        self.hiddenlayers = []
        for i in range(1, len(sizes)):
            self.hiddenlayers.append(HiddenLayer(sizes[i - 1], sizes[i]))

    def forwardProp(self, inputs):
        for layer in self.hiddenlayers:
            inputs = layer.getValues(inputs)
        return inputs

    def backPropagate(self, output_error, learn_rate):
        next_layer_error = output_error  
        for layer in reversed(self.hiddenlayers):
            for i, neuron in enumerate(layer.neurons):
                gradient = next_layer_error[i] * neuron.derivative()
                neuron.delta = gradient  

            prev_layer_size = len(layer.neurons[0].weights)
            new_error = np.zeros(prev_layer_size)

            for neuron in layer.neurons:
                for j in range(prev_layer_size):
                    new_error[j] += neuron.weights[j] * neuron.delta

            for neuron in layer.neurons:
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= learn_rate * neuron.delta * neuron.input[j]
                neuron.bias -= learn_rate * neuron.delta
            next_layer_error = new_error


class OutputLayer:
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(outputSize, inputSize) * np.sqrt(1.0 / inputSize)
        self.biases = np.random.randn(outputSize) * np.sqrt(1.0 / inputSize)
        self.output = np.zeros(outputSize)
        self.unactivated_sum = np.zeros(outputSize)
        self.input = None

    def calculate(self, inputs):
        self.input = inputs
        self.unactivated_sum = self.weights.dot(inputs) + self.biases
        self.output = self.softmax(self.unactivated_sum)
        return self.output

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def backPropagate(self, output_error, learn_rate):
        self.delta = output_error
        self.weights -= learn_rate * np.outer(self.delta, self.input)
        self.biases -= learn_rate * self.delta
        return np.dot(self.weights.T, output_error)

class NeuralNetwork:
    def __init__(self, sizes):
        self.inputSize = sizes[0]
        self.hiddenLayers = HiddenLayers(sizes[:-1])
        self.outputLayer = OutputLayer(sizes[-2], sizes[-1])

    def predict(self, pixels):
        hidden_output = self.hiddenLayers.forwardProp(pixels)
        return self.outputLayer.calculate(hidden_output)

    def oneHot(self, label):
        zeroArray = np.zeros(10)
        zeroArray[label] = 1
        return zeroArray

    def calculateCost(self, outputLayerPredictions, label):
        predicted_prob = outputLayerPredictions[label]
        return -np.log(predicted_prob + 1e-15)

    def learn(self, pixels, label, learnRate):
        hidden_output = self.hiddenLayers.forwardProp(pixels)
        output = self.outputLayer.calculate(hidden_output)
        output_error = output - self.oneHot(label)
        hidden_error = self.outputLayer.backPropagate(output_error, learnRate)
        self.hiddenLayers.backPropagate(hidden_error, learnRate)

def save_model(network):
    model_data = {
        'hidden_layer_weights': [[neuron.weights.tolist() for neuron in layer.neurons] for layer in network.hiddenLayers.hiddenlayers],
        'hidden_layer_biases': [[neuron.bias for neuron in layer.neurons] for layer in network.hiddenLayers.hiddenlayers],
        'output_layer_weights': network.outputLayer.weights.tolist(),
        'output_layer_biases': network.outputLayer.biases.tolist()
    }

    with open('temp_model.json', 'w') as f:
        json.dump(model_data, f)
    print("Model saved to 'temp_model.json'")

# Load data
data = pd.read_csv('mnist_train.csv').to_numpy()
lessData = data[:5000]
testData = data[6000:7000]
learnRate = 0.001
iterations = 500

network = NeuralNetwork(sizesOfLayers)

def train(examples, testexamples, learnRate, iterations, accuracy_threshold=0.91):
    for i in range(iterations):
        total_cost = 0
        example_bar = tqdm(examples, desc=f"Epoch {i+1}/{iterations}", unit="example")
        for example in example_bar:
            label = example[0]
            pixels = example[1:] / 255
            prediction = network.predict(pixels)
            cost = network.calculateCost(prediction, label)
            total_cost += cost
            network.learn(pixels, label, learnRate)
        correct_in_iteration = 0
        for testexample in testexamples:
            label = testexample[0]
            pixels = testexample[1:] / 255
            prediction = network.predict(pixels)
            if np.argmax(prediction) == label:
                correct_in_iteration += 1

        accuracy = correct_in_iteration / len(testexamples)
        average_cost = total_cost / len(examples)
        print(f"\nIteration {i} | Accuracy: {accuracy:.4f} | Cost: {average_cost:.4f}")
        if accuracy >= accuracy_threshold:
            print(f"Accuracy threshold reached: {accuracy:.4f}. Saving model...")
            save_model(network)
            break

train(lessData, testData, learnRate, iterations)

