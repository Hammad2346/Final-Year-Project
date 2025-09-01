import json
import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias
        self.alpha = 0.01
        self.neuronValue=None

    def activation(self, x):
        return np.maximum(self.alpha * x, x)

    def calculate(self, input):
        input = np.array(input)
        self.neuronValue=self.activation(np.dot(self.weights, input) + self.bias)
        return self.neuronValue

class Outputneuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def calculate(self, inputs):
        inputs = np.array(inputs)
        self.unactivated_sum = np.dot(self.weights, inputs) + self.bias
        return self.unactivated_sum

class Hiddenlayers:
    def __init__(self, hiddenLayers):
        self.hiddenLayers = hiddenLayers

    def forwordProp(self, input):
        for layer in self.hiddenLayers:
            input = np.array([neuron.calculate(input) for neuron in layer])
        return input

class Outputlayer:
    def __init__(self, layer):
        self.layer = layer

    def calculate(self, input):
        logits = np.array([neuron.calculate(input) for neuron in self.layer])
        exp_x = np.exp(logits - np.max(logits))
        return exp_x / np.sum(exp_x)

class Network:
    def __init__(self):
        self.hiddenLayers, self.outputLayer = self.initialize()

    def initialize(self):
        with open("NeuralNetwork/emnist_model_improved_augmented.json", 'r') as file:
            network = json.load(file)

        hiddenLayersWeights = network["hidden_layer_weights"]
        hiddenLayersBiases = network["hidden_layer_biases"]
        hiddenLayers = []

        for i in range(len(hiddenLayersWeights)):
            hiddenLayerNeuronWeights = hiddenLayersWeights[i]
            hiddenLayerBiases = hiddenLayersBiases[i]
            layer = [Neuron(hiddenLayerNeuronWeights[j], hiddenLayerBiases[j]) for j in range(len(hiddenLayerNeuronWeights))]
            hiddenLayers.append(layer)

        outputLayerWeights = network["output_layer_weights"]
        outputLayerBiases = network["output_layer_biases"]
        outputLayer = [Outputneuron(outputLayerWeights[i], outputLayerBiases[i]) for i in range(len(outputLayerWeights))]

        return Hiddenlayers(hiddenLayers), Outputlayer(outputLayer)

    def predict(self, input):
        hiddenOutput = self.hiddenLayers.forwordProp(input)
        prediction = self.outputLayer.calculate(hiddenOutput)
        return prediction


