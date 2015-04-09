import matplotlib.pyplot as plt
import random as rnd
import math
import sys

## A simple training set class
class TrainingClass:

  def __init__(self, set, target):
    self.set = set
    self.classification = [False]*len(set)
    self.target = target

  def classify(self, i, out):
    if (out > 0) == (self.target > 0):
      self.classification[i] = True
      return True
    return False

  def pointsCorrectlyClassified(self):
    return reduce(lambda acc, x: acc and x, self.classification, True)

# a collection of training set classes
class TrainingSet:

  def __init__(self, classes):
    self.classes = classes

  # the total training instances in this set
  def totalInstances(self):
    return reduce(lambda acc, clazz : acc+len(clazz.set), self.classes, 0)

  def pointsCorrectlyClassified(self):
    allCorrect = True
    for clazz in self.classes:
      allCorrect = allCorrect and clazz.pointsCorrectlyClassified()
    return allCorrect

# This class represents an activation function
class ActivationFunction:

  #this is the function
  def F(self, net):
    raise NotImplementedException("This is not a concrete activation function instance")

  #this is the derivative of the function
  def dF(self, net):
    raise NotImplementedException("This is not a concrete activation function instance")

# A linear activation function
class LinearActivationFunction(ActivationFunction):

  #just the identity function
  def F(self, x):
    return x

  def dF(self, x):
    return 1

# A sigmoid activation function
# F(x) = 1/(1+e^(-net))
class SigmoidActivationFunction(ActivationFunction):

  #return 1/1-e^net
  def F(self, x):
    return 1.0/(1.0 + math.exp(-x))

  def dF(self, x):
    return x*(1.0 - x)


#this represents a weight
class Weight:
    def __init__(self, fromNeuron, toNeuron):
        self.value = rnd.uniform(-0.1, 0.1)
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron
        fromNeuron.outputWeights.append(self)
        toNeuron.inputWeights.append(self)
        self.delta = 0.0 # delta value, this will accumulate and after each training cycle used to adjust the weight value

    def calculateDelta(self, network):
        self.delta += self.fromNeuron.output * self.toNeuron.error

#this is a neuron
class Neuron:
    def __init__(self):
        self.output = 0.0
        self.target = 0.0
        self.error = 0.0    # error between output and target
        #self.bias = rnd.uniform(-0.1,0.1) #randomly initialized bias
        self.inputWeights = []
        self.outputWeights = []

    def net(self):
      x = 0.0;
      for weight in self.inputWeights:
          x += weight.value * weight.fromNeuron.output
      return x + self.bias

    #why did I decide to pass network here?
    def activate(self, network, activation):
        net = self.net()
        self.output = activation.F(net)



# this is a layer
class Layer:
    def __init__(self, neurons, activation, bias):
        self.neurons = neurons
        self.activation = activation
        self.bias = bias
        for neuron in self.neurons:
          neuron.bias = self.bias

    def activate(self, network):
        for neuron in self.neurons:
            neuron.activate(network, self.activation)


class NeuralNetwork:
    def __init__(self, layers, learningRate, momentum):
        self.layers = layers
        self.learningRate = learningRate # the rate at which the network learns
        self.momentum = momentum
        self.weights = []

        #initialize the weights
        for hiddenNeuron in self.layers[1].neurons:
            for inputNeuron in self.layers[0].neurons:
                self.weights.append(Weight(inputNeuron, hiddenNeuron))
            for outputNeuron in self.layers[2].neurons:
                self.weights.append(Weight(hiddenNeuron, outputNeuron))

    #set a point as the input
    def setInputs(self, inputs):
        self.layers[0].neurons[0].output = float(inputs[0])
        self.layers[0].neurons[1].output = float(inputs[1])

    #set the target
    def setExpectedOutputs(self, expectedOutputs):
        self.layers[2].neurons[0].target = expectedOutputs[0]

    #compute the outputs
    def calculateOutputs(self, expectedOutputs):
        self.setExpectedOutputs(expectedOutputs)
        self.layers[1].activate(self) # activation function for hidden layer
        self.layers[2].activate(self) # activation function for output layer

    def calculateOutputErrors(self):
        for neuron in self.layers[2].neurons:
          #neuron.error = (neuron.target - neuron.output)
          neuron.error = (neuron.target - neuron.output) * self.layers[2].activation.dF(neuron.output)

    def calculateHiddenErrors(self):
        for neuron in self.layers[1].neurons:
            error = 0.0
            for weight in neuron.outputWeights:
                error += weight.toNeuron.error * weight.value
            #neuron.error = error *  neuron.output * (1 - neuron.output)
            neuron.error = error * self.layers[1].activation.dF(neuron.output)

    def calculateDeltas(self):
        for weight in self.weights:
            weight.calculateDelta(self)

    # this is backpropagation step
    def train(self, inputs, expectedOutputs):
        self.setInputs(inputs)
        self.calculateOutputs(expectedOutputs)
        self.calculateOutputErrors()
        self.calculateHiddenErrors()
        self.calculateDeltas()

    # this is weight update
    def learn(self, trainingClassSize):
        for weight in self.weights:
            #do I need to divide or not?
            weight.value += (self.learningRate * weight.delta) + self.momentum*weight.delta #/trainingClassSize
        #compute
        for layer in self.layers:
          for neuron in layer.neurons:
            layer.bias += self.learningRate*(neuron.error*layer.bias)/trainingClassSize
            neuron.bias = layer.bias


    def output(self, inputs):
        self.setInputs(inputs)
        self.layers[1].activate(self)
        self.layers[2].activate(self)
        return self.layers[2].neurons[0].output

#------------------------------ welcome to the show

def main():
  #initialize layers and learning rate
  seed = rnd.randint(0, sys.maxint)
  rnd.seed(seed)
  print seed

  inputLayer = Layer([Neuron() for n in range(2)], LinearActivationFunction(), 0)
  hiddenLayer = Layer([Neuron() for n in range(3)], SigmoidActivationFunction(), rnd.uniform(-0.1,0.1))
  outputLayer = Layer([Neuron() for n in range(1)], LinearActivationFunction(), rnd.uniform(-0.1,0.1))
  learningRate = 1.0/30.0
  # a small momentum could help?
  momentum = 0.25

  #initialize the network
  network = NeuralNetwork([inputLayer, hiddenLayer, outputLayer], learningRate, momentum)

  #initialize the training set
  class1 = [(4,2), (4,4), (5,3), (5,1), (7,2)]
  class2 = [(1,2), (2,1), (3,1), (6,5), (3,6), (6,7), (4,6), (7,6)]
  trainingClass1 = TrainingClass(class1, 1)
  trainingClass2 = TrainingClass(class2, -1)
  trainingSet = TrainingSet([trainingClass1,trainingClass2])

  #epochs = 12000
  #loop for 'a few thousand epochs'
  epochs = 0
  for e in range(0,10000):
  #while(not trainingSet.pointsCorrectlyClassified()): #this condition is not enough
    #for each point
    for clazz in trainingSet.classes:
      for j in range(0, len(clazz.set)):
        point = clazz.set[j]
        out = network.output(point)
        isOk = clazz.classify(j, out)
        if isOk:
         continue
        network.train(point, [clazz.target])
        network.learn(trainingSet.totalInstances())
      for w in network.weights:
        w.delta = 0.0
    epochs += 1

  print "Epochs: ", epochs
  # print "Classification of the training set:"
  #
  # for clazz in trainingSet.classes:
  #   print clazz.classification

  print "Results on the test set:"

  testClass1 = TrainingClass([(4,1), (5,2), (3,4), (5,4), (6,1), (7,1)],1)
  testClass2 = TrainingClass([(3,2), (8,7), (4,7), (7,5), (2,3), (2,5)], -1)
  testSet = TrainingSet([testClass1, testClass2])

  #plot training data
  # for point in trainingClass1.set:
  #   plt.scatter(point[0], point[1], c='r')
  # for point in trainingClass2.set:
  #   plt.scatter(point[0], point[1], c='b')

  #plot test data
  # for point in testClass1.set:
  #   plt.scatter(point[0], point[1], c='r')
  # for point in testClass2.set:
  #   plt.scatter(point[0], point[1], c='b')


  for clazz in testSet.classes:
    for point in clazz.set:
      out = network.output(point)
      print point, " -> ", out, " ", (clazz.target > 0) == (out > 0)

  #plot the points
  plt.show()

if __name__ == "__main__":
    main()
