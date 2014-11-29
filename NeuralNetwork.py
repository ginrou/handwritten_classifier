#!/usr/bin/env python

from math import exp
import numpy, pprint
pp = pprint.PrettyPrinter(indent=4)

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

class Node:
    def __init__(self, w):
        assert isinstance(w, numpy.ndarray), "w should be numpy.array"
        assert len(w.shape) == 1 , "w should be vector"
        self.w = w

    def activate(self, x):
        assert x.shape == self.w.shape, "input should be same shape with w"
        return sigmoid(numpy.dot(self.w, x))

class Layer:
    def __init__(self, *argv):
        if len(argv) == 2:
            self.__init_with_size(argv[0], argv[1])
        else:
            self.__init_with_values(argv[0])

    def __init_with_size(self, in_size, out_size):
        self.nodes = [ Node( 0.1 * (numpy.random.rand(in_size) -0.5) ) for i in range(out_size) ]

    def __init_with_values(self, w_list):
        self.nodes = [ Node(w) for w in w_list ]

    def fire(self, x):
        return [node.activate(x) for node in self.nodes ]

class NeuralNetwork:
    def __init__(self, shape):
        assert len(shape) > 1, "layer of network should be more than 1"
        self.layers = []
        for i in range(len(shape)-1):
            self.layers.append(Layer(shape[i]+1, shape[i+1]))

    def fire(self, x):
        for layer in self.layers:
            x = layer.fire(numpy.r_[ numpy.array([1]), x ])
        return x

    def predicate(self, x):
        y = self.fire(x)
        return numpy.array(y).argmax()
