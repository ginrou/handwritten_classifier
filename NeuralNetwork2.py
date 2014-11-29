#!/usr/bin/env python

from math import exp
import numpy

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def sigmoid_a(array):
    return numpy.vectorize(sigmoid)(array)

class NeuralNetwork:
    def __init__(self, in_size, hidden_size, out_size):
        self.hidden_weight = 0.1 * (numpy.random.random_sample((hidden_size, in_size+1)) - 0.5)
        self.output_weight = 0.1 * (numpy.random.random_sample((out_size, hidden_size+1)) - 0.5)

    def fit(self, x, t, update_ratio = 0.1):
        z, y = self.fire(x)
        dy = ( y - t ) *y * ( 1 - y )
        dz = (self.output_weight.T.dot(dy))[1:] * z * ( 1- z )

        output_input = numpy.r_[ numpy.array([1]), z ]
        self.output_weight -= update_ratio * dy.reshape(-1,1) * output_input

        hidden_input = numpy.r_[ numpy.array([1]), x ]
        self.hidden_weight -= update_ratio * dz.reshape(-1,1) * hidden_input

    def fire(self, x):
        z = sigmoid_a(self.hidden_weight.dot(numpy.r_[ numpy.array([1]), x ]))
        y = sigmoid_a(self.output_weight.dot(numpy.r_[ numpy.array([1]), z ]))
        return (z, y)

    def predicate(self, x):
        z, y = self.fire(x)
        return numpy.array(y).argmax()
