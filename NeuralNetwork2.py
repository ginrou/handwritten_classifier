#!/usr/bin/env python

from math import exp
import numpy

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def sigmoid_a(array):
    return numpy.vectorize(sigmoid)(array)

class NeuralNetwork:
    def __init__(self, in_size, hidden_size, out_size):
        self.hidden_weight = 0.1 * numpy.random.random_sample((hidden_size, in_size+1)) - 0.05
        self.output_weight = 0.1 * numpy.random.random_sample((out_size, hidden_size+1)) - 0.05

    def fit(self, in_sig, out_sig, update_ratio = 0.1):
        z_out, y_out = self.fire(in_sig)

        delta_y = y_out * ( 1 - y_out )
        tmp = (self.output_weight.T.dot(delta_y))[1:]
        delta_z = tmp * z_out * ( 1- z_out )

        output_input = numpy.r_[ numpy.array([1]), z_out ]
        self.output_weight -= update_ratio * output_input * (delta_y*(y_out-out_sig)).reshape(-1,1)

        hidden_input = numpy.r_[ numpy.array([1]), in_sig ]
        self.hidden_weight -= update_ratio * delta_z.reshape(-1,1) * hidden_input

    def fire(self, x):
        z_in = self.hidden_weight.dot(numpy.r_[ numpy.array([1]), x ])
        z_out = sigmoid_a(z_in)
        y_in = self.output_weight.dot(numpy.r_[ numpy.array([1]), z_out ])
        y_out = sigmoid_a(y_in)
        return (z_out, y_out)

    def predicate(self, x):
        z, y = self.fire(x)
        return numpy.array(y).argmax()
