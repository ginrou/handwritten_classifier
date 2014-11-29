#!/usr/bin/env python

import numpy, pprint
pp = pprint.PrettyPrinter(indent=4)

from NeuralNetwork import *
from TrainData import TrainData

class NeuralNetworkTrainer:
    def __init__(self, neural_network):
        self.nn = neural_network

    def train(self, in_sig, out_sig):
        z_in, z_out, y_in, y_out = self._get_layer_outputs(in_sig)
        assert z_in.shape == (300, ), "size incorrect"
        assert z_out.shape == (300, ), "size incorrect"
        assert y_in.shape == (10, ), "size incorrect"
        assert y_out.shape == (10, ), "size incorrect"

        delta_y = self._get_delta_y(y_out, out_sig)
        assert delta_y.shape == (10, ), "size incorrect"

        delta_z = self._get_delta_z(delta_y, z_out)
        assert delta_z.shape == (300, ), "size incorrect"

        for l in range(len(self.nn.layers[1].nodes)):
            self.nn.layers[1].nodes[l].w -= 0.1 * delta_y[l] * numpy.r_[ numpy.array([1]), z_out ]

        for m in range(len(self.nn.layers[0].nodes)):
            self.nn.layers[0].nodes[m].w -= 0.1 * delta_z[m] * numpy.r_[ numpy.array([1]), in_sig ]

    def _get_layer_outputs(self, in_sig):
        x0 = numpy.r_[ numpy.array([1]), in_sig ]
        z_in = numpy.array([ numpy.dot(x0, node.w) for node in self.nn.layers[0].nodes ])
        z_out = numpy.array([ sigmoid(z) for z in z_in ])
        x1 = numpy.r_[ numpy.array([1]), z_out ]
        y_in = numpy.array([ numpy.dot(x1, node.w) for node in self.nn.layers[1].nodes ])
        y_out = numpy.array([ sigmoid(y) for y in y_in ])

        return (z_in, z_out, y_in, y_out)

    def _get_delta_y(self, y_out, t):
        return ( y_out - t) * y_out * ( 1 - y_out )

    def _get_delta_z(self, delta_y, z_out):
        L = len(self.nn.layers[1].nodes)
        M = len(self.nn.layers[1].nodes[0].w)
        w = [ numpy.zeros(L) for i in range(M) ]
        assert len(w) == 301, "size incorrect"
        assert w[0].shape[0] == 10, "size incorrect"

        for l in range(L):
            for m in range(M):
                w[m][l] = self.nn.layers[1].nodes[l].w[m]

        d = numpy.array([ numpy.dot(delta_y, w_) for w_ in w ])[1:]
        assert d.shape == (M-1, ) , "size incorrect"
        return d * (1-z_out) * z_out
