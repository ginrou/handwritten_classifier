#!/usr/bin/env python

import numpy, sys, os, pprint, gzip
import pickle
from TrainData import TrainData
from NeuralNetwork2 import *
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

def train():
    nn = NeuralNetwork(in_size = 784, hidden_size = 300, out_size = 10)

    images, labels, length = TrainData.read(sys.argv[1], dataset = "testing")

    for i in range(length):
        x = images[i]
        y = TrainData.to_formatted_array(labels[i])

        ## update
        nn.fit(x, y)

        if i % 100 == 0:
            print("{0:05d}/{1:05d}".format(i, length))

    return nn

def evaluate(nn):
    images, labels, length = TrainData.read(sys.argv[1], dataset = "testing")

    ok = 0
    for i in range(length):
        x = images[i]
        y = labels[i]

        if int(y) == int(nn.predicate(x)):
            ok += 1

    print("{0:05d} / {1:05d} = {2:3.2f}%".format(ok, length, 100*ok/length))


if __name__ == "__main__":
    nn = train()
    nn.save("mat.npz")

    nn2 = NeuralNetwork(in_size = 784, hidden_size = 300, out_size = 10)
    nn2.load("mat.npz")
    evaluate(nn2)
