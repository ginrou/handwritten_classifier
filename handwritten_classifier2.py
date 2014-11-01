#!/usr/bin/env python

import numpy, sys, os, pprint
from TrainData import TrainData
from NeuralNetwork2 import *
import matplotlib.pyplot as plt
pp = pprint.PrettyPrinter(indent=4)

def main():
    in_size = 784 ## 28*28
    mid_size = 300
    out_size = 10
    nn = NeuralNetwork(in_size = 784, hidden_size = 300, out_size = 10)

    ittr = 0
    results = []

    data_dir = sys.argv[1]
    for file in os.listdir(data_dir):
        if file.endswith("png"):
            ittr += 1
            train_data = TrainData.read_from_file(os.path.join(data_dir, file))
            x = train_data.image.flatten() / 256.0
            y = train_data.list_formatted_number()

            ## update
            nn.fit(x, y)

            ## inspect
            t = int(train_data.number)
            py = int(nn.predicate(x))
            results.append(1 if t == py else 0)
            r = 100.0 * float(sum(results)) / float(len(results))
            print("{0:05d},{1:3.2f}".format(ittr, r))
            if len(results) >= 100:
                results.pop(0)


if __name__ == "__main__":
    main()
