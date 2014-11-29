#!/usr/bin/env python

import numpy, sys, os, pprint
from TrainData import TrainData
from NeuralNetwork import *
from NeuralNetworkTrainer import NeuralNetworkTrainer
import matplotlib.pyplot as plt
pp = pprint.PrettyPrinter(indent=4)

def main():
    in_size = 784 ## 28*28
    mid_size = 300
    out_size = 10
    nn = NeuralNetwork([in_size, mid_size, out_size])
    trainer = NeuralNetworkTrainer(nn)

    ittr = 0
    results = []

    data_dir = sys.argv[1]
    for file in os.listdir(data_dir):
        if file.endswith("png"):
            ittr += 1
            train_data = TrainData.read_from_file(os.path.join(data_dir, file))
            x = train_data.image.flatten() / 256.0
            y = train_data.list_formatted_number()
            trainer.train(x,y)

            results.append(int(train_data.number) == int(nn.predicate(x)))
            r = sum(results) / len(results)
            print ("{0:05d},{1:.2f}".format(ittr, 100* r))
            if len(results) >= 100:
                results.pop(0)


if __name__ == "__main__":
    main()
