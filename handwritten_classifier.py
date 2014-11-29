#!/usr/bin/env python

import numpy, sys, argparse
from TrainData import TrainData
from NeuralNetwork import *
from progress.bar import Bar

def train(args):
    nn = NeuralNetwork(in_size = 784, hidden_size = 300, out_size = 10)
    images, labels, length = TrainData.read(args.data_set, dataset = "training")
    bar = Bar('Training', max = length)

    for i in range(length):
        x = images[i]
        y = TrainData.to_formatted_array(labels[i])

        ## update
        nn.fit(x, y)

        bar.next()

    bar.finish()
    nn.save(args.nn)
    print("saved to %s" % args.nn)

def evaluate(args):
    images, labels, length = TrainData.read(args.data_set, dataset = "testing")
    nn = NeuralNetwork(in_size = 784, hidden_size = 300, out_size = 10)
    nn.load(args.nn)
    bar = Bar('Evaluating', max = length)

    ok = 0
    for i in range(length):
        x = images[i]
        y = labels[i]
        if int(y) == int(nn.predicate(x)):
            ok += 1
        bar.next()

    bar.finish()
    print("{0:05d} / {1:05d} = {2:3.2f}%".format(ok, length, 100*ok/length))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Handwritten numerical classifier')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--data-set', '-d',
                              type = str,
                              help = "data set to train. Expects mnist.pkl.gz")
    train_parser.add_argument('--nn', '-n',
                              type=str,
                              help = "output file path of trained NeuralNetwork weight parameters")

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.set_defaults(func=evaluate)
    evaluate_parser.add_argument('--data-set', '-d',
                                 type=str,
                                 help = "data set of evaluation data. Expects mnist.pkl.gz")
    evaluate_parser.add_argument('--nn', '-n',
                                 type=str,
                                 help = "file path of trained NeuralNetwork weight parameters")

    args = parser.parse_args()
    args.func(args)
