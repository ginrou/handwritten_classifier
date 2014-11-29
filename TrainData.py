#!/usr/bin/env python

import numpy, sys, os, pprint, gzip, pickle
from scipy import misc, ndimage
pp = pprint.PrettyPrinter(indent=4)

class TrainData:
    def __init__(self, image, number):
        self.image = image
        self.number = number

    def list_formatted_number(self):
        ret = numpy.zeros(10)
        ret[self.number] = 1
        return ret

    @classmethod
    def read_from_files(cls, files):
        return [TrainData.read_from_file(path) for path in files]

    @classmethod
    def read_from_file(cls, path):
        num = re.search('(\d{5})-(\d{1})', path).group(2)
        img = misc.imread(path, 1)
        return TrainData(img, num)


    @classmethod
    def read(cls, filepath, dataset = "training"):
        """
        MNSIT data set reader.
        returns (images, labels, length)
        """
        f = gzip.open(sys.argv[1], 'rb')
        train, valid, test = pickle.load(f, encoding='latin1')
        f.close()

        if dataset == "training":
            return (train[0], train[1], train[0].shape[0])
        elif dataset == "valid":
            return (valid[0], valid[1], valid[0].shape[0])
        else:
            return (test[0], test[1], test[0].shape[0])

    @classmethod
    def to_formatted_array(cls, number):
        ret = numpy.zeros(10)
        ret[number] = 1
        return ret
