#!/usr/bin/env python

import numpy, sys, os, pprint, re
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
