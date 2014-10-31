#!/usr/bin/env python
import unittest, os
from TrainData import TrainData
import numpy

class TrainDataTest(unittest.TestCase):
    def mock_data(self):
        files = ['00000-5.png', '00001-0.png', '00002-4.png']
        dir = os.path.split(os.path.abspath(__file__))[0]
        return [ os.path.join(dir, "data", f) for f in files ]

    def test_read_from_files(self):
        files = self.mock_data()
        objs = TrainData.read_from_files(files)
        self.assertIsInstance(objs, list)

        self.assertIsInstance(objs[0], TrainData)
        self.assertTrue(objs[0].number == '5')
        self.assertIsInstance(objs[0].image, numpy.ndarray)

    def test_read_from_files_empty(self):
        objs = TrainData.read_from_files([])
        self.assertTrue(len(objs) == 0 )

    def test_list_formatted_number(self):
        files = self.mock_data()
        objs = TrainData.read_from_files(files)
        num_list = objs[0].list_formatted_number()
        self.assertTrue( numpy.array_equal(num_list, numpy.array([0,0,0,0,0,1,0,0,0,0])))

if __name__ == '__main__':
    unittest.main()
