#!/usr/bin/env python
import unittest, os
from NeuralNetwork import *
import numpy

class SigmoidTest(unittest.TestCase):
    def test_sigmoid(self):
        self.assertTrue(abs(sigmoid(0.0) - 0.5) < 1e-10)
        self.assertTrue(abs(sigmoid(10.0) - 1.0) < 1e-4)
        self.assertTrue(abs(sigmoid(-10.0) - 0.0) < 1e-4)

class NodeTest(unittest.TestCase):
    def test_init(self):
        node = Node(numpy.array([1,2,3]))
        self.assertEqual(node.w.shape, (3,))

    def test_init_assert(self):
        self.assertRaises(AssertionError, Node, 0)
        self.assertRaises(AssertionError, Node, [])
        self.assertRaises(AssertionError, Node, [0])
        self.assertRaises(AssertionError, Node, numpy.array([[1,2],[3,4]]))

    def test_activate(self):
        node = Node(numpy.array([1,1,1]))
        self.assertRaises(AssertionError, lambda: node.activate(numpy.array([1,2])))
        self.assertEqual(node.activate(numpy.array([1,1,1])), sigmoid(3))

class LayerTest(unittest.TestCase):
    def test_init(self):
        layer = Layer(4,3)
        self.assertEqual(len(layer.nodes), 3)
        for node in layer.nodes:
            self.assertEqual(node.w.shape, (4,))

    def test_init_custom(self):
        wl = [numpy.array([1,2,3]), numpy.array([4,5,6])]
        layer = Layer(wl)
        self.assertEqual(layer.nodes[0].w.all(), wl[0].all())
        self.assertEqual(layer.nodes[1].w.all(), wl[1].all())

    def test_fire(self):
        layer = Layer([numpy.array([10,10,10]), numpy.array([1,-10,-10])])
        out = layer.fire(numpy.array([1,1,1]))
        self.assertTrue(abs(out[0] - 1.0) < 1e-4)
        self.assertTrue(abs(out[1]) < 1e-4)

class NNTest(unittest.TestCase):
    def test_init(self):
        nn = NeuralNetwork([4,3])
        self.assertEqual(len(nn.layers), 1)
        self.assertEqual(len(nn.layers[0].nodes), 3)
        self.assertEqual(len(nn.layers[0].nodes[0].w), 5) ## add input constant

    def test_init_dual_layer(self):
        nn = NeuralNetwork([100,30,10])
        self.assertEqual(len(nn.layers), 2)
        self.assertEqual(len(nn.layers[0].nodes), 30)
        self.assertEqual(len(nn.layers[0].nodes[0].w), 101)
        self.assertEqual(len(nn.layers[1].nodes), 10)
        self.assertEqual(len(nn.layers[1].nodes[1].w), 31)

    def test_fire(self):
        nn = NeuralNetwork([100,30,10])
        out = nn.fire(numpy.random.rand(100))
        for v in out:
            self.assertTrue(0 < v < 1)


if __name__ == '__main__':
    unittest.main()
