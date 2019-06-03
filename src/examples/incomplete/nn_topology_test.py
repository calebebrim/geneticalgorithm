import unittest
import numpy as np
from ai.genetic_algorithm.examples.incomplete.nn_topology import neuronValue

class TestNNTopology(unittest.TestCase):

    def testNeuronValue(self):
        V = np.zeros(8,dtype=float)
        b = np.zeros(8)
        W = np.zeros((8, 8))
        
        neuron_pos = 2
        V[[0,1]] = 1,3
        W[[0,1] ,2] = 2
        b[neuron_pos] = 1
        neuronValue(V, W, b, neuron_pos)
        self.assertEqual(V[neuron_pos],9,'should return 9')


if __name__ == '__main__':
    unittest.main()
