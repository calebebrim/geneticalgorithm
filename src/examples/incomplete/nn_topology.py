import sys
import numpy as np
from ai.genetic_algorithm import GeneticAlgorithm
from ai.data_processing.binary_ops import bitsToBytes
from ai.data_processing.binary_ops import bitsNeededToNumber

import unittest

def neuronValue(V,W,b, x_neuron):
    '''
    Where V = [1,2, ... ] represent the input values and neuron outputs.
    And W[i,j] is an matrix where i = j = len(V)

    W determine the weights of each connection between neurons.
     

    '''
    w_s = W[:, x_neuron] > 0
    V[x_neuron]=V[w_s].dot(W[w_s,x_neuron])+b[x_neuron]
    print('b:', b)
    print('x_neuron:', x_neuron)
    print('W:', W)
    print('V:',V)
    






def main():

    


    neurons = 13
    connections = 100
    # 3 is the dimension of N
    nbits = 3 * connections * bitsNeededToNumber(neurons)
    

    def fitness(gene):
        N = np.reshape(bitsToBytes(np.reshape(gene, (-1, bitsNeededToNumber(neurons)))),(-1,3))
        score = 0
        score -= sum(N[:,1]==N[:,2])
        


        # print(score)
        return score

    


    ga = GeneticAlgorithm.GA(gene_size=nbits, population_size=10,
                             epochs=1000, maximization=True,ephoc_generations=10)

    ga.debug = False
    ga.verbose = True

    best, pop, score = ga.run(fitness,multiple=True)
    # print(score)

    # def evaluate(gene):
    #     print('==========Evaluation=============')
    #     one_bits = gene[:, 0:nbits]
    #     # print(one_bits.shape)
    #     one = bitsToBytes(one_bits)
    #     two = bitsToBytes(gene[:, nbits:])
    #     score = np.sum([one, two], axis=0)

    #     # print(one,two,score)
    #     print('Achieved: ', score, 'Expected:', expected)

    #     return score
    # print('BEST: ', best)
    # evaluate(np.array([best]))
    # print(ga.history['statistics'])


if __name__ == '__main__':
    main()




