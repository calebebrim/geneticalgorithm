import sys
import numpy as np
from src import GeneticAlgorithm
from src.utils.binary_ops import bitsNeededToNumber

def main():

    gene_size = 2*bitsNeededToNumber(30)
    nbits = int(round(gene_size/2))

    def bitsToBytes(values):
        processed = np.array(values.dot(2**np.arange(values.shape[1])[::-1]))
        return processed

    def fitness(gene):
        one = bitsToBytes(gene[:, 0:nbits])
        two = bitsToBytes(gene[:, nbits:])
        score = np.sum([one, two], axis=0)

        # print(one,two,score)
        return score

    print('nbits:', nbits)
    expected = bitsToBytes(np.array([[True]*5])) + \
        bitsToBytes(np.array([[True]*5]))
    # print('DesiredValue:', expected)

    ga = GeneticAlgorithm.GA(gene_size, population_size=20,
                             epochs=1000, ephoc_generations=100, maximization=True)

    ga.debug = False
    ga.verbose = True

    best, pop, score = ga.run(fitness, multiple=True)
    # print(score)

    def evaluate(gene):
        print('==========Evaluation=============')
        one_bits = gene[:, 0:nbits]
        # print(one_bits.shape)
        one = bitsToBytes(one_bits)
        two = bitsToBytes(gene[:, nbits:])
        score = np.sum([one, two], axis=0)

        # print(one,two,score)
        print('Achieved: ', score, 'Expected:', expected)
        
        return score
    print('BEST: ', best)
    evaluate(np.array([best]))
    # print(ga.history['statistics'])


if __name__ == '__main__':
    main()
