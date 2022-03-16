import sys
import numpy as np
from src import GeneticAlgorithm
from src.utils.binary_ops import bitsToBytes, bitsNeededToNumber

def main():


    genesize = 2*bitsNeededToNumber(10)
    nbits = int(round(genesize/2))

    

    def fitness(gene):
        # print(gene)
        # print(nbits)
        one = bitsToBytes(gene[:, 0:nbits])
        two = bitsToBytes(gene[:, nbits:])

        return np.sum([one, two], axis=0)

    print('nbits:', nbits)
    expected = 0
    # print('DesiredValue:', expected)

    ga = GeneticAlgorithm.GA(genesize, population_size=10,
                             epochs=1000,ephoc_generations=100, maximization=False)

    ga.debug = False
    ga.verbose = True

    best, pop, score = ga.run(fitness, multiple=True)
    
    # print(ga.history["population"])

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
