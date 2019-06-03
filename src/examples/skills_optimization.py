import sys
import numpy as np
import GeneticAlgorithm
from DataProcessing import binary_ops


def main():
    js_skill = [10,5,2,3]
    speed_skill = [2,3,1,3]
    hakaton = 3
    genesize = binary_ops.bitsNeededToNumber(3)*hakaton



    

    
    def fitness(gene):
        bits = gene.reshape((hakaton,-1)) 
        val = binary_ops.bitsToBytes(bits)
        # raise Exception('myvalue:{}'.format(val))
        score = sum(np.array(js_skill)[val])
        score = score + sum(np.array(speed_skill)[val])
        
        # if score > sum(equipe_brabo):
        #     score = score - 10000
        # [1 2 1] = 
        for n1 in range(0,len(val)-1):
            for n2 in range(n1+1,len(val)):
                if val[n2] == val[n1]:
                    score -= 1000

        # raise Exception(val)
        return score

    expected = binary_ops.bitsToBytes(np.array([[True]*5])) + \
        binary_ops.bitsToBytes(np.array([[True]*5]))
    # print('DesiredValue:', expected)

    ga = GeneticAlgorithm.GA(genesize, population_size=10,
                             epochs=1000, maximization=True)

    ga.debug = False
    ga.verbose = True

    best, pop, score = ga.run(fitness,multiple=False)
    # print(score)

    def evaluate(gene):
        print('==========Evaluation=============')
        bits = gene.reshape((hakaton, -1))
        val = binary_ops.bitsToBytes(bits)
        # raise Exception('myvalue:{}'.format(val))
        skill_power = np.array(js_skill)[val]
        print(skill_power)
    
    print('BEST: ', best)
    evaluate(np.array([best]))
    # print(ga.history['statistics'])


if __name__ == '__main__':
    main()
