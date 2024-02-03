import sys
import numpy as np
from src import GeneticAlgorithm 
from src.utils import binary_ops


def main():
    """ 
    This is a hakaton with 3 participants for round. 
    This GA chooses the best candidates combinations. 
    Each position represent one participant positions 0 to 3: 4 canditates
    
    """
    
    # Parameters

    candidates = ["antonio", "romeo", "carlos", "frank", "mark","jon"]
    # js skills for each participant
    js_skill = [10,5,2,3,10,2]
    # programming speed for each participant
    speed_skill = [2,3,1,3,5,20]
    
    
    max_participants = 3
    count_candidates = len(candidates)
    
    # The GA gene size will be the ammount of bites needed to reference the number of candidates 
    # times the max number of participants per round. 
    genesize = binary_ops.bitsNeededToNumber(count_candidates-1)*max_participants



    

    
    def fitness(gene: np.array):
        # gene example: 
        #       [ True  True  True False False False]



        # format the gene to split every participant bits into one line for bitsToBytes funcion
        # ex: 
        #   [[False  True]
        #   [ True  True]
        #   [False False]]
        bits = gene.reshape((max_participants,-1)) 
        
        # The selected candidate index for each position 
        # ex: array([3, 1, 0])
        val = binary_ops.bitsToBytes(bits)
        
        bad_gene_score = 0

        # choosing an inexisting person
        inexistent_candidate_error_count = np.sum(val>=count_candidates)>0
        # choosing same person multiple times GA penalty
        same_candidate_error_count = np.unique(val).size-max_participants

        
        if same_candidate_error_count!=0:
            bad_gene_score += 1000*np.abs(same_candidate_error_count)
        
        if inexistent_candidate_error_count>0:
            bad_gene_score += 1000*np.abs(inexistent_candidate_error_count)

        if bad_gene_score!=0:
            return -bad_gene_score
        

        # js team score
        score = sum(np.array(js_skill)[val])
        # + speed team score
        score = score + sum(np.array(speed_skill)[val])
        

        

        return score+np.sum(np.diff(val))

    expected = binary_ops.bitsToBytes(np.array([[True]*5])) + \
               binary_ops.bitsToBytes(np.array([[True]*5]))
    

    ga = GeneticAlgorithm.GA(genesize, population_size=10, epochs=1000, maximization=True)

    ga.debug = False
    ga.verbose = True

    best, pop, score = ga.run(fitness)


    def evaluate(gene):
        print('==========Evaluation=============')
        bits = gene.reshape((max_participants, -1))
        val = binary_ops.bitsToBytes(bits)

        print("name\tJS\tspeed\n"+"\n".join([f"{candidates[v]}\t{js_skill[v]}\t{speed_skill[v]}" for v in val]))
        
    
    evaluate(np.array([best]))


if __name__ == '__main__':
    main()
