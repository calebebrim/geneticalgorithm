# Author: Calebe Brim
# Date: 02/08/18
import math
import time

import numpy as np


class GAU(object):
    '''
        Genetic Algorithm Util Functions

        __init_population__
        __mutation__
        __selection__
        __crossover__
        __statisics__

    '''

    @staticmethod
    def __init_population__(gene_size, population_size, dtype=bool, mn=-100000, mx=10000):
        '''  
            Must prepare for other types.
            currently only suport dtype bool
        '''

        
        if(dtype not in [bool, int
                         # TODO:
                         #   enable random population generation for dtypes:
                         #
                         #   np.float, np.double
                         #
                         ]):
            raise Exception('{} dtype not supported.'.format(dtype))

        if(dtype == bool):
            return np.random.choice([True, False], (population_size, gene_size))
        elif(dtype == int):
            return np.random.randint(mn, high=mx, size=(population_size, gene_size))

    @staticmethod
    def __init_population_with_mapping__(genome_size, population, dtype=bool):
            population_map = {}
            max_values = 2**genome_size
            population = min(max(100000, max_values), population)
            while len(population_map.keys()) < population:
                individual = np.random.choice([True, False], (1, genome_size))
                population_map[str(individual)] = individual
            return np.stack(list(population_map.values())).reshape((-1, genome_size))



    @staticmethod
    def __mutation__(pop, mutation_prob=0.30, mn=-10000, mx=10000, dtype=bool):
        selector = np.random.choice([True, False], pop.shape, p=[mutation_prob, 1-mutation_prob])
        # print(selector.shape)
        # print(selector)
        if(dtype == bool):
            pop[selector] = np.invert(pop[selector])
        elif (dtype == int):
            pop[selector] = np.random.randint(mn, high=mx, shape=selector.sum())

        return pop

    @staticmethod
    def __mutation__next__(pop, pop_size, mutation_prob=0.25, dtype=bool):
        index = 0
        genome_size = pop.shape[1]
        max_pop = 2**genome_size
        pop_size = min(pop_size, max_pop)
        population_map = {} # {str(row): row for row in pop}

        while len(population_map.keys()) < pop_size:
            genome = 1 == pop[index % pop.shape[0], :]
            selector = np.random.choice([True, False], genome.shape, p=[
                                        mutation_prob, 1-mutation_prob])
            
            genome[selector] = np.invert(genome[selector])
            population_map[str(genome)] = genome
            # pop = np.concatenate((pop, [genome]), axis=0)
            index += 1
        return np.stack(list(population_map.values())).reshape((-1, genome_size))
    
    @staticmethod
    def __selection__(pop, score, selection_count=4, maximization=False):
        
        if(maximization):
            sindex = (-score).argsort()[0:selection_count]
        else:
            sindex = score.argsort()[0:selection_count]
        
        selection = pop[sindex]
        return selection, score[sindex]

    @staticmethod
    def __crossover__(pop):
        ''' 
            Crossover genes information of all samples
        '''
        # print(pop.shape)
        crosspoints = np.random.randint(0,pop.shape[1]-1)
            
        # print(crosspoints)
        pop_PART1 = pop[:, :crosspoints]
        pop_PART2 = pop[:, crosspoints:]
        np.random.shuffle(pop_PART1)
        np.random.shuffle(pop_PART2)
        pop = np.concatenate((pop_PART1,pop_PART2),axis=1)
        # for i in range(0, pop.shape[0]-2):
        #     cross = np.append(pop[i:pop.shape[0]-2, :crosspoints[i]],
        #                       pop[i+1:pop.shape[0]-1, crosspoints[i]:], axis=1)
        #     pop = np.append(pop, cross, axis=0)
        return pop

    @staticmethod
    def __crossover__random_gene_choosing__(pop):
        lines = math.floor(pop.shape[0]/2)*2
        vlen = list(range(lines))
        np.random.shuffle(vlen)
        s_order = np.array(vlen)
        
        s_order = s_order.reshape(-1, 2)
        pop1 = pop[s_order[:,0],:]
        pop2 = pop[s_order[:,1],:]
        pop3 = pop[s_order[:, 1], :]
        pop4 = pop[s_order[:, 0], :]
        
        pops1 = np.concatenate((pop1,pop2),axis=1)
        pops2 = np.concatenate((pop3,pop4),axis=1)
        pops  = np.concatenate((pops1,pops2),axis=0)
        def cross(i):
            hlen = list(range(int(pops.shape[1])))
            np.random.shuffle(hlen)
            # print(pops[i, hlen])
            return pops[i, [hlen[:int(pops.shape[1]/2)]]]
        
        # print(list(range(min(pops.shape[0], childrens))))
        child = tuple([cross(i) for i in range(pops.shape[0])])

        pop = np.concatenate(child, axis=0)
        
        return pop

    @staticmethod
    def __crossover__allan_v_method__(pop):
        '''
            Crossover genes information of all samples
        '''
        # print(pop.shape)

        cross_pop = np.zeros(pop.shape, int)

        selector = np.random.randint(0, 2, pop.shape, bool)
        cross_pop[selector] = pop[selector]
        cross_pop[np.logical_not(selector)] = pop[1-np.array(list(range(pop.shape[0]))), :][np.logical_not(selector)]

        return cross_pop
        
    @staticmethod
    def __statisics__(pop, score):
        '''
            Calculate statistics of each individual and save the scores
        '''
        metrics = {
            'max': np.max(score), 
            'min': np.min(score), 
            'avg': np.average(score)}
        # print('Max: ', np.max(score), ' Min: ', np.min(score), ' Average: ', np.average(score))
        # print(metrics)
        return metrics

    @staticmethod
    def __last_unchanged__(ga,last=10):
        '''
            Use the last 10 to stop the main loop
        '''
        average = np.average(ga.history['bests'][-last:])
        if((average == ga.history['bests'][-1]) & (len(ga.history['bests'])>10)):
            ga.stop_requested = True
            if ga.verbose:
                print("Last {} bests scores keep unchanged: {}".format(last,ga.history['bests'][-1]))

    @staticmethod
    def __last_unchanged_mutation_update__(ga, last=10):
        """
            Use the last 10 to stop the main loop
        :param ga:
        :param last:
        :return:
        """
        average = np.average(ga.history['bests'][-last:])
        last_best = ga.history['bests'][-1]
        if (average == last_best) & (len(ga.history['bests'])>10):
            ga.stop_requested = True
            if ga.verbose:
                print("Last {} bests scores keep unchanged: {}".format(last, ga.history['bests'][-1]))

        new_prob = min(0.8, 0.25 + (0.10 * np.array(ga.history['bests'] == last_best).sum()))
        if (new_prob != ga.mutation_prob) & ga.debug:
            print("mutation prob: {}".format(new_prob))
        ga.mutation_prob = new_prob

class GA():
    import numpy as np
    
    def __init__(self,
                 genome_size,
                 gene_type=bool,
                 epochs=1000,
                 selection_count=10,
                 population_size=100,
                 maximization=False,
                 debug=False,
                 verbose=True,
                 ephoc_generations=100,
                 population=GAU.__init_population_with_mapping__,
                 mutation=GAU.__mutation__next__,
                 crossover=GAU.__crossover__allan_v_method__,
                 selection=GAU.__selection__,
                 statistics=GAU.__statisics__,
                 on_ephoc_ends=None,
                 stop_policy=GAU.__last_unchanged_mutation_update__):
        """
            Initialize Genetic Algorithm

            Default Usage:

            from geneticalgorithm import GeneticAlgorithm

            genesize = 10
            def bitsToBytes(values):
                return values.dot(2**np.arange(gene.shape[1])[::-1])
            ga_route_sort_all_fitness = lambda gene: sum(bitsToBytes(gene[:,:5]),bitsToBytes(gene[:,5:]))


            ga = GeneticAlgorithm.GA(genesize,population_size=100)
            best,pop,score = ga.run(ga_route_sort_all_fitness)
        :param genome_size:
        :param gene_type:
        :param epochs:
        :param selection_count:
        :param population_size:
        :param maximization:
        :param debug:
        :param verbose:
        :param ephoc_generations:
        :param population:
        :param mutation:
        :param crossover:
        :param selection:
        :param statistics:
        :param on_ephoc_ends:
        :param stop_policy:
        """

        # Documentation incomplete
        self.genome_size = genome_size
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.selection_count = selection_count
        self.population = population
        self.population_size = population_size
        self.populationType = gene_type
        self.maxepochs = epochs


        self.debug = debug
        self.verbose = verbose
        self.maximization = maximization
        self.ephoc_generations = ephoc_generations
        self.statisics = statistics
        self.on_ephoc_ends_callback = on_ephoc_ends
        self.stop_policy = stop_policy

        # resetable
        self.pop = None
        self.history = None
        self.pop = None
        self.mutation_prob = None
        self.best_score = None
        self.stop_requested = None
        self.reset()

        if self.verbose:
            print('''Generating Population With: 
                    \r\t- Gene Size: {}
                    \r\t- Population Size: {}
                    \r\t- Population Type: {}
                    \r\t- Generations: {}
                    \r\t- Generations for each ephoch: {}
                    \r\t- Selection Count: {}
            '''.format(genome_size, population_size, gene_type, epochs, ephoc_generations, self.selection_count))

    def reset(self):
        self.history = {
            "score": [],
            "bests": [],
            "population": [],
            "statistics": []
        }
        self.pop = None
        self.mutation_prob = 0.25
        self.best_score = -np.inf if self.maximization else np.inf
        self.stop_requested = False


    def on_ephoc_ends(self, pop, score, statistics, best_score):
        self.history['population'].append(pop)
        self.history['score'].append(score)
        self.history['bests'].append(best_score)
        self.history['statistics'].append(statistics)
        if self.verbose: 
            ephoch = len(self.history['bests'])
            print("================(Epoch: {} Generation: {})=======================".format(ephoch, ephoch*self.ephoc_generations))
            print(statistics)
            print('Best: ', best_score)

        if self.on_ephoc_ends_callback:
            self.on_ephoc_ends_callback(self, self.best_genome)
        

        if self.stop_policy is not None:
            self.stop_policy(self)
        if self.verbose:
            print("================(Epoch: {} Generation: {})=======================".format(ephoch, ephoch*self.ephoc_generations))

    def fitness_handler(self, pop, fitness, paralel, threads=1, multiple=True):
        if pop.shape[1] != self.genome_size:
            Warning("Population genes ({}) are different from preset genome_size ({})".format(pop.shape[1], self.genome_size))
        if paralel:
            from concurrent.futures import ThreadPoolExecutor

            # def executor(individual,index,scores):
            #     scores[index] = ga_route_sort_all_fitness(individual)

            def executor_fn(start,end):
                return self.fitness_handler(pop[start:min(end,pop.shape[0]),:],fitness=fitness,paralel=False,multiple=multiple,threads=threads)

            e = list(range(pop.shape[0]))
            score = np.empty(pop.shape[0])

            with ThreadPoolExecutor(max_workers=threads) as executor:

                e = []
                step = round(pop.shape[0]/threads)
                for p in range(0, pop.shape[0], step):
                    e.append(executor.submit(executor_fn, p, p+step))
                
                results = tuple([e[i].result() for i in range(len(e))])
                
                score = np.concatenate(results)
                
            score = np.array(score)
        elif multiple:
            # print('Using Multiple')
            score = fitness(pop)
        else:
            # print('Not Using Multiple')
            score = []
            for gene in pop:
                curr = fitness(gene)
                score.append(curr)
            score = np.array(score)

        evaluations = score.shape[0]
        samples = pop.shape[0]

        if threads == 1:
            if evaluations != samples:
                raise Exception("The number of returned evaluations ({}) must be equals to provided samples ({}). ".format(
                    evaluations, samples))
        return score

    def run_get_pop(self):
        if self.debug:
            self.execution_start = time.time()
        if self.pop is None:
            if(self.debug): 
                print('Initializing Population...')
                self.pop_init_start_time = time.time()
            self.pop = self.population(genome_size=self.genome_size, population=self.population_size, dtype=self.populationType)
            if self.debug: 
                self.pop_init_end_time = time.time()
                print("Take {} to init pop".format(self.pop_init_end_time-self.pop_init_start_time))
    
        return self.pop

    def run_get_scores(self):
        pop = self.run_get_pop()
        if self.debug: t = time.time()
        score = self.fitness_handler(pop,fitness=self.fitness,paralel=self.paralel,threads=self.threads,multiple=self.multiple)
        if self.debug:
            t = time.time() - t
            print("Take {} to get score".format(t))
        return score

    def run(self, fitness, paralel=False, threads=1, multiple=False):
        '''
            Used to effectivelly run the genetic algorithm

            Usage: 
            
                ga = GA(genome_size=1000,population_size=100,maximization=False,epochs=100)

                (best_genome, pop, score) = ga.run(lambda genes: sum(genes),multiple=False)
        '''
        self.threads = threads
        self.multiple = multiple
        self.fitness = fitness
        self.paralel = paralel

        pop = self.run_get_pop()
        score = self.run_get_scores()

        statistics = self.statisics(pop, score)
        pop, score = self.selection(pop, score, self.selection_count, maximization=self.maximization)

        self.best_genome, self.best_score = (pop[0], score[0])
        
        
        self.stop_requested = False




        for i in range(1,self.maxepochs+1):

            if self.debug:
                print('generation>>', i)
            
            pop = np.concatenate((pop, np.array([self.best_genome])), axis=0)

            if self.debug: t = time.time()
            pop = self.crossover(pop)
            if self.debug:
                t = time.time() - t
                print("Take {} to crossover: {}".format(t,pop.shape))
            # if(self.debug):print('Crossover>>', pop.shape)
            if self.debug:
                t = time.time()

            pop = self.mutation(pop, self.population_size, mutation_prob=self.mutation_prob)
            if self.debug:
                t = time.time() - t
                print("Take {} to mutation {} ".format(t,pop.shape))
            # if(self.debug):print('Mutation>>', pop.shape)
            
            if self.debug: t = time.time()
            # print(pop.shape)
            score = self.fitness_handler(pop, fitness=fitness, paralel=paralel, threads=threads, multiple=multiple)
            if self.debug:
                t = time.time() - t
                print("Take {} to score".format(t))

            statistics = self.statisics(pop, score)
            pop, score = self.selection(pop, score, selection_count=self.selection_count,maximization=self.maximization)
            
            if self.maximization:
                if score[0] > self.best_score:
                    self.best_genome, self.best_score = (pop[0], score[0])
            else:
                if score[0] < self.best_score:
                    self.best_genome, self.best_score = (pop[0], score[0])

            if (i % self.ephoc_generations)==0:
                self.on_ephoc_ends(pop, score, statistics, self.best_score)
            

            if(self.debug):
                print('Selection>>', pop.shape)
            self.pop = pop    
            if(self.stop_requested):
                break
        return self.best_genome, pop, score

    

