from search_space import HybridSearchSpace
from individual import Individual
from random import sample, choices
import numpy as np
from gnn_model_manager import GNNModelManager
import copy

class Population(object):
    
    def __init__(self, args):
        
        self.args = args
        hybrid_search_space = HybridSearchSpace(self.args.num_gnn_layers)
        self.hybrid_search_space = hybrid_search_space
        
        # prepare data set for training the gnn model
        self.load_trining_data()
    
    def load_trining_data(self):
        self.gnn_manager = GNNModelManager(self.args)
        self.gnn_manager.load_data(self.args.dataset)
        
        # dataset statistics
        print(self.gnn_manager.data)
        
    def init_population(self):
        
        struct_individuals = []
        
        for i in range(self.args.num_individuals):
            net_genes = self.hybrid_search_space.get_net_instance()
            param_genes = self.hybrid_search_space.get_param_instance()
#             param_genes = [self.args.lr, self.args.in_drop, self.args.weight_decay]
            instance = Individual(self.args, net_genes, param_genes)
            struct_individuals.append(instance)
        
        self.struct_individuals = struct_individuals
    
    # run on the single model with more training epochs
    def single_model_run(self, num_epochs, actions):
        self.args.epochs = num_epochs
        self.gnn_manager.train(actions)
        
        
    def cal_fitness(self):
        """calculate fitness scores of all individuals,
          e.g., the classification accuracy from GNN"""
        for individual in self.struct_individuals:
            individual.cal_fitness(self.gnn_manager)
            
    def parent_selection(self):
        "select k individuals by fitness probability"
        k = self.args.num_parents
        
        # select the parents for structure evolution
        fitnesses = [i.get_fitness() for i in self.struct_individuals]
        fit_probs = fitnesses / np.sum(fitnesses)
        struct_parents = choices(self.struct_individuals, k=k, weights=fit_probs)
        
        # select the parents for hyper-parameter evolution
        fitnesses = [i.get_fitness() for i in self.param_individuals]
        fit_probs = fitnesses / np.sum(fitnesses)
        param_parents = choices(self.param_individuals, k=k, weights=fit_probs)
#         print(parents[0].get_net_genes())
#         print(parents[1].get_net_genes())
        return struct_parents, param_parents
    
    def crossover_net(self, parents):  
        "produce offspring from parents for better net architecture"
        p_size = len(parents)
        maximum = p_size * (p_size - 1) / 2
        if self.args.num_offsprings > maximum:
            raise RuntimeError("number of offsprings should not be more than " 
                               + maximum)
            
        # randomly choose crossover parent pairs
        parent_pairs = []
        while len(parent_pairs) < self.args.num_offsprings:
            indexes = sample(range(p_size), k=2)
            pair = (indexes[0], indexes[1])
            if indexes[0] > indexes[1]:
                pair = (indexes[1], indexes[0])
            if not pair in parent_pairs:
                parent_pairs.append(pair)
        
#         print(parent_pairs)
        # crossover to generate offsprings
        offsprings = []
        gene_size = len(parents[0].get_net_genes())
        for i, j in parent_pairs:
            parent_gene_i = parents[i].get_net_genes()
            parent_gene_j = parents[j].get_net_genes()
            # select a random crossover point
            point_index = parent_gene_j.index(sample(parent_gene_j, 1)[0])
            offspring_gene_i = parent_gene_j[:point_index]
            offspring_gene_i.extend(parent_gene_i[point_index:])
            offspring_gene_j = parent_gene_i[:point_index]
            offspring_gene_j.extend(parent_gene_j[point_index:])
            
            # create offspring individuals
            offspring_i = Individual(self.args, offspring_gene_i, 
                                     parents[i].get_param_genes())
            offspring_j = Individual(self.args, offspring_gene_j, 
                                     parents[j].get_param_genes())
            
            offsprings.append([offspring_i, offspring_j])
            
        return offsprings   

    def crossover_param(self, parents):
        p_size = len(parents)
        maximum = p_size * (p_size - 1) / 2
        if self.args.num_offsprings > maximum:
            raise RuntimeError("number of offsprings should not be more than " 
                               + maximum)
        # randomly choose crossover parent pairs
        parent_pairs = []
        while len(parent_pairs) < self.args.num_offsprings:
            indexes = sample(range(p_size), k=2)
            pair = (indexes[0], indexes[1])
            if indexes[0] > indexes[1]:
                pair = (indexes[1], indexes[0])
            if not pair in parent_pairs:
                parent_pairs.append(pair)

        offsprings = []
        params_size = len(parents[0].get_param_genes())
        for i, j in parent_pairs:
            parent_gene_i = parents[i].get_param_genes()
            parent_gene_j = parents[j].get_param_genes()
            # select a random crossover point
            point_index = parent_gene_j.index(sample(parent_gene_j, 1)[0])
            offspring_gene_i = parent_gene_j[:point_index]
            offspring_gene_i.extend(parent_gene_i[point_index:])
            offspring_gene_j = parent_gene_i[:point_index]
            offspring_gene_j.extend(parent_gene_j[point_index:])
            
            # create offspring individuals
            offspring_i = Individual(self.args, parents[i].get_net_genes(), 
                                     offspring_gene_i)
            offspring_j = Individual(self.args, parents[j].get_net_genes(), 
                                     offspring_gene_j)
            
            offsprings.append([offspring_i, offspring_j])
        return offsprings  
        
    
    def mutation_net(self, offsprings):
        """perform mutation for all new offspring individuals"""
        for pair in offsprings:
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_net_gene()
                pair[0].mutation_net_gene(index, gene, 'struct')
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_net_gene()
                pair[1].mutation_net_gene(index, gene, 'struct')
                
    def mutation_param(self, offsprings):
        """perform mutation for all new offspring individuals"""
        for pair in offsprings:
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_param_gene()
                pair[0].mutation_net_gene(index, gene, 'param')
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_param_gene()
                pair[1].mutation_net_gene(index, gene, 'param')
                
    def find_least_fittest(self, individuals):
        fitness = 10000
        index =-1
        for elem_index, elem in enumerate(individuals):
            if fitness > elem.get_fitness():
                fitness = elem.get_fitness()
                index = elem_index
                
        return index
                           
    def cal_fitness_offspring(self, offsprings):
        survivors = []
        for pair in offsprings:
            offspring_1 = pair[0]
            offspring_2 = pair[1]
            offspring_1.cal_fitness(self.gnn_manager)
            offspring_2.cal_fitness(self.gnn_manager)
            if offspring_1.get_fitness() > offspring_2.get_fitness():
                survivors.append(offspring_1)
            else:
                survivors.append(offspring_2)
        
        return survivors
                
    def update_population_struct(self, survivors):
        """update current population with new offsprings"""
        for elem in survivors:
            out_index = self.find_least_fittest(self.struct_individuals)
            self.struct_individuals[out_index] = elem

    def compare_action(self, a1, a2):
        for i in range(len(a1)):
            if a1[i] != a2[i]:
                return False
        return True

    def combine_population(self):
        for elem in self.param_individuals:
            action_param = elem.get_net_genes()
            for i in range(self.args.num_individuals):
                out_index = self.find_least_fittest(self.struct_individuals)
                action_struct = self.struct_individuals[i].get_net_genes()
                if self.compare_action(action_struct, action_param):
                    if self.struct_individuals[i].get_fitness() < elem.get_fitness():
                        self.struct_individuals[i] = elem
                        print('ssssssssssssssssssssss')
                elif self.struct_individuals[out_index].get_fitness() < elem.get_fitness():
                    self.struct_individuals[out_index] = elem
                    print('ssssssssssssssssssssssmmmmmmmmm')

            
    def update_population_param(self, survivors):
        """update current population with new offsprings"""
        for elem in survivors:
            out_index = self.find_least_fittest(self.param_individuals)
            self.param_individuals[out_index] = elem
    
    def print_models(self, iter):
        
        print('===begin, current population ({} in {} generations)===='.format(
                                    (iter+1), self.args.num_generations))
        
        best_individual = self.struct_individuals[0]
        for elem_index, elem in enumerate(self.struct_individuals):
            if best_individual.get_fitness() < elem.get_fitness():
                best_individual = elem
            print('struct space: {}, param space: {}, validate_acc={}, test_acc={}'.format(
                                elem.get_net_genes(), 
                                elem.get_param_genes(),
                                elem.get_fitness(),
                                elem.get_test_acc()))
        print('------the best model-------')
        print('struct space: {}, param space: {}, validate_acc={}, test_acc={}'.format(
                           best_individual.get_net_genes(), 
                           best_individual.get_param_genes(),
                           best_individual.get_fitness(),
                           best_individual.get_test_acc()))    
           
        print('====end====\n')
        
        return best_individual
    
                    
    def evolve_net(self):
        # initialize population
        self.init_population()
        # calculate fitness for population
        self.cal_fitness()
        self.param_individuals = copy.deepcopy(self.struct_individuals)
        
        actions = []
        params = []
        train_accs = []
        test_accs = []
        
        for i in range(self.args.num_generations):
            
            struct_parents, param_parents = self.parent_selection() # parents selection
            
            # GNN structure evolution
            print('GNN structure evolution')
            struct_offsprings = self.crossover_net(struct_parents) # crossover to produce offsprings
            self.mutation_net(struct_offsprings) # perform mutation
            struct_survivors = self.cal_fitness_offspring(struct_offsprings) # calculate fitness for offsprings
            self.update_population_struct(struct_survivors) # update the population 
            
            # GNN hyper parameter evolution
            print('NN hyper parameter evolution')
            param_offsprings = self.crossover_param(param_parents)
            self.mutation_param(param_offsprings) # perform mutation
            param_survivors = self.cal_fitness_offspring(param_offsprings) # calculate fitness for offsprings
            self.update_population_param(param_survivors) # update the population      
            
            # combine the structure population and parameter population
            print('combine the structure population and parameter population')
#            self.update_population_struct(self.param_individuals)
            self.combine_population()
            self.param_individuals = copy.deepcopy(self.struct_individuals)
            
            best_individual = self.print_models(i)
            actions.append(best_individual.get_net_genes())
            params.append(best_individual.get_param_genes())
            train_accs.append(best_individual.get_fitness())
            test_accs.append(best_individual.get_test_acc())
        
        print(actions)           
        print(params)           
        print(train_accs)           
        print(test_accs)           
        