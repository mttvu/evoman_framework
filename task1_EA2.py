###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

class Objective:
    def objective(self,env,x):
        # run simulation
        f,p,e,t = env.play(pcont=x)
        return f
class EA(object):

    def __init__(self, objective, environment, pop_size,tournament_size,mutation_probability, gene_length, bounds_min=None, bounds_max=None):
        self.objective = objective
        self.environment = environment
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.gene_length = gene_length
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max


    def parent_selection(self, x_old, f_old):
        # half of the population will be selected
        parents_size = int(self.pop_size / 2)
        x_parents = np.zeros((parents_size, self.gene_length), np.float64)
        f_parents = np.zeros(parents_size)

        # tournament selection
        for i in range(parents_size):
            # select tournament_size random individuals
            candidates = np.random.choice(self.pop_size, int(self.pop_size * self.tournament_size), replace=False)
            # choose the best individual among them as parent
            winner = np.argmax(f_old[candidates])
            x_parents[i] = x_old[candidates[winner]]
            f_parents[i] = f_old[candidates[winner]]

        return x_parents, f_parents

    def recombination(self, x_parents, f_parents):
        x_children = []
        parent_size = len(x_parents)
        # Uniform crossover
        for i in range(parent_size):
            # select two parents at random
            parents_idx = np.random.choice(parent_size, 2, replace=False)
            parent1 = x_parents[parents_idx[0]]
            parent2 = x_parents[parents_idx[1]]
            # generate a random binary mask for crossover
            mask = np.random.randint(0, 2, len(parent1)).astype(bool)
            # create a child with the first part from parent1 and the second part from parent2
            child = np.zeros_like(parent1)
            child[mask] = parent1[mask]
            child[~mask] = parent2[~mask]

            # create second child with the inverted mask
            child2 = np.zeros_like(parent1)
            child2[~mask] = parent1[~mask]
            child2[mask] = parent2[mask]
            x_children.append(child)
            x_children.append(child2)

        return np.array(x_children)

    def mutation(self, x_children):
        for i in range(len(x_children)):
                # loop through the genes
            for j in range(self.gene_length): 
                # generate random number between 0 and 1  
                u = np.random.uniform(0,1)
                if u < self.mutation_probability:
                    # random number between bounds
                    x_children[i][j] = np.random.uniform(low=self.bounds_min, high=self.bounds_max)
                    
        return x_children

    def survivor_selection(self, x_old, x_children, f_old, f_children):
        n_children = len(x_children)

        x_combined = np.concatenate((x_old, x_children))
        f_combined = np.concatenate((f_old, f_children))

        x_new = np.empty_like(x_old)
        f_new = np.empty_like(f_old)

        # tournament selection
        for i in range(self.pop_size):
            # select tournament_size random individuals
            candidates = np.random.choice(self.pop_size + n_children, int(self.pop_size * self.tournament_size), replace=False)
            # add best to survivors
            best = candidates[np.argmax(f_combined[candidates])]
            x_new[i] = x_combined[best]
            f_new[i] = f_combined[best]

        return x_new, f_new

    def evaluate(self, x):
        return np.array(list(map(lambda y: self.objective.objective(self.environment,y), x)))

    def step(self, x_old, f_old):
        x_parents, f_parents = self.parent_selection(x_old, f_old)
        x_children = self.recombination(x_parents, f_parents)
        x_children = self.mutation(x_children)
        f_children = self.evaluate(x_children)

        # only the children are selected to move on to the next generation

        return x_children, f_children

def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
    gene_length = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # start writing your own code from here
    bounds_max = 1
    bounds_min = -1
    population_size = 30
    num_generations = 30
    mutation_prob = 0.2
    tournament_size = 0.3
    population = np.random.uniform(bounds_min, bounds_max, (population_size, gene_length))

    objective = Objective()
    ea = EA(objective, env, population_size,tournament_size,mutation_prob, gene_length, bounds_min, bounds_max)
    f = ea.evaluate(population)

    populations = []
    populations.append(population)
    f_best = [f.max()]

    for i in range(num_generations):
        max_idx = np.argmax(f)
        f_max = f[max_idx]
        print(f"Generation: {i}, best fitness: {f_max}")
        population, f = ea.step(population, f)
        populations.append(population)
        if f.max() > f_best[-1]:
            f_best.append(f.max())
        else:
            f_best.append(f_best[-1])
    print("FINISHED!")

if __name__ == '__main__':
    main()