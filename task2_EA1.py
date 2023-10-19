###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import matplotlib.pyplot as plt

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import math
# import optuna
# from optuna.study import StudyDirection
# from optuna_dashboard import run_server


"""
file_to_write = open("EA1/best_enemies.txt", "w")

def append_file(filestream, defeated, enemies):
    filestream.write(f"Games won: {defeated}/8\tEnemies: {enemies}\n")
"""

class NewFitnessEnvironment(Environment):
    def fitness_single(self):
        hp = self.get_playerlife()
        enemy = self.get_enemylife()
        time = self.get_time()
        a = 0.3
        b = 0.7
        return (a*hp + b*(100-enemy))/(a+b)

    def cons_multi(self, values):
        return values.mean() # - values.std()
    
    def multiple(self,pcont,econt):

        vfitness, vplayerlife, venemylife, vtime = [],[],[],[]
        for e in self.enemies:

            fitness, playerlife, enemylife, time  = self.run_single(e,pcont,econt)
            vfitness.append(fitness)
            vplayerlife.append(playerlife)
            venemylife.append(enemylife)
            vtime.append(time)

        vfitness = self.cons_multi(np.array(vfitness))

        return    vfitness, vplayerlife, venemylife, vtime

class Objective:
    def objective(self,env,x):
        # run simulation
        f,p,e,t = env.play(pcont=x)
        return f
class EA(object):

    def __init__(self, objective, environment, pop_size,tournament_size,mutation_probability, gene_length, mutation_size_1, mutation_size_2, generation, num_generations,bounds_min=None, bounds_max=None):
        self.objective = objective
        self.environment = environment
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.gene_length = gene_length
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.mutation_size_1 = mutation_size_1 
        self.mutation_size_2 = mutation_size_2 
        self.generation = generation 
        self.num_generations = num_generations
        self.mutation_size = mutation_size_1
        self.all_time_best_fitness = 0


    def parent_selection(self, x_old, f_old):
        # half of the population will be selected
        parents_size = int(self.pop_size / 2)
        x_parents = np.zeros((parents_size, self.gene_length), np.float64)
        f_parents = np.zeros(parents_size)

        population_indices = np.arange(self.pop_size)
        # tournament selection
        for i in range(parents_size):
            # select tournament_size random individuals
            if self.tournament_size < len(population_indices):
                candidates = np.random.choice(population_indices, int(self.tournament_size), replace=False)
            else:
                candidates = population_indices
            # choose the best individual among them as parent
            best = candidates[np.argmax(f_old[candidates])]
            x_parents[i] = x_old[best]
            f_parents[i] = f_old[best]
            population_indices = np.delete(population_indices, np.where(population_indices == best))

        return x_parents, f_parents

    def recombination(self, x_parents, f_parents):
        x_children = []
        parent_size = len(x_parents)
        parent_indices = np.arange(parent_size)
        # Uniform crossover
        for i in range(parent_size):
            # select two parents at random 
            parents_idx = np.random.choice(parent_indices, 2, replace=False)          
            if len(parent_indices) > 3:
                parent_indices = np.delete(parent_indices, np.where(np.isin(parent_indices, parents_idx)))

            parent1 = x_parents[parents_idx[0]]
            parent2 = x_parents[parents_idx[1]]
            # generate a random binary mask for crossover
            # mask = np.random.randint(0, 2, len(parent1)).astype(bool)
            # create a child with the first part from parent1 and the second part from parent2
            # child = np.zeros_like(parent1)
            # child[mask] = parent1[mask]
            # child[~mask] = parent2[~mask]

            # # create second child with the inverted mask
            # child2 = np.zeros_like(parent1)
            # child2[~mask] = parent1[~mask]
            # child2[mask] = parent2[mask]
            # x_children.append(child)
            # x_children.append(child2)

            child = np.zeros_like(parent1)
            child2 = np.zeros_like(parent1)
            child3 = np.zeros_like(parent1)

            for i in range(len(parent1)):
                chance = np.random.uniform(0, 1)
                if chance > 0.5:
                    child[i] = parent1[i]
                    child2[i] = parent2[i]
                else:
                    child[i] = parent2[i]
                    child2[i] = parent1[i]

                if i % 2 == 0:
                    child3[i] = parent1[i]
                else:
                    child3[i] = parent2[i]

            x_children.append(child)
            x_children.append(child2)
            x_children.append(child3)

        return np.array(x_children)

    def mutation(self, x_children):
        for i in range(len(x_children)):
                # loop through the genes
            for j in range(self.gene_length): 
                # generate random number between 0 and 1  
                u = np.random.uniform(0,1)
                if u < self.mutation_probability:
                    # create bounds for multiplier
                    upper_multiplier = 1.0 + self.mutation_size 
                    lower_multiplier = 1.0 - self.mutation_size
                    # create random multiplier for parent value
                    random_multiplier = np.random.uniform(lower_multiplier, upper_multiplier)
                    # random number between bounds
                    x_children[i][j] = x_children [i][j] * random_multiplier
                    if x_children[i][j] > self.bounds_max: 
                        x_children[i][j] = self.bounds_max
                    elif x_children[i][j] < self.bounds_min: 
                        x_children[i][j] = self.bounds_min
                    
        return x_children

    def survivor_selection(self, x_old, x_children, f_old, f_children):
        # simulated annealing
        x_new = np.empty_like(x_old)
        f_new = np.empty_like(f_old)
        children_indices = np.arange(len(x_children))

        for i in range(len(x_old)):
            selected_index = np.random.choice(children_indices)
            children_indices = np.delete(children_indices, np.where(children_indices == selected_index))

            if f_old[i] < f_children[selected_index]:
                x_new[i] = x_children[selected_index]
                f_new[i] = f_children[selected_index]
            else:
                # always keep individual with largest fitness
                # or we could keep track of the highest achieved fitness
                if i == np.argmax(f_old):
                    x_new[i] = x_old[i]
                    f_new[i] = f_old[i] 
                else:
                    accept = 1 - np.exp(-((f_old[i] - f_children[selected_index])/abs(self.all_time_best_fitness - np.mean(f_old))))
                    u = np.random.uniform(0,1)
                    # print(f'accept: {accept}, u: {u}')
                    if accept > u:
                        x_new[i] = x_children[selected_index]
                        f_new[i] = f_children[selected_index]

                    else:
                        x_new[i] = x_old[i]
                        f_new[i] = f_old[i]

        return x_new, f_new

    def evaluate(self, x):
        return np.array(list(map(lambda y: self.objective.objective(self.environment,y), x)))

    def step(self, x_old, f_old):
        x_parents, f_parents = self.parent_selection(x_old, f_old)
        x_children = self.recombination(x_parents, f_parents)
        x_children = self.mutation(x_children)
        f_children = self.evaluate(x_children)
        current_gen_best = np.maximum(max(f_children),max(f_old))
        if current_gen_best > self.all_time_best_fitness:
            self.all_time_best_fitness = current_gen_best
        x, f = self.survivor_selection(x_old, x_children, f_old, f_children)

        self.generation += 1 
        if self.generation > self.num_generations/3: 
            self.mutation_size = self.mutation_size_2
            

        return x, f

def run_EA(population_size,num_generations,mutation_prob,mutation_size_1, mutation_size_2,tournament_size,enemies, show_plot=True):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'EA1'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    # env = Environment(
    env = NewFitnessEnvironment(
                    experiment_name=experiment_name,
                    enemies=enemies,
                    multiplemode="yes",
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
    generation = 0
    population = np.random.uniform(bounds_min, bounds_max, (population_size, gene_length))

    objective = Objective()
    ea = EA(objective, env, population_size,tournament_size,mutation_prob, gene_length, mutation_size_1, mutation_size_2, generation, num_generations,bounds_min, bounds_max)
    f = ea.evaluate(population)

    populations = []
    fitness = []
    populations.append(population)
    fitness.append(f)
    f_best = [f.max()]
    
    best_f = []
    std_f = []
    mean_f = []
    best_f_idx = 0

    file_aux = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best std mean')

    for i in range(num_generations):
        best = f[np.argmax(f)]
        std = np.std(f)
        mean = np.mean(f)
        
        best_f.append(best)
        std_f.append(std)
        mean_f.append(mean)
        print(f"Generation: {i + 1}, best fitness: {best}, std: {std}, mean: {mean}")
        
        # save the population and the best result
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n'+str(i + 1)+' '+str(round(best,6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(i + 1))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt', population[np.argmax(f)])

        # saves file with the best population
        np.savetxt(experiment_name+'/bestpopulation.txt', populations[best_f_idx])

        population, f = ea.step(population, f)
        populations.append(population)
        fitness.append(f)

        if f.max() > f_best[-1]:
            f_best.append(f.max())
            best_f_idx = i
        else:
            f_best.append(f_best[-1])
    print("FINISHED!")

    
    def simulation(env,x):
        f,p,e,t = env.play(pcont=x)
        return f

    test_env = NewFitnessEnvironment(
                    experiment_name="EA1",
                    enemies=[1,2,3,4,5,6,7,8],
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    with open("EA1/best.txt", "r") as ff:
        arr = []
        for i in range(265):
            arr.append(np.float64(ff.readline().strip()))
            # print(arr[i])

        pcont = np.array(arr)
        fit, player_hp, enemy_hp, t = test_env.play(pcont = pcont)

    print("Gain:", sum(player_hp) - sum(enemy_hp))
    print(f"Defeated enemies: {enemy_hp.count(0)}/8")
    # print(f"Sum of player life: {round(sum(player_hp),0)}/800")
    # print("Sum of time:", sum(t))


    '''
    with open("EA1/best.txt", "r") as ff:
        # population = np.array([list(map(lambda num: np.float64(num), line.split())) for line in ff.read().split("\n")][0:90])
        arr = []
        for i in range(265):
            arr.append(np.float64(ff.readline().strip()))

        pcont = np.array(arr)
        # index = np.argmax(np.array(list(map(lambda y: simulation(test_env, y), population))))
        fit, player_hp, enemy_hp, t = test_env.play(pcont = pcont)
        append_file(file_to_write, enemy_hp.count(0), enemies)
    '''
    

    if show_plot:
        plt.plot(best_f)
        plt.plot(std_f)
        plt.plot(mean_f)
        plt.legend(["best", "std", "mean"])
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title = f"Enemies: {enemies}"
        plt.show()

    return f_best[-1]

# def optuna_objective(trial):
#     population_size = trial.suggest_int("population_size", 20, 150)
#     num_generations = trial.suggest_int("num_generations", 10, 150)
#     mutation_prob = trial.suggest_float("mutation_prob", 0, 1)
#     mutation_size_1 = trial.suggest_float("mutation_size_1", 0, 1)
#     mutation_size_2 = trial.suggest_float("mutation_size_2", 0, 1)
#     tournament_size = trial.suggest_int("tournament_size", 2, 20)
#     enemies = [2,5,6]
#     print(f'population_size: {population_size}')
#     print(f'num_generations: {num_generations}')
#     print(f'mutation_prob: {mutation_prob}')
#     print(f'mutation_size_1: {mutation_size_1}')
#     print(f'mutation_size_2: {mutation_size_2}')
#     print(f'tournament_size: {tournament_size}')
#     return run_EA(population_size,num_generations,mutation_prob, mutation_size_1,mutation_size_2,tournament_size,enemies)

# def optuna_optimization():
#     # storage = optuna.storages.InMemoryStorage()
#     storage = 'sqlite:///C:/Users/thaom/Documents/School/VU/master/evolutionary computing/optuna/db.db'
#     study = optuna.create_study(direction=StudyDirection.MAXIMIZE,storage=storage)
#     study.optimize(optuna_objective, n_trials=100)
#     # run_server(storage)
#     print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

if __name__ == '__main__':
    population_size = 90
    num_generations = 70
    mutation_prob = 0.5467432719648073
    mutation_size_1 = 0.13551215453840582
    mutation_size_2 = 0.12234388084305348
    generation = 0
    tournament_size = 8
    enemies = [[3,6], [5,6], [2, 5, 6, 8], [2, 3, 5, 6, 8], [2, 4, 5, 6, 8]]
    enemy_index = 1

    run_EA(population_size,num_generations,mutation_prob,mutation_size_1,mutation_size_2,tournament_size,enemies[0],True)

    '''
    enemy_conf = list(powerset([2,3,4,5,6,8]))
    enemy_conf.remove([2])
    enemy_conf.remove([3])
    enemy_conf.remove([4])
    enemy_conf.remove([5])
    enemy_conf.remove([6])
    enemy_conf.remove([8])
    enemy_conf.remove([])
    # print(enemy_conf)

    index = 1
    for enemies in enemy_conf:
        print(f"\n{index}. run / 57\tenemies: {enemies}")
        run_EA(population_size,num_generations,mutation_prob,mutation_size_1,mutation_size_2,tournament_size,enemies,False)
        index += 1
    '''
    # optuna_optimization()
# 2 4 4

