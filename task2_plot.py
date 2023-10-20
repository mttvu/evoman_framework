import sys
import matplotlib.pyplot as plt

from evoman.environment import Environment
from demo_controller import player_controller


import numpy as np
import os
import math


os.environ["SDL_VIDEODRIVER"] = "dummy"


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
    
n_hidden_neurons = 10
env_overall = NewFitnessEnvironment(
                    experiment_name="plot",
                    enemies=[1,2,3,4,5,6,7,8],
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


for ea in [1,2]:
    for enemy_index, enemy_list in enumerate(["[2,3,5,6,8]", "[3,6,8]"]):
        for run in range(1,11):
            with open(f"solution_files/TASK2/EA{ea}/{enemy_list}/run{run}.txt") as file:
                arr = []
                for i in range(265):
                    arr.append(np.float64(file.readline().strip()))
                pcont = np.array(arr)
                
                fitness, player_hp, enemy_hp, t = env_overall.play(pcont = pcont)
                # fitness is the return value of multi fitness function (in this case mean(fitness))
                # the other values are lists of values

                print(fitness)
                print(player_hp)
                print(enemy_hp)

                print("Gain:", sum(player_hp) - sum(enemy_hp))
                print(f"Defeated enemies: {enemy_hp.count(0)}/8")

                print(f"Sum of player life: {round(sum(player_hp),0)}/800")
                print("Sum of time:", sum(t))
        print("\n\n\n")