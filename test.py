import sys
import matplotlib.pyplot as plt

from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import decimal

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

class NewFitnessEnvironment(Environment):
    def fitness_single(self):
        hp = self.get_playerlife()
        enemy = self.get_enemylife()
        time = self.get_time()
        a = 0.3
        b = 0.7
        return (a*hp + b*(100-enemy))/(a+b)
    
    def cons_multi(self, values): # [1,3,6,8]
        return values.mean()

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

class TestEnvironment(Environment):

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

# env = TestEnvironment(
env = NewFitnessEnvironment(
                    experiment_name="EA2",
                    # enemies=[3,4,5,7],
                    enemies=[1,2,3,4,5,6,7,8],
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

with open("EA2/best.txt", "r") as ff:
    arr = []
    for i in range(265):
        arr.append(np.float64(ff.readline().strip()))
        # print(arr[i])

    pcont = np.array(arr)
    f, player_hp, enemy_hp, t = env.play(pcont = pcont)
    print(f)    
    print(player_hp)
    print(enemy_hp)

print("Gain:", sum(player_hp) - sum(enemy_hp))
print(f"Defeated enemies: {enemy_hp.count(0)}/8")
print(f"Sum of player life: {round(sum(player_hp),0)}/800")
print("Sum of time:", sum(t))