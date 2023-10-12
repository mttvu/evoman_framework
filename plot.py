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

data = []
boxplot_data = []
for ea in range(1,3):
    data.append([])
    for enemy in range(1,4):
        env = Environment(experiment_name=f"EA{ea}",
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        gains = []
        
        for run in range(1,11):
            with open(f"solution_files/EA{ea}/Enemy{enemy}/best_run{run}.txt", "r") as ff:
                population = np.array([list(map(lambda num: np.float64(num), line.split())) for line in ff.read().split("\n")][0:30])
                index = np.argmax(np.array(list(map(lambda y: simulation(env, y), population))))
                f, player_hp, enemy_hp, t = env.play(pcont = population[index])
                gains.append(player_hp - enemy_hp)

        data[ea - 1].append(gains)
        boxplot_data.append(gains)
    # exit()



print(boxplot_data)
fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(boxplot_data, patch_artist = True)
 
colors = ['#311432', '#710193', '#B65FCF', '#32612D', '#028A0F', '#74B72E']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

 
# changing color and linewidth of
# caps
# for cap in bp['caps']:
#     cap.set(color ='#8B008B',
#             linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='#E3242B',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax.set_xticklabels(['EA1 Enemy1', 'EA1 Enemy2', 'EA1 Enemy3', 'EA2 Enemy1', 'EA2 Enemy2', 'EA2 Enemy3'])
 
# Adding title
plt.title("Individual gain of EAs vs enemies 1-3")
 
# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
     
# show plot
plt.show()


# plt.boxplot(boxplot_data)
# plt.show()
