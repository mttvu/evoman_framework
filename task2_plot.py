import sys
import matplotlib.pyplot as plt

from evoman.environment import Environment
from demo_controller import player_controller


import numpy as np
import os
import math
import pandas as pd

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

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
    

def plot_gains():
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

    df = pd.DataFrame(columns=['ea','enemies','gain'])

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
                    print(f"ea: {ea}, run: {run}, enemies: {enemy_list}")
                    print(f"Defeated enemies: {enemy_hp.count(0)}/8")

                    df = pd.concat([df, pd.DataFrame([{'ea': ea,'enemies': enemy_list,'gain': (sum(player_hp) - sum(enemy_hp))/8}])], ignore_index=True)
                    
            print("\n\n\n")


    # Create a boxplot using Plotly Express
    fig = px.box(df, x='ea', y='gain', facet_col='enemies', title='Boxplot of Gain for Different "ea" Values', labels={'ea': 'EA', 'gain': 'average gain'})
    fig.show()
    # Perform a t-test
    t_stat, p_value = stats.ttest_ind(df[df["ea"] == 1]["gain"], df[df["ea"] == 2]["gain"])

    # Check if the p-value is less than the significance level (e.g., 0.05)
    alpha = 0.05
    print(p_value)
    if p_value < alpha:
        print("The differences in means are statistically significant.")
    else:
        print("The differences in means are not statistically significant.")


def read_results(ea,enemies):
    data = []

    # Flag to check if we are within a section
    in_section = False

    # Iterate through the lines and extract data
    for line in open(f"results/EA{ea}_{enemies}.txt"):
        if line.startswith("gen best mean std"):
            if in_section:
                data.append(section_data)
            section_data = []
            in_section = True
        elif in_section:
            section_data.append(line.split())

    # Append the last section if it exists
    if in_section:
        data.append(section_data)

    # Flatten the data list and add section numbers
    flattened_data = []
    for i, section in enumerate(data, start=1):
        for row in section:
            flattened_data.append([i] + row)

    # Create a DataFrame
    df = pd.DataFrame(flattened_data, columns=["section", "gen", "best", "mean","std" ])

    df = df.dropna()
    df["section"] = df["section"].astype(int)
    df["gen"] = df["gen"].astype(int)
    df["best"] = df["best"].astype(float)
    df["std"] = df["std"].astype(float)
    df["mean"] = df["mean"].astype(float)
    return df
def plot10runs(enemies):

    ea1 = read_results(1,enemies)
    ea2 = read_results(2,enemies)

    
    # Calculate the overall average and standard deviation of mean fitness for ea1 and ea2
    mean_df_ea1 = ea1.groupby(['gen'])['mean'].mean().reset_index()
    std_df_ea1 = ea1.groupby(['gen'])['mean'].std().reset_index()
    mean_df_ea2 = ea2.groupby(['gen'])['mean'].mean().reset_index()
    std_df_ea2 = ea2.groupby(['gen'])['mean'].std().reset_index()

    # Calculate the overall average of maximum fitness for ea1 and ea2
    max_df_ea1 = ea1.groupby(['gen'])['best'].mean().reset_index()
    max_df_ea2 = ea2.groupby(['gen'])['best'].mean().reset_index()

    # Create a figure with subplots
    fig = go.Figure()

    # Add lines for average mean fitness for ea1 and ea2
    fig.add_trace(go.Scatter(x=mean_df_ea1['gen'], y=mean_df_ea1['mean'], mode='lines', name='EA1 avg', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=mean_df_ea2['gen'], y=mean_df_ea2['mean'], mode='lines', name='EA2 avg', line=dict(color='black')))

    # Add scatter plots with smoothed lines for the standard deviation without a legend
    fig.add_trace(go.Scatter(x=mean_df_ea1['gen'], y=mean_df_ea1['mean'] + std_df_ea1['mean'], mode='lines', fill=None, line=dict(width=0), line_shape='spline', showlegend=False))
    fig.add_trace(go.Scatter(x=mean_df_ea1['gen'], y=mean_df_ea1['mean'] - std_df_ea1['mean'], mode='lines', fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), line_shape='spline', showlegend=False))
    fig.add_trace(go.Scatter(x=mean_df_ea2['gen'], y=mean_df_ea2['mean'] + std_df_ea2['mean'], mode='lines', fill=None, line=dict(width=0), line_shape='spline', showlegend=False))
    fig.add_trace(go.Scatter(x=mean_df_ea2['gen'], y=mean_df_ea2['mean'] - std_df_ea2['mean'], mode='lines', fill='tonexty', fillcolor='rgba(0,0,0,0.2)', line=dict(width=0), line_shape='spline', showlegend=False))

    # Add lines for average maximum fitness for ea1 and ea2
    fig.add_trace(go.Scatter(x=max_df_ea1['gen'], y=max_df_ea1['best'], mode='lines', name='EA1 max', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=max_df_ea2['gen'], y=max_df_ea2['best'], mode='lines', name='EA2 max', line=dict(color='purple')))

    # Customize the layout
    enemies_str = ','.join(str(digit) for digit in str(enemies))
    fig.update_layout(
        title=f'enemies=[{enemies_str}]',
        xaxis_title='Generation',
        yaxis_title='Fitness',
        width=500,
    height=400,
    )
    fig.update_layout(yaxis_range=[0,80])

    # Show the plot
    fig.show()



# plot10runs(368)
plot_gains()