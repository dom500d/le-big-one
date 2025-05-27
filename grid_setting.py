import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import hamming
import random, main
from datetime import datetime

def initialize_grid(num_agents, num_attributes):
    """Initialize grid with agents and attributes."""
    grid = np.zeros((main.L, main.W), dtype=int)  # 0 for empty
    agent_positions = random.sample([(i, j) for i in range(main.L) for j in range(main.W)], num_agents)
    agents = []
    agent_id = 1
    
    # Generate possible attribute combinations
    if num_attributes == 2:
        attributes_list = [(a1, a2) for a1 in [1, 2] for a2 in [1, 2]]
    else:
        attributes_list = [(a1, a2, a3) for a1 in [1, 2] for a2 in [1, 2] for a3 in [1, 2]]
    
    # Assign attributes to agents evenly
    agents_per_type = num_agents // len(attributes_list)
    for attr in attributes_list:
        for _ in range(agents_per_type):
            if agent_positions:
                pos = agent_positions.pop()
                grid[pos] = agent_id
                agents.append({'id': agent_id, 'pos': pos, 'attributes': attr})
                agent_id += 1
    
    
    return grid, agents
def plot_grid(grid, agents, iteration, segregation, color_based_on='race'):
    """Plot the grid with color-coded agents."""
    fig, ax = plt.subplots(figsize=(10, 8))
    race_labels = [f"Race {i}" for i in range(4)]
    color_race = {0: 'purple', 1: 'green', 2: 'yellow', 3: 'red', 4: 'blue', 5: 'black', 6: 'orange'}
    race_patches = [mpatches.Patch(color=color_race[i], label=race_labels[i]) for i in range(4)]
    
    income_labels = [f"Income Quartile {i}" for i in range(1, 7)]
    color_income = color_race
    income_patches = [mpatches.Patch(color=color_income[i], label=income_labels[i]) for i in range(6)]

    image = np.zeros((main.L, main.W, 3))
    for i in range(main.L):
        for j in range(main.W):
            if grid[i, j] == 0:
                image[i, j] = [1, 1, 1]  # White for empty
            else:
                agent = agents[grid[i, j]]
                color = 'green'
                if color_based_on == 'race':
                    color = color_race[agent.race]
                    ax.legend(handles=race_patches)
                elif color_based_on == 'income':
                    color = color_income[agent.starting_income_quartile]
                    ax.legend(handles=income_patches)
                else:
                    print(f"Color based on {color_based_on} is not currently supported.")
                # color = 'green'
                if color == 'green':
                    image[i, j] = [0, 1, 0]
                elif color == 'yellow':
                    image[i, j] = [1, 1, 0]
                elif color == 'red':
                    image[i, j] = [1, 0, 0]
                elif color == 'blue':
                    image[i, j] = [0, 0, 1]
                elif color == 'orange':
                    image[i, j] = [1, 0.5, 0]
                elif color == 'purple':
                    image[i, j] = [0.5, 0, 0.5]
                elif color == 'grey':
                    image[i, j] = [0.5, 0.5, 0.5]
                elif color == 'cyan':
                    image[i, j] = [0, 1, 1]
    ax.imshow(image)
    ax.set_title(f"Iteration {iteration}, Segregation Level: {segregation}")
    ax.axis('off')
    return fig
    # plt.savefig(f'schelling_agent_{num_attributes}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_iter_{iteration}.png')