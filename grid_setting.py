import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb, LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy.spatial.distance import hamming
import random, main
from datetime import datetime


def truncate_colormap(cmap_in, minval=0.2, maxval=1.0, n=100):
    """Truncates a colormap to exclude very light or dark values."""
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap_in.name},{minval:.2f},{maxval:.2f})',
        cmap_in(np.linspace(minval, maxval, n))
    )
    return new_cmap

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


def encode_income_group(quartile):
    return min((quartile - 1) // 6 + 1, 5)


def plot_grid(grid, agents, iteration, segregation, percentage_satisfied,pop_density = 0.5, color_based_on='race'):
    """Plot the grid with color-coded agents."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define discrete colors for race and income categorical
    color_race = {0: 'purple', 1: 'green', 2: 'yellow', 3: 'red'}
    color_income = {1: 'purple', 2: 'green', 3: 'yellow', 4: 'red', 5: 'blue'}
    
    race = ['White', 'Black', 'Asian', 'Latino']

    # Prepare legend patches depending on mode
    if color_based_on == 'race':

        labels = [f"Race {i}" for i in color_race]

        labels = race

        legend_patches = [mpatches.Patch(color=color_race[i], label=labels[i]) for i in color_race]

    elif color_based_on == 'income':
        labels = [f"Income Quartile {i}" for i in color_income]
        legend_patches = [mpatches.Patch(color=color_income.get(i, 'grey'), label=labels[i-1]) for i in color_income]

    elif color_based_on == 'income_intensity':
        # Continuous blue shades for income quartiles 1 to 5
        norm = mcolors.Normalize(vmin=1, vmax=5)
        cmap_income = truncate_colormap(cm.get_cmap('Greens'), minval=0.25, maxval=1.0)
        labels = [f"Income Quartile {i}" for i in range(1,6)]
        legend_patches = [mpatches.Patch(color=cmap_income(norm(i)), label=labels[i-1]) for i in range(1,6)]


    elif color_based_on == 'race_with_schools':
        labels = race
        legend_patches = [mpatches.Patch(color=color_race[i], label=labels[i]) for i in color_race]
        legend_patches.append(mpatches.Patch(color='black', label="Good School"))
        legend_patches.append(mpatches.Patch(color='gray', label="Bad School"))


    else:
        raise ValueError(f"Unsupported color basis: {color_based_on}")

    image = np.ones((main.L, main.W, 3))  # Start with all white

    for i in range(main.L):
        for j in range(main.W):
            if grid[i, j] == 0:

                if color_based_on == 'race_with_schools':
                    if (i, j) in main.good_school:
                        image[i, j] = to_rgb('black')  # Black for good schools
                    elif (i, j) in main.bad_school:
                        image[i, j] = to_rgb('gray')  # Gray for bad schools

                continue  # Leave as white (empty)
            else:
                agent = agents[grid[i, j]]
                if color_based_on == 'race':
                    color = color_race.get(agent.race, 'grey')
                    rgb = to_rgb(color)

                elif color_based_on == 'race_with_schools':
                    if (i, j) in main.good_school:
                        color = 'black'  # Black for good schools
                    elif (i, j) in main.bad_school:
                        color = 'gray'  # Gray for bad schools
                        
                    else:
                        color = color_race.get(agent.race, 'grey')
                    rgb = to_rgb(color)

                elif color_based_on == 'income':
                    color = color_income.get(agent.starting_income_quartile, 'grey')
                    rgb = to_rgb(color)
                elif color_based_on == 'income_intensity':
                    # Map income quartile to blue shade intensity

                    income_level = agent.starting_income_quartile
                    rgb = cmap_income(norm(income_level))[:3]  # ignore alpha channel

                    income_level = encode_income_group(agent.starting_income_quartile)
                    
                    rgb = cmap_income(norm(income_level))[:3]  # ignore alpha channel
                

                image[i, j] = rgb

    ax.imshow(image)
    ax.set_title(f"Iteration {iteration},Population Density {pop_density}, Homophilly {round(segregation, 3)}, Ratio Satisfied {round(percentage_satisfied, 3)}")
    ax.axis('off')
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.0, 0.5), borderaxespad=0.)

    return fig
