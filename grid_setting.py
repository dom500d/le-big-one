import numpy as np
import matplotlib.pyplot as plt
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
def plot_grid(grid, agents, num_attributes, iteration, segregation):
    """Plot the grid with color-coded agents."""
    
    # Colors for agent types
    COLORS_2 = { (1,1): 'green', (1,2): 'yellow', (2,1): 'red', (2,2): 'blue' }
    COLORS_3 = {
        (1,1,1): 'green', (1,1,2): 'yellow', (1,2,1): 'red', (1,2,2): 'blue',
        (2,1,1): 'orange', (2,1,2): 'purple', (2,2,1): 'grey', (2,2,2): 'cyan'
    }
    
    plt.figure(figsize=(10, 8))
    image = np.zeros((main.L, main.W, 3))
    for i in range(main.L):
        for j in range(main.W):
            if grid[i, j] == 0:
                image[i, j] = [1, 1, 1]  # White for empty
            else:
                agent = next(a for a in agents if a.id == grid[i, j])
                attr = agent.attributes
                
                color = COLORS_2[attr] if num_attributes == 2 else COLORS_3[attr]
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
    plt.imshow(image)
    plt.title(f"Iteration {iteration}, Segregation Level: {segregation}")
    plt.axis('off')
    plt.savefig(f'schelling_agent_{num_attributes}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_iter_{iteration}.png')
    plt.close()