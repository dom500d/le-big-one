import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from model import Grid
import numpy as np
import matplotlib.patches as mpatches


def main():
    grid_size = 50
    max_iterations = 30
    density = 0.79
    income_tolerance = 0.25
    threshold = 0.25

    g = Grid(size=grid_size, threshold = threshold, income_tolerance = income_tolerance, density=density)
    show_init = g.render_grid_as_array()

    frames = []
    homophily_history = []
    dissimilarity_history = []
    
    frames.append(show_init.copy())
    for i in range(max_iterations):
        print(f"Iteration {i + 1}")
        g.move_unsatisfied_agents()
        metrics = g.calculate_segregation_metrics()

        homophily_history.append(metrics['average_homophily'])
        dissimilarity_history.append(metrics['average_dissimilarity_index'])

        grid_array = g.render_grid_as_array()
        frames.append(grid_array.copy())
 
        print(f"  Homophily: {metrics['average_homophily']:.3f}, Dissimilarity: {metrics['average_dissimilarity_index']:.3f}")

        if g.all_agents_satisfied():
            print("All agents satisfied. Ending early.")
            break

    # Plot segregation metrics over time
    plt.figure(figsize=(8, 4))
    plt.plot(homophily_history, label='Homophily')
    plt.plot(dissimilarity_history, label='Dissimilarity Index')
    plt.xlabel('Iteration')
    plt.ylabel('Segregation Metric')
    plt.legend()
    plt.title('Segregation Metrics Over Time')
    plt.tight_layout()
    plt.show()

     # Create legend handles and labels for race colors + empty
    fig, ax = plt.subplots()
    
    colors = [(1, 1, 1, 1)] + list(plt.cm.tab10.colors[:4])  # white + 4 colors
    cmap = ListedColormap(colors)
    
    # Match data values: -1 -> white, 0-3 -> race colors
    bounds = [-1.5] + [i - 0.5 for i in range(4)] + [3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    labels = ['EMPTY', 'Race 0', 'Race 1', 'Race 2', 'Race 3']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(5)]
    im = ax.imshow(frames[0], cmap=cmap, norm=norm, origin = 'lower')
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    def update(frame):
        im.set_data(frame)
        return [im]

    plt.title("Agent Race Movement Over Time")
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=500, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
