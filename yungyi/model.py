import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


class Agent:
    def __init__(self, race, income):
        self.race = race
        self.income = income

class Grid:
    def __init__(self, size, race_count=4, income_levels=5, threshold=0.5, income_tolerance=0.25, density=0.85):
        self.size = size
        self.race_count = race_count
        self.income_levels = income_levels
        self.threshold = threshold
        self.income_tolerance = income_tolerance
        self.density = density
        self.grid = np.full((size, size), None)
        self.populate_grid()

    def populate_grid(self):
        total_cells = self.size * self.size
        num_agents = int(total_cells * self.density)
        random.seed(int(time.time()))
        # Generate agent races uniformly
        races = np.random.choice(self.race_count, size=num_agents)
        
        self.income_levels = [15000, 30000, 50000, 80000, 120000]
        # Conditional income distribution by race
        race_income_probs = {
            0: [0.1, 0.2, 0.4, 0.2, 0.1],
            1: [0.2, 0.3, 0.3, 0.1, 0.1],
            2: [0.3, 0.3, 0.2, 0.1, 0.1],
            3: [0.1, 0.2, 0.3, 0.2, 0.2],
        }

        # Create agents list
        agents = []
        for race in races:
            income_bracket = np.random.choice(len(self.income_levels), p=race_income_probs[race])
            income = self.income_levels[income_bracket]
            agents.append(Agent(race, income))

        # Create full cell list: agents + None (empty spots)
        cells = agents + [None] * (total_cells - num_agents)

        # Shuffle the combined list so agents and empties are randomly distributed
        
        random.shuffle(cells)
        
        # Assign cells to grid in row-major order
        for idx, cell in enumerate(cells):
            x = idx // self.size
            y = idx % self.size
            self.grid[x][y] = cell
        

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    neighbor = self.grid[nx][ny]
                    if neighbor:
                        neighbors.append(neighbor)
        return neighbors

    def is_satisfied(self, x, y):
        agent = self.grid[x][y]
        neighbors = self.get_neighbors(x, y)
        if not neighbors:
            return True
        similar = sum(1 for n in neighbors if n.race == agent.race)
        return (similar / len(neighbors)) >= self.threshold

    def find_empty_cells(self):
        return [(x, y) for x in range(self.size) for y in range(self.size) if self.grid[x][y] is None]

    def move_unsatisfied_agents(self):

        unsatisfied = [(x, y) for x in range(self.size) for y in range(self.size)
                    if self.grid[x][y] and not self.is_satisfied(x, y)]

        empty_cells = self.find_empty_cells()
        random.shuffle(empty_cells)  
        #print(empty_cells)
        planned_moves = []
        used_targets = set()

        for x, y in unsatisfied:
            agent = self.grid[x][y]
            for new_x, new_y in empty_cells:
                if (new_x, new_y) in used_targets:
                    continue
                neighbors = self.get_neighbors(new_x, new_y)
                if not neighbors:
                    continue
                neighbor_avg_income = np.mean([n.income for n in neighbors])

                # Allow moving if agent income is ≥ threshold (not less than)
                if agent.income >= neighbor_avg_income * (1 - self.income_tolerance):
                    planned_moves.append(((x, y), (new_x, new_y)))
                    used_targets.add((new_x, new_y))
                    break
        
        for (old_x, old_y), (new_x, new_y) in planned_moves:
            self.grid[new_x][new_y] = self.grid[old_x][old_y]
            self.grid[old_x][old_y] = None


    def all_agents_satisfied(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x][y] and not self.is_satisfied(x, y):
                    return False
        return True

    def calculate_segregation_metrics(self):
        total_agents = 0
        same_race_neighbors = 0
        race_counts = {r: 0 for r in range(self.race_count)}
        neighborhood_counts = {r: 0 for r in range(self.race_count)}

        for x in range(self.size):
            for y in range(self.size):
                agent = self.grid[x][y]
                if not agent:
                    continue
                total_agents += 1
                race_counts[agent.race] += 1
                neighbors = self.get_neighbors(x, y)
                same_race_count = sum(1 for n in neighbors if n.race == agent.race)
                same_race_neighbors += same_race_count
                neighborhood_counts[agent.race] += len(neighbors)

        avg_homophily = same_race_neighbors / max(1, sum(neighborhood_counts.values()))
        dissimilarity_sum = 0
        for r1 in range(self.race_count):
            for r2 in range(r1 + 1, self.race_count):
                A = race_counts[r1]
                B = race_counts[r2]
                if A == 0 or B == 0:
                    continue
                D_local_sum = 0
                for x in range(self.size):
                    for y in range(self.size):
                        neighbors = self.get_neighbors(x, y)
                        if not neighbors:
                            continue
                        a_i = sum(1 for n in neighbors if n.race == r1)
                        b_i = sum(1 for n in neighbors if n.race == r2)
                        D_local_sum += abs(a_i / A - b_i / B)
                D = 0.5 * D_local_sum / total_agents
                dissimilarity_sum += D

        num_pairs = self.race_count * (self.race_count - 1) / 2
        avg_dissimilarity = dissimilarity_sum / max(1, num_pairs)
        return {'average_homophily': avg_homophily, 'average_dissimilarity_index': avg_dissimilarity}

    def render_grid_as_array(self):
        arr = np.full((self.size, self.size), -1)
        for x in range(self.size):
            for y in range(self.size):
                agent = self.grid[x][y]
                if agent is not None:
                    arr[x, y] = agent.race
        return arr
