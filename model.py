
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random
import grid_setting, main
from collections import Counter

class RaceType:
    pass

class IncomeType:
    pass

class AttrType:
    pass


class Agent:
    
    def __init__(self, race, income, starting_income_quartile, starting_position, attributes, id):
        self.race = race
        self.income = income
        self.starting_income_quartile = starting_income_quartile
        self.pos = starting_position
        self.attributes = attributes
        self.id = id

class IncomeGenerator:
    def __init__(self, percentiles):
        self.percentiles = percentiles
        
    def get_counts(self, num_agents):
        ret = []
        for x in self.percentiles:
            ret.append(round(x * num_agents))
            
        return ret
    
class RaceGenerator:
    def __init__(self, percentiles):
        self.percentiles = percentiles
        
    def get_counts(self, num_agents):
        ret = []
        for x in self.percentiles:
            ret.append(round(x * num_agents))
            
        return ret
    
class Environment:
    def __init__(self, height, width, population_density, num_attributes, income: IncomeGenerator, race: RaceGenerator, income_difference_threshold):
        self.height = height
        self.width = width
        self.density = population_density
        self.num_agents = int(height * width * population_density)
        """Initialize grid with agents and attributes."""
        self.grid = np.zeros((main.L, main.W), dtype=int)  # 0 for empty
        self.agent_positions = random.sample([(i, j) for i in range(main.L) for j in range(main.W)], self.num_agents)
        self.agents = []
        self.open_spots = []
        self.income_difference_threshold = income_difference_threshold
        agent_id = 1
        
        # Generate possible attribute combinations
        if num_attributes == 2:
            attributes_list = [(a1, a2) for a1 in [1, 2] for a2 in [1, 2]]
        else:
            attributes_list = [(a1, a2, a3) for a1 in [1, 2] for a2 in [1, 2] for a3 in [1, 2]]
        
        # Generate race/income distributions
        porportions = race.get_counts(self.num_agents)
        races = list(range(1, len(porportions) + 1))
        race_list = [attr for attr, count in zip(races, porportions) for _ in range(count)]
        diff = self.num_agents-len(race_list)
        race_list.extend([1] * diff)
        random.shuffle(race_list)
        

        porportions = income.get_counts(self.num_agents)
        incomes = list(range(1, len(porportions) + 1))
        income_list = [attr for attr, count in zip(incomes, porportions) for _ in range(count)]
        diff = self.num_agents-len(income_list)
        income_list.extend([1] * diff)
        random.shuffle(income_list)
        
        
        agents_per_type = self.num_agents // len(attributes_list)
        listttt = [agents_per_type for _ in range(len(attributes_list))]
        attr_list = [attr for attr, count in zip(attributes_list, listttt) for _ in range(count)]
        diff = self.num_agents-len(attr_list)
        attr_list.extend([(1, 1)] * diff)
        random.shuffle(attr_list)
        
        for i in range(0, self.num_agents):
            pos = self.agent_positions.pop()
            self.grid[pos] = agent_id
            self.agents.append(Agent(race=race_list[i], income=None, starting_income_quartile=income_list[i], starting_position=pos, attributes=attr_list[i], id=agent_id))
            agent_id += 1
            
        for i in range(height):
            for j in range(width):
                if self.grid[i, j] == 0:
                    self.open_spots.append((i, j))
        
    def get_neighbors(self, agent: Agent) -> list[Agent]:
        i, j = agent.pos
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < main.L and 0 <= nj < main.W and self.grid[ni, nj] != 0:
                    id = self.grid[ni, nj]
                    neighbors.append(next((obj for obj in self.agents if obj.id == id), None))
        return neighbors
    
    def compute_similarity(self, attr1, attr2):
        """Compute similarity using Hamming distance."""
        return 1 - hamming(attr1, attr2)

    def is_satisfied(self, agent: Agent, neighbors: list[Agent], tau_u, tau_s):
        """Check if agent is satisfied based on utility and similarity thresholds."""
        if not neighbors:
            return True  # No neighbors, satisfied
        similar_neighbors = 0
        for neighbor in neighbors:
            if self.compute_similarity(agent.attributes, neighbor.attributes) >= tau_s:
                similar_neighbors += 1
        theta = similar_neighbors / len(neighbors)
        return theta >= tau_u
    
    def compute_segregation(self, type):
        """Compute segregation level as sum of identical neighbors."""
        segregation = 0
        if isinstance(type, RaceType):
            segregation = 0
            for agent in self.agents:
                neighbors = self.get_neighbors(agent)
                for neighbor in neighbors:
                    if agent.race == neighbor.race:
                        segregation += 1
        elif isinstance(type, IncomeType):
            segregation = 0
            for agent in self.agents:
                neighbors = self.get_neighbors(agent)
                for neighbor in neighbors:
                    if agent.starting_income_quartile == neighbor.starting_income_quartile:
                        segregation += 1
        elif isinstance(type, AttrType):
            segregation = 0
            for agent in self.agents:
                neighbors = self.get_neighbors(agent)
                for neighbor in neighbors:
                    if agent.attributes == neighbor.attributes:
                        segregation += 1
        else:
            raise(TypeError("This type of segregation isn't supported."))
        
        return segregation
         
    def find_vacant_spot(self, agent: Agent, tau_u, tau_s):
        """Find nearest vacant spot where agent would be satisfied."""
        i, j = agent.pos
        original_pos = (i, j)
        for r in range(1, max(self.width, self.height)):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if abs(di) != r and abs(dj) != r:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.height and 0 <= nj < self.width and self.grid[ni, nj] == 0:
                        # Save current state
                        original_id = agent.id
                        self.grid[i, j] = 0
                        self.grid[ni, nj] = original_id
                        agent.pos = (ni, nj)

                        neighbors = self.get_neighbors(agent)
                        
                        satisfied = self.is_satisfied(agent, neighbors, tau_u, tau_s)
                        has_money = self.can_move(agent, neighbors, self.income_difference_threshold)
                        # Revert changes
                        self.grid[ni, nj] = 0
                        self.grid[i, j] = original_id
                        agent['pos'] = original_pos

                        if satisfied and has_money:
                            return (ni, nj)
        return None

    def can_move(self, agent: Agent, neighbors: list[Agent], income_difference_treshold):
        if not neighbors:
            return True
        income_avg = 0
        for neighbor in neighbors:
            income_avg += neighbor.income
        income_avg = income_avg / len(neighbors)
        if agent.income + income_difference_treshold >= income_avg:
            return True
        else:
            return False
        
    def move_agent(self, agent: Agent, new_position):
        # Here we can remove money if we want to ()
        if self.grid[new_position] == 0:
            self.grid[new_position] = agent.id
            self.grid[agent.pos] = 0
            agent.pos = new_position
        else:
            raise(TypeError("Passed a non-empty spot to move_agent"))
        
    def get_unsatisfied_agents(self, tau_u, tau_s):
        sad = []
        for agent in self.agents:
            neighbors = self.get_neighbors(agent)
            if not self.is_satisfied(agent, neighbors, tau_u, tau_s):
                sad.append(agent)
        return sad


def simulate(height, width, population_density, num_attributes, income: IncomeGenerator, race: RaceGenerator, income_difference_threshold, tau_u, tau_s, max_iter=10000, segregation_type=RaceType()):
    """Run the simulation for the extended Schelling model."""
    env = Environment(height, width, population_density, num_attributes, income, race, income_difference_threshold)
    
    iteration = 0
    while iteration < max_iter:
        segregation = env.compute_segregation(segregation_type)
        if iteration % 100 == 0 or iteration == 0:
            grid_setting.plot_grid(env.grid, env.agents, num_attributes, iteration, segregation)
        unsatisfied = env.get_unsatisfied_agents(tau_u, tau_s)
        if not unsatisfied:
            print("There are no unsatisfied agents.")
            break
        random.shuffle(unsatisfied)
        for agent in unsatisfied:
            vacant = env.find_vacant_spot(agent, tau_u, tau_s)
            if vacant is not None:
                env.move_agent(agent, vacant)
            else:
                print(f"Agent {agent.id} with race: {agent.race}, income {agent.income}, percentile {agent.starting_income_percentile}, at {agent.pos} cannot be moved")
        iteration += 1
        
    segregation = env.compute_segregation(segregation_type)
    grid_setting.plot_grid(env.grid, env.agents, num_attributes, iteration, segregation)
    return iteration, segregation

    