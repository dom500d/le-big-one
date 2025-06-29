import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random
import grid_setting, main
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime
from scipy.stats import truncnorm
import math

#from line_profiler import profile

class RaceType:
    pass

class IncomeType:
    pass

class AttrType:
    pass


class Agent:
    def __init__(self, race, income, starting_income_quartile, starting_position, id):
        self.race = race
        self.income = income
        self.starting_income_quartile = starting_income_quartile
        self.pos = starting_position
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
    
class PropertyGenerator:
    def __init__(self, race_income_probs: dict, race_gen: RaceGenerator):
        self.race_income_probs = race_income_probs
        self.race_gen = race_gen
        self.num_races = len(race_income_probs.keys())

    def get_counts(self, num_agents):
        ret = []
        incomes = list(range(1, len(self.race_income_probs[0]) + 1))
        porportions = self.race_gen.get_counts(num_agents)
        races = list(range(0, self.num_races))
        race_list = [attr for attr, count in zip(races, porportions) for _ in range(count)]
        random.shuffle(race_list)
        diff = num_agents-len(race_list)
        race_list.extend([0] * diff)
      
        for i in range(num_agents):
            race = race_list[i]
            probs = self.race_income_probs[race]
            income_percentile = random.choices(incomes, weights=probs, k=1)[0]
            ret.append((race, income_percentile))
        
        return ret
        
        
    
class Environment:
    def __init__(self, height, width, population_density, property_generator: PropertyGenerator, income_difference_threshold):
        self.height = height
        self.width = width
        self.density = population_density
        self.num_agents = int(height * width * population_density)
        """Initialize grid with agents and attributes."""
        self.grid = np.zeros((main.L, main.W), dtype=int)  # 0 for empty
        self.agent_positions = random.sample([(i, j) for i in range(main.L) for j in range(main.W)], self.num_agents)
        self.agents = {}
        self.open_spots = []
        self.income_difference_threshold = income_difference_threshold
        agent_id = 1
        
        # Generate race/income distributions
        race_income_list = property_generator.get_counts(self.num_agents)
        diff = self.num_agents-len(race_income_list)
        race_income_list.extend([(0, 1)] * diff)
        random.shuffle(race_income_list)
        
        for i in range(0, self.num_agents):
            race_and_income = race_income_list[i]
            pos = self.agent_positions.pop()
            self.grid[pos] = agent_id
            self.agents[agent_id] = Agent(race=race_and_income[0], income=10, starting_income_quartile=race_and_income[1], starting_position=pos, id=agent_id)
            agent_id += 1
            
        for i in range(height):
            for j in range(width):
                if self.grid[i, j] == 0:
                    self.open_spots.append((i, j))
        
    #@profile
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
                    neighbors.append(self.agents[id])
        return neighbors
    
    #@profile
    def compute_similarity(self, attr1, attr2):
        """Compute similarity using Hamming distance."""
        return 1.0 if attr1 == attr2 else 0.0

    #@profile
    def is_satisfied(self, agent: Agent, neighbors: list[Agent], tau_u, tau_s):
        """Check if agent is satisfied based on utility and similarity thresholds."""
        if not neighbors:
            return True  # No neighbors, satisfied
        similar_neighbors = 0
        for neighbor in neighbors:
            if self.compute_similarity(agent.race, neighbor.race) >= tau_s:
                similar_neighbors += 1
        theta = similar_neighbors / len(neighbors)

        #return theta >= tau_u
    



        if main.relaxation_applied:
            # Fast zone lookup using precomputed sets
            pos = agent.pos
            in_good_school_range = pos in main.good_school_zone
            in_bad_school_range = pos in main.bad_school_zone

            # Modify theta based on school proximity
            if in_good_school_range:
                theta += 0.3
            if in_bad_school_range:
                theta -= 0.3


        # Ensure theta is in [0, 1]
        theta = np.clip(theta, 0, 1)

        return theta >= tau_u
    



    def compute_segregation(self, type):
        """Compute segregation level as sum of identical neighbors."""
        #segregation = 0
        if isinstance(type, RaceType):
            total_links = 0
            same_type_links = 0

            for agent in self.agents.values():
                neighbors = self.get_neighbors(agent)
                for neighbor in neighbors:
                    total_links += 1
                    if agent.race == neighbor.race:
                        same_type_links += 1

            segregation = same_type_links / total_links if total_links > 0 else 0.0
            
        '''elif isinstance(type, IncomeType):
            segregation = 0
            for agent in self.agents.values():
                neighbors = self.get_neighbors(agent)
                for neighbor in neighbors:
                    if agent.starting_income_quartile == neighbor.starting_income_quartile:
                        segregation += 1
        elif isinstance(type, AttrType):
            segregation = 0
            for agent in self.agents.values():
                neighbors = self.get_neighbors(agent)
                for neighbor in neighbors:
                    if agent.attributes == neighbor.attributes:
                        segregation += 1
        else:
            raise(TypeError("This type of segregation isn't supported."))'''
        
        return segregation
    #@profile
    def find_vacant_spot(self, agent: Agent, tau_u, tau_s):
        """Find nearest vacant spot where agent would be satisfied."""
        i, j = agent.pos
        original_pos = (i, j)
        for index, (ni, nj) in enumerate(self.open_spots):
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
            agent.pos = original_pos

            if satisfied and has_money:
                self.open_spots.pop(index)
                self.open_spots.append(original_pos)
                return (ni, nj)
        # for r in range(1, max(self.width, self.height)):
        #     for di in range(-r, r + 1):
        #         for dj in range(-r, r + 1):
        #             if abs(di) != r and abs(dj) != r:
        #                 continue
        #             ni, nj = i + di, j + dj
        #             if 0 <= ni < self.height and 0 <= nj < self.width and self.grid[ni, nj] == 0:
        #                 # Save current state
        #                 original_id = agent.id
        #                 self.grid[i, j] = 0
        #                 self.grid[ni, nj] = original_id
        #                 agent.pos = (ni, nj)

        #                 neighbors = self.get_neighbors(agent)
                        
        #                 satisfied = self.is_satisfied(agent, neighbors, tau_u, tau_s)
        #                 has_money = self.can_move(agent, neighbors, self.income_difference_threshold)
        #                 # Revert changes
        #                 self.grid[ni, nj] = 0
        #                 self.grid[i, j] = original_id
        #                 agent.pos = original_pos

        #                 if satisfied and has_money:
        #                     return (ni, nj)
        return None

    def can_move(self, agent: Agent, neighbors: list[Agent], income_difference_treshold):
        if not neighbors:
            return True
        income_avg = 0
        for neighbor in neighbors:
            income_avg += neighbor.starting_income_quartile
        income_avg = income_avg / len(neighbors)
        if agent.starting_income_quartile + income_difference_treshold >= income_avg:
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
        for agent in self.agents.values():
            neighbors = self.get_neighbors(agent)
            if not self.is_satisfied(agent, neighbors, tau_u, tau_s):
                sad.append(agent)
        return sad
    
def get_school_range(school_positions):
    range_set = set()
    for x, y in school_positions:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < main.L and 0 <= ny < main.W:  # keep within bounds
                    range_set.add((nx, ny))
    return range_set

#@profile
def simulate(height, width, population_density, race_income: PropertyGenerator, income_difference_threshold, tau_u, tau_s, max_iter=10000, segregation_type=RaceType(), break_early=True):
    """Run the simulation for the extended Schelling model."""
    env = Environment(height, width, population_density, race_income, income_difference_threshold)
    race_frames = []
    income_frames = []
    race_w_school_frames = []
    iteration = 0
    un_over_t = []
    seg_over_t = []
    money_increase = []
    percentage_sat = 0
    num_agents = int(height*width*population_density)
    orig_tau_u = tau_u
    orig_tau_s = tau_s
    
    
    #relaxation_applied = False  # To ensure relaxation is only applied once unless you want to do it repeatedly

    while iteration < max_iter:
        segregation = env.compute_segregation(segregation_type)
        seg_over_t.append(segregation)
        fig = grid_setting.plot_grid(env.grid, env.agents, iteration, segregation, percentage_sat,population_density,color_based_on='race' )
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        race_frames.append(img)
        plt.close(fig)
        
        fig = grid_setting.plot_grid(env.grid, env.agents, iteration, segregation,percentage_sat,population_density, color_based_on='income_intensity' )
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        income_frames.append(img)
        plt.close(fig)

        if main.relaxation_applied:
            fig = grid_setting.plot_grid(env.grid, env.agents, iteration, segregation, percentage_sat, color_based_on='race_with_schools')
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            race_w_school_frames.append(img)
            plt.close(fig)

        
        unsatisfied = env.get_unsatisfied_agents(tau_u, tau_s)
        un_over_t.append(len(unsatisfied))
        moved_any = False
        if not unsatisfied:
            print("There are no unsatisfied agents.")
            break
        random.shuffle(unsatisfied)
        
        for agent in unsatisfied:
            vacant = env.find_vacant_spot(agent, tau_u, tau_s)
            if vacant is not None:
                env.move_agent(agent, vacant)
                moved_any = True

        if not moved_any:
            if not main.relaxation_applied and break_early:
                all_positions = [(i, j) for i in range(main.L) for j in range(main.W)]
                for i in range((main.L * main.W)//1000):
                    agent_positions = random.sample(all_positions, 1) 
                    main.good_school.append(agent_positions[0])  
                    #print(f"good school:{agent_positions[0]}")
                    
                    agent_positions = random.sample(all_positions, 1) 
                    main.bad_school.append(agent_positions[0])  
                    #print(f"bad school:{agent_positions[0]}")
                    
                main.good_school_zone = get_school_range(main.good_school)
                main.bad_school_zone = get_school_range(main.bad_school)

                print("No moves possible, increasing da money and lowering the thresholds.")
                money_increase.append(iteration)

                # Relax constraints

                env.income_difference_threshold = env.income_difference_threshold + 3

                tau_u = tau_u - 0.3  # Decrease satisfaction threshold
                tau_s = tau_s - 0.3
                main.relaxation_applied = True


                # Continue from the current state
                continue

            elif break_early:
                print("No movement and relaxation already applied, breaking.")
                break
            else:
                print("No movement, increasing income threshold.")
                #env.income_difference_threshold += 1
                money_increase.append(iteration)
        percentage_sat = (num_agents-len(unsatisfied))/num_agents
        iteration += 1
    
    unsatisfied = env.get_unsatisfied_agents(tau_u, tau_s)
    un_over_t.append(len(unsatisfied))
    percentage_sat = (num_agents-len(unsatisfied))/num_agents #Calculate final satisfaction

    

    segregation = env.compute_segregation(segregation_type)
    seg_over_t.append(segregation)
    fig = grid_setting.plot_grid(env.grid, env.agents, iteration, segregation,percentage_sat,population_density, color_based_on='race')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    race_frames.append(img)
    plt.close(fig)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    race_frames[0].save(
        f"./simulation_results/model_race_{height}_{width}_{population_density}_{income_difference_threshold}_{orig_tau_u}_{orig_tau_s}_{timestamp}.gif",        # Output filename
        format='GIF',
        save_all=True,
        append_images=race_frames[1:],
        duration=300,           # Duration per frame in ms
    )
    
    fig = grid_setting.plot_grid(env.grid, env.agents, iteration, segregation,percentage_sat,population_density, color_based_on='income_intensity')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    income_frames.append(img)
    plt.close(fig)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    income_frames[0].save(
        f"./simulation_results/model_income_{height}_{width}_{population_density}_{income_difference_threshold}_{orig_tau_u}_{orig_tau_s}_{timestamp}.gif",        # Output filename
        format='GIF',
        save_all=True,
        append_images=income_frames[1:],
        duration=300,           # Duration per frame in ms
    )

    fig = grid_setting.plot_grid(env.grid, env.agents, iteration, segregation, percentage_sat, color_based_on='race_with_schools')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    race_w_school_frames.append(img)
    plt.close(fig)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    race_w_school_frames[0].save(
        f"./simulation_results/model_race_with_schools_{height}_{width}_{population_density}_{income_difference_threshold}_{orig_tau_u}_{orig_tau_s}_{timestamp}.gif",        # Output filename
        format='GIF',
        save_all=True,
        append_images=race_w_school_frames[1:],
        duration=300,           # Duration per frame in ms
    )


    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    # --- Plot 1: Unsatisfied Agents Over Time ---
    iteration = list(range(len(un_over_t)))
    axs[0].plot(iteration, un_over_t, label="Unsatisfied Agents", color="blue")
    for i in money_increase:
        if i < len(un_over_t):
            axs[0].scatter(i, un_over_t[i], color='red', zorder=5, label='Income Threshold Increased' if i == money_increase[0] else "")

    axs[0].set_ylabel("Unsatisfied Count")
    axs[0].set_title("# of Unsatisfied Agents Over Iterations")
    axs[0].legend()
    axs[0].grid(True)

    # Plottinv
    axs[1].plot(iteration, seg_over_t, color='green', label="Homophilly")

    for i in money_increase:
        if i < len(seg_over_t):
            axs[1].scatter(i, seg_over_t[i], color='red', zorder=5)

    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Homophilly")
    axs[1].set_title("Homophilly Over Iterations")
    axs[1].legend()
    axs[1].grid(True)

    plt.savefig(f"./simulation_results/unsatisfied_over_t_{height}_{width}_{population_density}_{income_difference_threshold}_{orig_tau_u}_{orig_tau_s}_{timestamp}.png") 
    return un_over_t
    