import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random
import grid_setting
from model import simulate, IncomeGenerator, RaceGenerator, PropertyGenerator
L, W = 100, 100 # Grid size
if __name__=="__main__":
#### variable setting 
    # Grid and simulation parameters
      
    POP_DENSITY = 0.7  # 80% population density
    NUM_AGENTS = int(L * W * POP_DENSITY)
    MAX_ITER_2 = 1000  # Max iterations for 2-attribute model
    MAX_ITER_3 = 2000  # Max iterations for 3-attribute model
    TAU_U = 0.25  # Utility threshold (50%)
    TAU_S_2 = 0.25  # Similarity threshold for 2-attribute (25%)

    
### simulation
    # random.seed(42)  # For reproducibility
    # # Run 2-attribute model
    # print("Running 2-attribute model...")
    # iter_2, seg_2 = simulate(2, TAU_U, TAU_S_2, MAX_ITER_2,NUM_AGENTS)
    # print(f"2-attribute model converged in {iter_2} iterations with segregation level {seg_2}")
    
    # # Run 3-attribute model
    # print("Running 3-attribute model...")
    # iter_3, seg_3 = simulate(3, TAU_U, TAU_S_3, MAX_ITER_3,NUM_AGENTS)
    # print(f"3-attribute model converged in {iter_3} iterations with segregation level {seg_3}")
    income = IncomeGenerator([0.25, 0.25, 0.25, 0.25])
    race = RaceGenerator([0.25, 0.25, 0.25, 0.25])
    thing = PropertyGenerator({0: [0.1, 0.2, 0.4, 0.2, 0.1],
            1: [0.2, 0.3, 0.3, 0.1, 0.1],
            2: [0.3, 0.3, 0.2, 0.1, 0.1],
            3: [0.1, 0.2, 0.3, 0.2, 0.2],}, race_gen=race)
    _, _, un_over_t = simulate(L, W, POP_DENSITY, thing, 0, TAU_U, TAU_S_2)
    
    
    # maybe take the most unsatisfied people and then add some money or something to see if they can move.
    # get the number of unsatisified agents over time