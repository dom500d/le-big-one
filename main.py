import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random
import grid_setting
from model import simulate, IncomeGenerator, RaceGenerator
L, W = 25, 25 # Grid size
if __name__=="__main__":
#### variable setting 
    # Grid and simulation parameters
      
    POP_DENSITY = 0.5  # 80% population density
    NUM_AGENTS = int(L * W * POP_DENSITY)
    MAX_ITER_2 = 1000  # Max iterations for 2-attribute model
    MAX_ITER_3 = 2000  # Max iterations for 3-attribute model
    TAU_U = 0.5  # Utility threshold (50%)
    TAU_S_2 = 0.25  # Similarity threshold for 2-attribute (25%)
    TAU_S_3 = 0.5   # Similarity threshold for 3-attribute (50%)

    
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
    simulate(L, W, POP_DENSITY, 2, income, race, 100000, TAU_U, TAU_S_2)