import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random
import grid_setting

from model import simulate, IncomeGenerator, RaceGenerator, PropertyGenerator,Property_Gaussian_Generator
L, W = 75, 75 # Grid size
good_school = []
bad_school = []

if __name__=="__main__":
#### variable setting 
    # Grid and simulation parameters
      
    POP_DENSITY = [0.6, 0.7, 0.8]  # 80% population density
    MAX_ITER_2 = 1000  # Max iterations for 2-attribute model
    MAX_ITER_3 = 2000  # Max iterations for 3-attribute model
    TAU_U = 0.50  # Utility threshold (50%)
    TAU_S_2 = 0.5 # Similarity threshold for 2-attribute (25%)
    income_threshold = 0
    
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


    '''
    white = 0
    black = 1
    asian = 2
    hispanic = 3
    '''

    # san_diego = {0: [0.15, 0.15, 0.16, 0.19, 0.22],
    #         1: [0.22, 0.24, 0.23, 0.17, 0.14],
    #         2: [0.13, 0.13, 0.14, 0.19, 0.24],
    #         3: [0.25, 0.25, 0.20, 0.18, 0.16]}
    san_diego = {
        0: {'mu': 2.5, 'sigma': 1.5, 'min': 0, 'max': 5},  # White: N(2.5, 1.5)
        1: {'mu': 1.5, 'sigma': 1.5, 'min': 0, 'max': 5},  # Black: N(1.5, 1.5)
        2: {'mu': 3.0, 'sigma': 1.5, 'min': 0, 'max': 5},  # Asian: N(3.0, 1.5)
        3: {'mu': 1.0, 'sigma': 1.5, 'min': 0, 'max': 5}   # Hispanic: N(1.0, 1.5)
    }
    thing = Property_Gaussian_Generator(san_diego, race_gen=race)
    # thing = PropertyGenerator(san_diego, race_gen=race)
    for i in range(0,len(POP_DENSITY)):
        NUM_AGENTS = int(L * W * POP_DENSITY[i])
        un_over_t = simulate(L, W, POP_DENSITY[i], thing, income_threshold, TAU_U, TAU_S_2)
    
    
    # maybe take the most unsatisfied people and then add some money or something to see if they can move.
    # get the number of unsatisified agents over time

