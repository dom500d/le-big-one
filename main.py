import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random
import grid_setting
from scipy.stats import lognorm
#from line_profiler import profile


from model import simulate, IncomeGenerator, RaceGenerator, PropertyGenerator
L, W = 200, 200 # Grid size
good_school = []
bad_school = []
good_school_zone = []
bad_school_zone = []
relaxation_applied = False
#@profile
def main():
    #### variable setting 
    # Grid and simulation parameters
      
    POP_DENSITY = 0.8  # 80% population density
    MAX_ITER_2 = 1000  # Max iterations for 2-attribute model
    MAX_ITER_3 = 2000  # Max iterations for 3-attribute model
    TAU_U = 0.50  # Utility threshold (50%)
    TAU_S_2 = 0.5 # Similarity threshold for 2-attribute (25%)
    TAU_S_3 = 0.5   # Similarity threshold for 3-attribute (50%)
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

    approx_pdf_values_hispanics = np.array([
        0.5, 2.2, 4.8, 5.3, 5, 4.9, 4.0, 3.2,
        2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5,
        0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08,
        0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.6  # last spike at 200k
    ])


    approx_pdf_values_black = np.array([
        0.4, 1.5, 3.2, 4.8, 5.2, 5.0, 4.6, 3.9,
        3.0, 2.4, 1.8, 1.4, 1.1, 0.9, 0.7, 0.6,
        0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.12,
        0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.6  # spike at 200k
    ])

    approx_pdf_values_asian = np.array([
        0.9, 2.0, 2.8, 3.0, 2.9, 2.8, 2.7, 2.6,
        2.4, 2.3, 2.1, 1.9, 1.6, 1.3, 1.1, 0.9,
        0.75, 0.66, 0.52, 0.45, 0.4, 0.36, 0.3, 0.24,
        0.2, 0.15, 0.11, 0.08, 0.06, 0.05, 5  # spike at 200k
    ])
    approx_pdf_values_white = np.array([
        0.5, 1.2, 2.5, 3.2, 3.8, 4.0, 4.1, 4.0,
        3.9, 3.7, 3.4, 3.0, 2.5, 2.1, 1.7, 1.3,
        1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25,
        0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 2.5  # spike at 200k
    ])

    income_bins = np.linspace(0, 200000, len(approx_pdf_values_hispanics))

    # Normalize to form a valid PDF
    pdf_values_hispanics = approx_pdf_values_hispanics /np.sum(approx_pdf_values_hispanics)
    pdf_values_black = approx_pdf_values_black /np.sum(approx_pdf_values_black)
    pdf_values_asian = approx_pdf_values_asian /np.sum(approx_pdf_values_asian)
    pdf_values_white = approx_pdf_values_white /np.sum(approx_pdf_values_white)

    '''
    white = 0
    black = 1
    asian = 2
    hispanic = 3
    '''

    '''san_diego = {0: [0.15, 0.15, 0.16, 0.19, 0.22],
            1: [0.22, 0.24, 0.23, 0.17, 0.14],
            2: [0.13, 0.13, 0.14, 0.19, 0.24],
            3: [0.25, 0.25, 0.20, 0.18, 0.16]}'''
    
    
    
    # maybe take the most unsatisfied people and then add some money or something to see if they can move.
    # get the number of unsatisified agents over time

    san_diego = {
        0: pdf_values_white,
        1: pdf_values_black,
        2: pdf_values_asian,
        3: pdf_values_hispanics
    }
    # san_diego = {0: [0.15, 0.15, 0.16, 0.19, 0.22],
    #         1: [0.22, 0.24, 0.23, 0.17, 0.14],
    #         2: [0.13, 0.13, 0.14, 0.19, 0.24],
    #         3: [0.25, 0.25, 0.20, 0.18, 0.16]}

    #Gaussian distribution
    '''san_diego = {
        0: {'mu': 2.5, 'sigma': 1.5, 'min': 0, 'max': 5},  # White: N(2.5, 1.5)
        1: {'mu': 1.5, 'sigma': 1.5, 'min': 0, 'max': 5},  # Black: N(1.5, 1.5)
        2: {'mu': 3.0, 'sigma': 1.5, 'min': 0, 'max': 5},  # Asian: N(3.0, 1.5)
        3: {'mu': 1.0, 'sigma': 1.5, 'min': 0, 'max': 5}   # Hispanic: N(1.0, 1.5)
    }'''

    '''for i in range(0,len(POP_DENSITY)):
        relaxation_applied = False
        good_school = []
        bad_school = []
        good_school_zone = []
        bad_school_zone = []
        TAU_U = 0.50  # Utility threshold (50%)
        TAU_S_2 = 0.5 # Similarity threshold for 2-attribute (25%)
        income_threshold = 0
        thing = Property_Gaussian_Generator(san_diego, race_gen=race)
        thing = PropertyGenerator(san_diego, race_gen=race)
        NUM_AGENTS = int(L * W * POP_DENSITY[i])
        un_over_t = simulate(L, W, POP_DENSITY[i], thing, income_threshold, TAU_U, TAU_S_2)'''
    #thing = Property_Gaussian_Generator(san_diego, race_gen=race)
    thing = PropertyGenerator(san_diego, race_gen=race)
    NUM_AGENTS = int(L * W * POP_DENSITY)
    un_over_t = simulate(L, W, POP_DENSITY, thing, income_threshold, TAU_U, TAU_S_2)
    


'''
# Plot the estimated PDF
plt.figure(figsize=(10, 5))
plt.plot(income_bins, pdf_values_hispanics, marker='o', label='hispanics')
plt.plot(income_bins, pdf_values_black, marker='o', label='black')
plt.plot(income_bins, pdf_values_asian, marker='o', label='asian')
plt.plot(income_bins, pdf_values_white, marker='o', label='white')
plt.xlabel("Income ($)")
plt.ylabel("Probability")
plt.title("Discretized Probability Distribution of Hispanic Incomes (2016)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()'''





if __name__=="__main__":
    main()