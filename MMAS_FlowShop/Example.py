# -*- coding: utf-8 -*-

from ACOFlowShop import *

# Default example ready to execute

# Read problem data from file and construct initial sequence using NEH Heuristic
aco = ACOFlowShop(filepath="resources/default_example.txt") #20x5
#aco = ACOFlowShop(filepath="resources/default_example2.txt") #50x10
#aco = ACOFlowShop(filepath="resources/default_example3.txt") #100x5


# Solve problem constructing n (number of iterations)
#  ant_sequences and improving them using job-index-based local search
aco.solve_problem()
