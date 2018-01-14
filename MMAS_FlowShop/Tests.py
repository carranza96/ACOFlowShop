# -*- coding: utf-8 -*-

import Utils
from ACOFlowShop import *
import numpy as np
import os
import time

files = []
for file in os.listdir("resources"):
    if file[0].isdigit():
        files.append(os.path.join("resources", file))




def run_test(file):
    Utils.blockPrint()
    aco_problems= Utils.read_problem_set(file)
    n_machines,n_jobs = aco_problems[0].number_machines,aco_problems[0].number_jobs

    test_output = open("tests/testaux_"+ file[12:] ,"w+")
    test_output.write("Tests for problems with size: {} jobs and {} machines\n".format(n_jobs,n_machines))
    test_output.write("Problem ID;Upper Bound;Best Value;Worst Value;Mean Value;Standard Deviation\n")

    for aco in aco_problems:
        z_values = []
        for _ in range(20):
            # start = time.time()
            aco.solve_problem()
            z_values.append(aco.z_best)
            aco.clear_solution()
            # print(time.time()-start)
        problem_id = aco.ID
        upper_bound = aco.upper_bound
        best_value = min(z_values)
        worst_value = max(z_values)
        mean_value = np.mean(z_values).round(2)
        std_dev = np.std(z_values).round(2)
        test_output.write("{};{};{};{};{};{}\n".format(
                          problem_id, upper_bound, best_value, worst_value, mean_value, std_dev))



run_test("resources/7_100x5.txt")









