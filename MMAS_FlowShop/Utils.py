# -*- coding: utf-8 -*-

from ACOFlowShop import *
import sys, os

"""
Reads a text file and returns the problem size (number of jobs and machines)
and the data matrix
"""

def read_problem_from_file(path):
    inputf = open(path, "r")
    inputf.readline()
    problem_size = list(map(int,inputf.readline().split()))
    number_jobs = problem_size[0]
    number_machines = problem_size[1]
    upper_bound = problem_size[3]
    lower_bound = problem_size[4]
    inputf.readline()
    graph = [list(map(int, line.split())) for line in inputf.readlines()]
    return number_jobs, number_machines,upper_bound,lower_bound,graph


def read_problem_set(path):
    inputf = open(path, "r")
    aco_problems = list()
    for i in range(10):
        inputf.readline()
        problem_size = list(map(int, inputf.readline().split()))
        number_jobs = problem_size[0]
        number_machines = problem_size[1]
        upper_bound = problem_size[3]
        lower_bound = problem_size[4]
        inputf.readline()
        graph = [list(map(int, line.split())) for line in [inputf.readline() for _ in range(number_machines)]]
        ID = get_id(path,i)
        param_dict = {'number_jobs': number_jobs, 'number_machines': number_machines, 'upper_bound': upper_bound,
                      'lower_bound': lower_bound, 'graph': graph,'ID':ID}
        aco = ACOFlowShop(parameters=param_dict)
        aco_problems.append(aco)
    return aco_problems





def get_id(filename,index):
    size = int(filename[filename.index("/")+1])-1
    number = index+1
    id = number if size==0 else 10*size+number
    return id


def filter_by_problem_number(problems,number):
    return (x for x in problems if x.ID == number).next()






# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__