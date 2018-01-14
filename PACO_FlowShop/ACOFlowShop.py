# -*- coding: utf-8 -*-

from Utils import *
from Ant import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class ACOFlowShop:

    def __init__(self,filepath=None, parameters=None):
        """
        :param filepath: If the file contains a single problem
        :param parameters: If the file contains a set of problems, initialize one by one with parameters
        """


        if filepath:
            number_jobs, number_machines, upper_bound, lower_bound, graph = Utils.read_problem_from_file(filepath)
        else:
            number_jobs, number_machines, upper_bound, lower_bound, graph, ID = \
                parameters['number_jobs'], parameters['number_machines'], parameters['upper_bound'],\
                parameters['lower_bound'], parameters['graph'],parameters['ID']
            self.ID = ID


        self.number_jobs = number_jobs
        self.number_machines = number_machines
        self.graph = np.array(graph).transpose().tolist()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound


        # Initialization of parameters
        self.initial_sequence, self.initial_makespan = self.find_initial_sequence()
        self.best_seq = self.initial_sequence
        self.z_best = self.initial_makespan
        self.p = 0.75    # Persistence of trail
        self.pheromone_trails = self.initialize_trail_intensities()



        print("\nACO FOR FLOW SHOP SCHEDULING ({} jobs, {} machines)".format(self.number_jobs,self.number_machines))
        if hasattr(self,'ID'):
            print("Problem number: " + str(self.ID))
        print("====================================================")
        print("Initial solution: " + str(self.best_seq))
        print("Initial Z: " + str(self.z_best))
        #print("Initial pheronomone trails: ")
        #print(self.pheromone_trails)

        self.number_ants = 1
        self.number_iterations = 40
        self.ant_results = [] # List to plot results after each iteration
        self.ant_results_ls = [] # List to plot results after local search after each iteration




    def find_initial_sequence(self):
        # Find seed initial permutation schedule using NEH heuristic, needed for initializing trail intensities

        # Step 1: Order the jobs by non-increasing sums of processing times on the machines:
        processing_times = [(i,sum(self.graph[i])) for i in range(self.number_jobs)]
        sorted_processing_times = list(map(lambda x: x[0], sorted(processing_times,key=lambda x: x[1],reverse=True)))

        # Step 2: Take the first two jobs and schedule them in order to minimise
        # the partial makespan as if there were only these two jobs.
        first_two_jobs = sorted_processing_times[:2]
        possible_seqs = [first_two_jobs,list(reversed(first_two_jobs))]
        partial_seqs_makespan = [(seq, get_schedule_makespan(seq,self.graph,self.number_machines)) for seq in possible_seqs]
        best_partial_seq = min(partial_seqs_makespan,key=lambda x:x[1])


        # Step 3: Insert the kth job at the place which minimises the partial makespan among the k possible ones
        for k in range(2,self.number_jobs):
            next_job = sorted_processing_times[k]
            partial_seq = best_partial_seq[0]
            possible_seqs = [partial_seq[:i] + [next_job] + partial_seq[i:] for i in range(k+1)]
            partial_seqs_makespan = [(seq, get_schedule_makespan(seq,self.graph,self.number_machines)) for seq in possible_seqs]
            best_partial_seq = min(partial_seqs_makespan, key=lambda x: x[1])

        # Step 4: Apply three times the job-index-based local search technique
        for _ in range(3):
            best_partial_seq = job_index_based_local_search(best_partial_seq[0],best_partial_seq[1],self.graph,self.number_machines)

        # Return sequence schedule and makespan
        return best_partial_seq[0],best_partial_seq[1]


    def initialize_trail_intensities(self):

        trails = [[0 for _ in range(self.number_jobs)] for _ in range(self.number_jobs)]
        n = self.number_jobs

        for i in range(n):
            for k in range(n):

                h = self.initial_sequence.index(i) #Position of job i in the seed initial sequence
                condition = abs(h-k)+1

                if  condition <= n/4:
                    trails[i][k] = 1/self.z_best

                elif condition > n/4 and condition <= n/2:
                    trails[i][k] = 1/(2*self.z_best)

                else:
                    trails[i][k] = 1/(4*self.z_best)

        return trails



    def solve_problem(self):
        print("\nSTARTING ITERATIONS\n")
        # print("Number of iterations: "+ str(self.number_iterations))

        for it in range(self.number_iterations):
            print("Current iteration: " + str(it))

            ant = Ant(self.number_jobs,self.number_machines,self.graph,self.pheromone_trails,self.best_seq) # Only one ant per iteration

            ant.build_sequence()
            self.ant_results.append(ant.z_graph) # Save result to plot
            self.ant_results_ls.append(ant.z_ls_graph)
            self.update_best_solution(ant)


            self.update_trail_intensities(ant)

            #print(self.pheromone_trails)

        # Job-index-based swap scheme
        self.before_swap_seq, self.before_swap_z = self.best_seq,self.z_best
        self.best_seq,self.z_best = job_index_based_swap_scheme(self.before_swap_seq,self.before_swap_z,self.graph,self.number_machines)

        self.show_solution()
        self.show_graphics()





    def update_best_solution(self,ant):
        z_current = ant.z_ant_sequence
        if z_current < self.z_best:
            self.best_seq = ant.ant_sequence
            self.z_best = z_current



    def update_trail_intensities(self, ant):

        # Pheromone updates on Ant sequence
        sequence = ant.ant_sequence
        z_current = ant.z_ant_sequence
        n = len(sequence)

        for i in range(len(self.pheromone_trails)):
            for k in range(len(self.pheromone_trails)):

                old_value = self.pheromone_trails[i][k]
                h = sequence.index(i)
                diff = (abs(h-k)+1)**(1/2)

                if(n <= 40):
                    if abs(h-k) <= 1:
                        new_value = self.p*old_value + (1/(diff*z_current))
                    else:
                        new_value = self.p*old_value
                else:
                    if abs(h-k) <= 2:
                        new_value = self.p*old_value + (1/(diff*z_current))
                    else:
                        new_value = self.p*old_value

                self.pheromone_trails[i][k] = new_value




    def clear_solution(self):
        # Restart the problem only with initial data so that initial sequence only has to be calculated once.
        self.best_seq = self.initial_sequence
        self.z_best = self.initial_makespan
        self.pheromone_trails = self.initialize_trail_intensities()




    def show_solution(self):
        print("\n============================\n")
        print("BEFORE SWAP\n")
        print("Best sequence found:" + str(self.before_swap_seq))
        print("Best makespan: " + str(self.before_swap_z))
        print("\n============================\n")
        print("AFTER SWAP\n")
        print("Best sequence found:" + str(self.best_seq))
        print("Best makespan: " + str(self.z_best))

    def show_graphics(self):
        plt.title(('Flowshop ACO (PACO): {} trabajos, {} máquinas ').format(self.number_jobs,self.number_machines) )
        x = range(len(self.ant_results))
        min_z = min(self.ant_results_ls)
        x_min_z = self.ant_results_ls.index(min_z)
        diff = min_z - self.upper_bound
        plt.text(max(x)-len(x)*0.15, self.upper_bound-10, 'Upper bound = '+ str(self.upper_bound), color='blue')

        plt.annotate('Min Z =' + str(min_z), xy=(x_min_z,self.z_best),  xytext=(x_min_z-2,min_z-12),arrowprops=dict(arrowstyle="->"))
        plt.ylim(self.upper_bound-15,max(self.ant_results)+3)
        x = range(len(self.ant_results))
        #plt.xticks(x)
        #x = range(len(self.ant_results))

        plt.plot(self.ant_results,'ro',self.ant_results,'r')
        plt.plot(self.ant_results_ls,'go',self.ant_results_ls,'g')
        plt.plot(x_min_z,self.z_best,'bo')

        for i in x:
            ymin = self.ant_results_ls[i]
            ymax = self.ant_results[i]
            plt.vlines(x=i, ymin=ymin, ymax=ymax,linestyle='--',color='gray')

        plt.ylabel('Z (Makespan)')
        plt.xlabel('Iteración')

        red_line = mpatches.Patch(color='red', label='Solución original')
        green_line =  mpatches.Patch(color='green', label='Solución tras búsqueda local')
        blue_dot = mpatches.Patch(color='blue', label='Solución tras esquema de intercambio')
        plt.legend(handles=[red_line,green_line,blue_dot])

        plt.axhline(y=self.upper_bound, color='b', linestyle='-')

        plt.show()



# Auxiliary functions

def job_index_based_local_search(sequence, z, graph, number_machines):
    current_sequence = sequence
    current_z = z
    for i in range(len(sequence)):
        modified_sequences = []

        for k in range(len(sequence)):

            if sequence[k] != i:
                # Insert job i in position k of current seed sequence
                mod_seq = list(current_sequence)
                mod_seq.remove(i)
                mod_seq.insert(k, i)
                # Calculate value of objective function of modified sequence
                z = get_schedule_makespan(mod_seq, graph, number_machines)
                modified_sequences.append((mod_seq, z))

        # Choose best sequence among such (n-1) modified sequences
        best_mod_seq = min(modified_sequences, key=lambda x: x[1])

        # If objective-function value is improved, replace current sequence by best one found.
        if best_mod_seq[1] < current_z:
            current_sequence = best_mod_seq[0]
            current_z = best_mod_seq[1]

    return current_sequence, current_z






def get_schedule_makespan(schedule, graph, number_machines):
    # sigma = σ -> ordered set of jobs already scheduled, out of n jobs; partial sequence
    # Dictionary q = q(σ,m) -> completion time of partial sequence σ on machine m
    # Dictionary of lists qi = q(σi,m) -> completion time of job i on machine m,
    #                   when the job is appended to partial sequence σ


    # Initialize q(null_schedule,m) to 0 for every machine m
    q = {m: 0 for m in range(number_machines + 1)}

    # Initialize q(σi,0) equal to zero.
    # This time indicates the time of availability of a job in the flowshop,
    # and it is equal to 0 for all jobs in case of static flowshops
    qi = {job: [0] for job in schedule}

    # Schedule jobs
    for job in schedule:
        for m in range(1, number_machines + 1):
            # q(σi,m) = max{ q(σ,m) ; q(σi, m-1)} + time(job i,machine m)
            time_append_job = max(q[m], qi[job][m - 1]) + graph[job][m - 1]
            qi[job].append(time_append_job)
            q[m] = time_append_job

    # Total flowtime for each job Ci -> q(σi,m)
    C = [qi[job][number_machines] for job in schedule]
    makespan = max(C)
    return makespan



def job_index_based_swap_scheme(seed_sequence, z, graph, number_machines):

    n = len(seed_sequence) # Number of jobs
    best_sequence = seed_sequence
    best_z = z

    for i in range(n):
        position_job_i = best_sequence.index(i)
        swap_sequences = []
        # Swap job i with every other job taken one at a time and store in swap_sequences
        for j in range(i,n):
            position_job_j = best_sequence.index(j)
            swap_sequence = list(best_sequence)
            swap_sequence[position_job_i] = j
            swap_sequence[position_job_j] = i
            z = get_schedule_makespan(swap_sequence, graph, number_machines)
            swap_sequences.append((swap_sequence, z))


        best_swap_seq = min(swap_sequences, key=lambda x: x[1])

        # If objective-function value is improved, replace current sequence by best one found.
        if best_swap_seq[1] < best_z:
            best_sequence = best_swap_seq[0]
            best_z = best_swap_seq[1]

    return best_sequence, best_z