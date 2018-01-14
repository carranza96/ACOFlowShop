# -*- coding: utf-8 -*-

import random
import Utils
import ACOFlowShop

class Ant():
    def __init__(self,number_jobs,number_machines,graph,pheromone_trails,best_seq):

        self.number_jobs = number_jobs
        self.number_machines = number_machines
        self.graph = graph
        self.pheromone_trails = pheromone_trails
        self.best_seq = best_seq


        self.ant_sequence = []
        self.current_index = 0
        self.z_ant_sequence = 0

        self.z_graph = 0  # Save result to plot
        self.z_ls_graph = 0  # Save result to plot



    def build_sequence(self):
        # print("BEST SOLUTION FOUND: " + str(self.best_seq))
        # print("BUILDING SEQUENCE")


        while (self.current_index < self.number_jobs):
            # print(self.ant_sequence)
            next_job = self.select_next_job()
            self.append_job(next_job)
            self.current_index += 1

        self.z_ant_sequence = ACOFlowShop.get_schedule_makespan(self.ant_sequence, self.graph, self.number_machines)
        print("Original solution > Makespan: " + str(self.z_ant_sequence) + ", Schedule: " + str(self.ant_sequence))
        self.z_graph = self.z_ant_sequence  # Save result to plot
        # Job-index-based local search
        self.improve_solution()
        print("After Local Search > Makespan: "+ str(self.z_ant_sequence)+", Schedule: " + str(self.ant_sequence) + "\n")
        self.z_ls_graph = self.z_ant_sequence  # Save result to plot



    def select_next_job(self):
        u = random.uniform(0,1)
        k = self.current_index

        first_five_unscheduled_jobs = [i for i in self.best_seq if i not in self.ant_sequence][:5]

        t = {i: sum(self.pheromone_trails[i][:k + 1]) for i in first_five_unscheduled_jobs}
        #print("U: " + str(u))
        #print(t)


        if u <= 0.4:
            ## ERROR job = max(t, key=t.get)
            job = first_five_unscheduled_jobs[0]

        else:
            if u <= 0.8:
                job = max(t,key=t.get)
            else:
                sum_t = sum(t.values())
                probabilities = {i: t[i]/sum_t for i in first_five_unscheduled_jobs}
                r = random.random()
                #print("r: " + str(r))
                #print("Probabilities: " + str(probabilities))
                sampling_sum = 0
                for i in probabilities:
                    sampling_sum += probabilities[i]
                    if sampling_sum > r:
                        job = i
                        break
        #print("\n---")
        return job




    def append_job(self,next_job):
        self.ant_sequence.append(next_job)



    def improve_solution(self):
        # Apply three times the job-index-based local search technique
        for _ in range(3):
            self.ant_sequence,self.z_ant_sequence = \
                ACOFlowShop.job_index_based_local_search(self.ant_sequence, self.z_ant_sequence,self.graph,self.number_machines)






