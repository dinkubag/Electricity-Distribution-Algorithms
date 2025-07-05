#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:39:18 2022

@author: dineshkumarbaghel
"""

import json
import ffk
import sys
import numpy as np
import experiments_csv
import os
from typing import List, Tuple
import logging
import datetime
import math
import time
import heuristic_algorithm1 as ha1
import heuristic_algorithm2 as ha2
import heuristic_algorithm3 as ha3
import heuristic_algorithm4 as ha4

import logging.handlers
import queue

class InvalidK(Exception):
    def __init__(self, K):
        self.__K = K
        
    def get_exception_detail(self):
        print(f"Invalid K = {self.__K}")

class ElectricityDistribution:
    def __init__(self, hrs_ina_day:int = 24):
        self.tot_hrs = None
        self.numof_agents = None
        self.hesc = None
        self.avg_consumption = None
        self.agents_comfort_vector = []
        self.agents_consumption_vector = []
        self.agents_consumption_vector_byhour = []
        self.agents_comfort_vector_byhour = []
        
        # how many hours in a day. Default is 24.
        self.hours_window = hrs_ina_day
        
        self.binsize = None
        self.k_hd = 1
        self.k_kbp = 1
        
        self.k=1
        
        # self.r = 1  #as in alternative geometric grouping. It is used as in a group the demands sufficient to make the sum equal to exceed r*max_demand
        # self.e = 1/2    
        
    def read_data(self):
        '''
        It reads the data from the data directory.
        
        The header of file should be Hour_ID, Consumption(kWh), Agent_ID, Comfort.

        Returns
        -------
        tuple
            DESCRIPTION.
            
        # >>> ed = ElectricityDistribution()
        # >>> ed.read_data()

        '''
        
        #  Uncomment the below line if it is not working.
        fr = open("./../data/developing_country_household_consumption_data.csv",'r')
        
        agents_set = set()           #set of agents.
        hrs_set = set()          #set of hours.
        l = fr.readline()
        
        for l in fr:
            agents_set.add(int(l.rstrip('\n').split(',')[2]))
            hrs_set.add(int(l.rstrip('\n').split(',')[0]))
        
        self.tot_hrs = len(list(hrs_set)) #total number of hours
        self.numof_agents = len(list(agents_set))
        
        #creating list of empty lists for agent comfort and consumption.
        for i in range(self.numof_agents):    
            self.agents_comfort_vector.append([])
            self.agents_consumption_vector.append([]) 
           
        # initialize agents consumption by hour i.e. for each hour there is a vector of consumptions of agents
        # of length numof_agents
        for hr in range(self.tot_hrs):
            self.agents_consumption_vector_byhour.append([])
            self.agents_comfort_vector_byhour.append([])
            
        fr.seek(0)
        line = fr.readline()
        for line in fr:
            agent_ID = int(line.rstrip('\n').split(',')[2])
            hr_ID = int(line.rstrip('\n').split(',')[0])
            self.agents_comfort_vector[agent_ID-1].append(float(line.rstrip('\n').split(',')[3]))
            self.agents_comfort_vector_byhour[hr_ID-1].append(float(line.rstrip('\n').split(',')[3]))
            
            self.agents_consumption_vector[agent_ID-1].append(float(line.rstrip('\n').split(',')[1]))
            self.agents_consumption_vector_byhour[hr_ID - 1].append(float(line.rstrip('\n').split(',')[1]))
        
        # hesc: Hourly estimated supply capacity for the entire data.
        self.hesc = sum([sum(vector) for vector in self.agents_consumption_vector])/self.tot_hrs
        self.avg_consumption = self.hesc/self.numof_agents
        

        if os.path.exists('./data/data_comfort.csv') and os.path.exists('./data/data_consumption.csv'):
            print("Files: data_comfort.csv and data_consumption.csv exists.")
        else:
            fw_comfort = open("./../../data/data_comfort.csv",'w+')
            fw_consumption = open("./../../data/data_consumption.csv",'w+')    
                
            
            for i in range(len(self.agents_comfort_vector)):
                json.dump(self.agents_comfort_vector[i],fw_comfort)
                json.dump(self.agents_consumption_vector[i],fw_consumption)
                if i < len(self.agents_comfort_vector)-1:
                    fw_comfort.write('\n')
                    fw_consumption.write('\n')
            
            
            fw_comfort.close()
            fw_consumption.close()
        
    
    def electricity_distribution_supply_algorithms(self, alg: str, simulations: int, supply_alg:str, k_hd:int=1, k_kbp:int=1, delta: float=0, epsilon: float=0, allowed_supply_diff:float=0, demand_tomaxdemand_ratio:float = 0, r:float = 0) -> dict:
        '''
        
        This function the egalitarian electricity distribution of supply. It uses four different algorithms.
        
        Default parameters except supply are algorithm specifc.
        
        Parameters
        ----------
        alg : str
            DESCRIPTION.
        delta : float
            DESCRIPTION.
        simulations : int
            DESCRIPTION.
        K : int
            DESCRIPTION.
        Other parameters are specific to the algorithms used in this function.

        Returns
        -------
        dict
            DESCRIPTION.
            
        >>> ed = ElectricityDistribution()
        >>> ed.tot_hrs = 2
        >>> ed.numof_agents = 3
        >>> ed.agents_comfort_vector = [[1,0.5],[0.67,1],[1,1]]
        >>> ed.agents_consumption_vector = [[1,0.5],[1,1.5],[1,1]]
        >>> ed.agents_consumption_vector_byhour = [[1,1,1], [0.5,1.5,1]]
        >>> ed.agents_comfort_vector_byhour = [[1,0.67,1], [0.5,1,1]]
        >>> ed.hours_window = 2
        >>> ed.binsize = 2
        >>> ed.electricity_distribution_supply_algorithms("FFk", 1,"heuristic_algorithm1", 1, 0, 0, 0, 0.05)
        {'Connection_Hours': 'Connection_Hours', 'Utilitarian(Connection_Hours)': 4.0, 'Utilitarian(Connection_Hours)_SD': 0.0, 'Egalitarian(Connection_Hours)': 0.83333, 'Egalitarian(Connection_Hours)_SD': 0.0, 'Max-Util-diff(Connection_Hours)': 1.1666699999999999, 'Max-Util-diff(Connection_Hours)_SD': 0.0, 'Numof-Agents-ConnTime-ge2050': 0.0, 'Numof-Agents-ConnTime-ge2050_SD': 0.0, 'Comfort': 'Comfort', 'Utilitarian(Comfort)': 3.335, 'Utilitarian(Comfort)_SD': 0.0, 'Egalitarian(Comfort)': 0.4001976047904192, 'Egalitarian(Comfort)_SD': 0.0, 'Max-Util-diff(Comfort)': 0.5998023952095808, 'Max-Util-diff(Comfort)_SD': 0.0, 'Supply': 'Supply', 'Utilitarian(Supply)': 3.66667, 'Utilitarian(Supply)_SD': 0.0, 'Egalitarian(Supply)': 0.4, 'Egalitarian(Supply)_SD': 0.0, 'Max-Util-diff(Supply)': 0.6, 'Max-Util-diff(Supply)_SD': 0.0, 'EgalitarianSupply(Watts)': 1.0, 'Max_Supply-diff(Watts)': 0.5, 'Avg-Min-Supply(Watts)': 1.0, 'Avg-Max_Supply-diff(Watts)': 0.16666999999999987}
        
        >>> ed = ElectricityDistribution()
        >>> ed.tot_hrs = 2
        >>> ed.numof_agents = 3
        >>> ed.agents_comfort_vector = [[1,0.5],[0.67,1],[1,1]]
        >>> ed.agents_consumption_vector = [[1,0.5],[1,1.5],[1,1]]
        >>> ed.agents_consumption_vector_byhour = [[1,1,1], [0.5,1.5,1]]
        >>> ed.agents_comfort_vector_byhour = [[1,0.67,1], [0.5,1,1]]
        >>> ed.hours_window = 2
        >>> ed.binsize = 2
        >>> ed.electricity_distribution_supply_algorithms("FFk", 1, "heuristic_algorithm1", 2, 0, 0, 0, 0.05)
        {'Connection_Hours': 'Connection_Hours', 'Utilitarian(Connection_Hours)': 4.00001, 'Utilitarian(Connection_Hours)_SD': 0.0, 'Egalitarian(Connection_Hours)': 1.06667, 'Egalitarian(Connection_Hours)_SD': 0.0, 'Max-Util-diff(Connection_Hours)': 0.5999999999999999, 'Max-Util-diff(Connection_Hours)_SD': 0.0, 'Numof-Agents-ConnTime-ge2050': 0.0, 'Numof-Agents-ConnTime-ge2050_SD': 0.0, 'Comfort': 'Comfort', 'Utilitarian(Comfort)': 3.2800089, 'Utilitarian(Comfort)_SD': 0.0, 'Egalitarian(Comfort)': 0.5069873652694611, 'Egalitarian(Comfort)_SD': 0.0, 'Max-Util-diff(Comfort)': 0.27079263473053883, 'Max-Util-diff(Comfort)_SD': 0.0, 'Supply': 'Supply', 'Utilitarian(Supply)': 3.7000100000000002, 'Utilitarian(Supply)_SD': 0.0, 'Egalitarian(Supply)': 0.506668, 'Egalitarian(Supply)_SD': 0.0, 'Max-Util-diff(Supply)': 0.2711119999999999, 'Max-Util-diff(Supply)_SD': 0.0, 'EgalitarianSupply(Watts)': 1.1666699999999999, 'Max_Supply-diff(Watts)': 0.10000000000000009, 'Avg-Min-Supply(Watts)': 1.26667, 'Avg-Max_Supply-diff(Watts)': 0.0}
        
        '''
        if __name__ == "__main__":
            logger = logging.getLogger(__name__)
            # if testing individual algorithm you can use the below line
            # logging.basicConfig(filename='eds_' + supply_alg+ '.log', encoding='utf-8', level=logging.DEBUG, force=True)
            # otherwise below line is default
            logging.basicConfig(filename="time_model_electricity_distribution.log", encoding="utf-8", level=logging.DEBUG, force=True)
        else:
            logger = logging.getLogger("Supply Model: Electricity Distribution Algorithms: ")
        
        
        # uncomment the below lines to test this function-----START
        # logger.info(datetime.datetime.now())
        # try:
        #     logger.debug("inside try:")
        #     if K<=0:
        #         logger.debug("K is 0.")
        #         raise InvalidK(K)
        # except InvalidK as ik:
        #     logger.debug("Exception catched here and handled.")
        #     ik.get_exception_detail()
        #     return
        # --------------END
        
        # uncommen the below line if you want to measure the exdecution time of the algorithm(s)
        # alg_start = time.perf_counter()
            
        avg_total_connection = []
        avg_egal_hours = []
        avg_connection_EF = []
        
        avg_total_comfort = []
        avg_acs = []
        avg_comfort_EF = []
        
        avg_total_supply = []
        avg_asr = []
        avg_supply_EF = []
        
        avg_agent_min_supply = []
        avg_agent_max_supply = []
        
        avg_min_supply = []
        avg_max_supply = []
        
        self.k_hd=k_hd
        self.k_kbp = k_kbp
        
        # uncomment the below line for testing the function.
        # logger.debug(f"k: {self.k}")
        
        alg = "online" if alg == "FFk" else "decreasing"
        
        for sml in range(simulations):
            
            # uncomment the below line if you want to measure the computation time of one simulation
            # sml_start = time.perf_counter()
            
            # uncomment the below one line to test this function for each simulation
            # logger.debug(f"Simulation: {sml}")
            
            total_connection = 0    # Sum of the total connection time of all the agents.
            egal_hour_connection = 0 #out of total number of hours, at least, for how many hours all agents are connected.
            agent_total_connection = np.zeros(self.numof_agents)    # the total connectin time for each agent.
            
            total_comfort = 0   # total comfort received of all the agents during their connection time.
            agent_total_comfort = np.zeros(self.numof_agents)    # agent's total comfort in their connection time.
            acs_vector = np.zeros(self.numof_agents)    # agent's agent comfort rate. Defined as comfort-delivered/total-agent-comfort
            
            total_supply = 0    # sum of the total supply delivered of all the agents during their connection hrs.
            agent_total_supply = np.zeros(self.numof_agents)    # agent's total supply during their connection time.
            asr_vector = np.zeros(self.numof_agents)    # # agent's agent supply rate. Defined as supply-delivered/total-agent-supply
            
            hours = 0   # define how many hours have passed.
            estimated_dayahead_agents_consumption_vector = []
            
            allhour_max_supply = []
            allhour_min_supply = []
            
            # derived agents consumption vector
            derived_agents_consumption_vector = []
            
            for each_agent in range(self.numof_agents):
                derived_agents_consumption_vector.append([])
            
            while hours < self.tot_hrs:
                # uncomment the below line while testing this function.
                # logger.info(f"\nSimulation: {sml}, hours: {hours}")
            
                for each_hour in range(hours, hours+ self.hours_window):
                    dayahead_hourly_agents_consumption = []
                    for each_agent in range(self.numof_agents):
                        eachhr_each_agent_consumption = abs(np.random.normal(self.agents_consumption_vector_byhour[each_hour][each_agent], delta))
                        derived_agents_consumption_vector[each_agent].append(eachhr_each_agent_consumption)
                        dayahead_hourly_agents_consumption.append(eachhr_each_agent_consumption)
                    estimated_dayahead_agents_consumption_vector.append(dayahead_hourly_agents_consumption)
                
                # logging. Uncommnent the below line when testing this function.
                # logger.debug(f"Estimated Day Ahead Agents Consumption Vector: {estimated_dayahead_agents_consumption_vector}\n")
                
                
                # calculate the day ahead supply.                
                if self.binsize == None:
                    self.binsize = sum([sum(estimated_dayahead_agents_consumption_vector[each_hour]) for each_hour in range(hours, hours + self.hours_window)])/(self.hours_window)
    
                
                for hr in range(hours, hours+self.hours_window): 
                    
                    # uncomment the below lines while testing this function----START
                    # logger.debug(f"Hour: {hr}")
                    # logger.debug(f"Agents total demand: {sum(estimated_dayahead_agents_consumption_vector[hr])}, Supply: {self.binsize}")
                    # logger.info(f"Algorithm: {alg}, hour: {hr}")
                    # packing  = esd.compute_egal_alg5(alg, estimated_dayahead_agents_consumption_vector[hr], self.binsize, epsilon, allowed_supply_diff)[0]
                    # ----------------END
                    
                    if supply_alg == "heuristic_algorithm1":
                        packing = ha1.compute_egal_alg1_v2(alg, demands=estimated_dayahead_agents_consumption_vector[hr], supply=self.binsize, demand_tomaxdemand_ratio=demand_tomaxdemand_ratio, k_ofhighest_demand=k_hd)
                    elif supply_alg == "heuristic_algorithm2":
                        packing = ha2.compute_egal_alg2_v2(alg, demands=estimated_dayahead_agents_consumption_vector[hr], supply=self.binsize, k=k_kbp, demand_tomaxdemand_ratio=demand_tomaxdemand_ratio, e=epsilon)
                    elif supply_alg == "heuristic_algorithm3":
                        packing  = ha3.compute_egal_alg3_v3(alg, demands=estimated_dayahead_agents_consumption_vector[hr], supply=self.binsize, k=k_hd , r=r )
                    elif supply_alg == "heuristic_algorithm4":
                        packing = ha4.compute_egal_alg4_v1(alg, demands=estimated_dayahead_agents_consumption_vector[hr], supply=self.binsize, k=k_kbp)
                        
                    
                    # logging------------START
                    # uncomment the below lines while testing this function
                    # logger.debug(f"Total connection delivered for hour {hr}: {sum([conn[0] for conn in packing.agents_conn_vector])}")
                    # logger.debug(f"Total supply delivered for hour {hr}: {sum([egal[0] for egal in packing.agents_egal_vector])}")
                    # logger.debug(f"Total comfort delivered for hour {hr}: {sum([self.agents_comfort_vector_byhour[hr][conn[1]]*conn[0] for conn in packing.agents_conn_vector])}")
                    # logging---------END
                    
                    for each_agent_connection in packing.agents_conn_vector:
                        # adding the connection of each agent to the total_connection, and also
                        # computing the total connection for each agent.
                        total_connection += each_agent_connection[0]
                        agent_total_connection[each_agent_connection[1]] += each_agent_connection[0]                                
                        
                        # computing the total comfort and comfort  for each agent
                        total_comfort += self.agents_comfort_vector_byhour[hr][each_agent_connection[1]]*each_agent_connection[0]
                        agent_total_comfort[each_agent_connection[1]] += self.agents_comfort_vector_byhour[hr][each_agent_connection[1]]*each_agent_connection[0]
                    
                    # add to the egal_hour_connection the min number of hours (in a given solution) an agent is connected
                    egal_hour_connection += packing.egalconn_val[0]
                    
                    # compute the total supply delivered, and 
                    # for each agent compute the agent's total supply
                    for each_agent_supply in packing.agents_egal_vector:
                        total_supply += each_agent_supply[0]
                        agent_total_supply[each_agent_supply[1]] += each_agent_supply[0]
                    
                    max_sup, min_sup = packing.determine_max_and_min()
                    
                    # uncomment the below line while testing this function
                    # logger.debug(f"Max Supply: {max_sup}, Min Supply: {min_sup}")
                    
                    allhour_max_supply.append(max_sup)
                    allhour_min_supply.append(min_sup)
                    
                hours += self.hours_window            
            
            avg_total_connection.append(total_connection)
            # avg_egal_hours.append(egal_hour_connection)
            avg_egal_hours.append(min(agent_total_connection))
            avg_connection_EF.append(max(agent_total_connection) - min(agent_total_connection))
            
            avg_total_comfort.append(total_comfort)
            for agent in range(self.numof_agents):
                acs_vector[agent] = agent_total_comfort[agent] / sum(self.agents_comfort_vector[agent])
            avg_acs.append(min(acs_vector))
            avg_comfort_EF.append(max(acs_vector) - min(acs_vector))
            
            avg_total_supply.append(total_supply)
            for agent in range(self.numof_agents):
                asr_vector[agent] = agent_total_supply[agent] / sum(derived_agents_consumption_vector[agent])
            avg_asr.append(min(asr_vector))
            avg_supply_EF.append(max(asr_vector) - min(asr_vector))
            
            avg_agent_min_supply.append(min(agent_total_supply))
            avg_agent_max_supply.append(max(agent_total_supply))
            
            avg_max_supply.append(sum(allhour_max_supply))
            avg_min_supply.append(sum(allhour_min_supply))
            
            # logging-----------START
            # uncomment the below lines while testing this function
            # logger.info("\n\n")
            # logger.debug(f"Simulation: {sml}")
            # logger.info(f"Utilitarian(Connection_Hours): {sum(avg_total_connection)/(sml+1)}"),
            # logger.info(f"Utilitarian(Connection_Hours)_SD: {np.std(avg_total_connection)}")
            # logger.info(f"Egalitarian(Connection_Hours): {sum(avg_egal_hours)/(sml+1)}")
            # logger.info(f"Egalitarian(Connection_Hours)_SD: {np.std(avg_egal_hours)}")
            # logger.info(f"Max-Util-diff(Connection_Hours): {sum(avg_connection_EF)/(sml+1)}")
            # logger.info(f"Max-Util-diff(Connection_Hours)_SD: {np.std(avg_connection_EF)}")
            # logger.info("\n\n")
            # logger.info(f"Utilitarian(Comfort): {sum(avg_total_comfort)/(sml+1)}")
            # logger.info(f"Utilitarian(Comfort)_SD: {np.std(avg_total_comfort)}")
            # logger.info(f"Egalitarian(Comfort): {sum(avg_acs)/(sml+1)}")
            # logger.info(f"Egalitarian(Comfort)_SD: {np.std(avg_acs)}")
            # logger.info(f"Max-Util-diff(Comfort): {sum(avg_comfort_EF)/(sml+1)}")
            # logger.info(f"Max-Util-diff(Comfort)_SD: {np.std(avg_comfort_EF)}")
            # logger.info("\n\n")
            # logger.info(f"Utilitarian(Supply): {sum(avg_total_supply)/(sml+1)}")
            # logger.info(f"Utilitarian(Supply)_SD: {np.std(avg_total_supply)}")
            # logger.info(f"Egalitarian(Supply): {sum(avg_asr)/(sml+1)}")
            # logger.info(f"Egalitarian(Supply)_SD: {np.std(avg_asr)}")
            # logger.info(f"Max-Util-diff(Supply): {sum(avg_supply_EF)/(sml+1)}") 
            # logger.info(f"Max-Util-diff(Supply)_SD: {np.std(avg_supply_EF)}")
            
            # logger.info(f"EgalitarianSupply(Watts): {sum([sup for sup in avg_agent_min_supply])/(sml+1)}")
            # logger.info(f"Maximum Supply difference(Watts): {sum([sup for sup in avg_agent_max_supply])/(sml+1) - sum([sup for sup in avg_agent_min_supply])/(sml+1)}")
            
            # logger.info(f"Avg-Min-Supply(Watts) : {(sum(avg_min_supply))/(sml+1)}")
            # logger.info(f"Avg-Max_Supply-diff(Watts) : {(sum(avg_max_supply) - sum(avg_min_supply))/(sml+1)}")
            
            # sml_end = time.perf_counter()
            # logger.info(f"Time taken for simulation {sml} : {sml_end - sml_start:0.4f} seconds")
            # logger.info("\n\n")
        
        # uncomment the below two lines if you want to measure the computation time of algorithm
        # alg_end = time.perf_counter()
        # logger.info(f"Time taken for entire process : {alg_end - alg_start:0.4f} seconds")
        
        return {
                "Connection_Hours": "Connection_Hours",
                "Utilitarian(Connection_Hours)": sum(avg_total_connection)/simulations,
                "Utilitarian(Connection_Hours)_SD": np.std(avg_total_connection),
                "Egalitarian(Connection_Hours)": sum(avg_egal_hours)/simulations,
                "Egalitarian(Connection_Hours)_SD": np.std(avg_egal_hours),
                "Max-Util-diff(Connection_Hours)": sum(avg_connection_EF)/simulations,
                "Max-Util-diff(Connection_Hours)_SD": np.std(avg_connection_EF),
                "Comfort": "Comfort",
                "Utilitarian(Comfort)": sum(avg_total_comfort)/simulations,
                "Utilitarian(Comfort)_SD": np.std(avg_total_comfort),
                "Egalitarian(Comfort)": sum(avg_acs)/simulations,
                "Egalitarian(Comfort)_SD": np.std(avg_acs),
                "Max-Util-diff(Comfort)": sum(avg_comfort_EF)/simulations,
                "Max-Util-diff(Comfort)_SD": np.std(avg_comfort_EF),
                "Supply": "Supply",
                "Utilitarian(Supply)": sum(avg_total_supply)/simulations,
                "Utilitarian(Supply)_SD": np.std(avg_total_supply),
                "Egalitarian(Supply)": sum(avg_asr)/simulations,
                "Egalitarian(Supply)_SD": np.std(avg_asr),
                "Max-Util-diff(Supply)": sum(avg_supply_EF)/simulations, 
                "Max-Util-diff(Supply)_SD": np.std(avg_supply_EF),
                "EgalitarianSupply(Watts)": sum(avg_agent_min_supply)/simulations,
                "Max_Supply-diff(Watts)" : (sum(avg_agent_max_supply) - sum(avg_agent_min_supply))/simulations,
                "Avg-Min-Supply(Watts)" : (sum(avg_min_supply))/simulations,
                "Avg-Min-Supply(Watts)_SD" : np.std(avg_min_supply),
                "Avg-Max_Supply-diff(Watts)" : (sum(avg_max_supply) - sum(avg_min_supply))/simulations,
                "Avg-Max_Supply-diff(Watts)_SD" : np.std([avg_max_supply[i]-avg_min_supply[i] for i in range(simulations)]),
            } 
        
        
        

def call_ed():
    '''
    This is a function to call electricity distribution function.

    Returns
    -------
    None.

    '''
    
    # Comment or uncomment the below lines depending on which algorithm(s) you want to execute.
    
    
    # input_ranges1 = {
    #         "supply_alg": ["heuristic_algorithm1"],
    #         "alg": ["FFk", "FFDk"],
    #         # "K": range(1,20),
    #         # "k": [1,5,10,15,20],
    #         "k_hd": [1,2,3,4,5,6,7,8],
    #         # "delta": [0.05],
    #         "demand_tomaxdemand_ratio": [0.01],
    #         "delta": [round(0.5*0.5,4), 0.5, round(1.5*0.5,4), round(2*0.5,4), round(2.5*0.5,4), round(3*0.5,4) ],
    #         "simulations": [4],
    #     }
    
    
    
    # input_ranges2 = {
    #         "supply_alg": ["heuristic_algorithm2"],
    #         "alg": ["FFk", "FFDk"],
    #         # "K": range(1,20),
    #         # "k": [1,5,10,15,20],
    #         "k_kbp": [50],
    #         "epsilon": [0.5],
    #         "delta": [0.05],
    #         # "epsilon":[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #         "demand_tomaxdemand_ratio": [0.05, 0.01],
    #         # "delta": [round(0.5*0.5,4), 0.5, round(1.5*0.5,4), round(2*0.5,4), round(2.5*0.5,4), round(3*0.5,4) ],
    #         "simulations": [4],
    #         # only for alg 5
    #         # "epsilon" : [0.2],
    #         # "allowed_supply_diff":range(7,11) 
    #     }
    
    # input_ranges3 = {
    #         "supply_alg": ["heuristic_algorithm3"],
    #         "alg": ["FFk", "FFDk"],
    #         # "K": range(1,20),
    #         # "k": [1,5,10,15,20],
    #         "k_hd": [1,5],
    #         "r": [0.5, 0.75, 1],
    #         "delta": [0.05],
    #         # "epsilon":[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #         # "delta": [round(0.5*0.5,4), 0.5, round(1.5*0.5,4), round(2*0.5,4), round(2.5*0.5,4), round(3*0.5,4) ],
    #         "simulations": [4],
    #         # only for alg 5
    #         # "epsilon" : [0.2],
    #         # "allowed_supply_diff":range(7,11) 
    #     }
    
    # input_ranges4 = {
    #         "supply_alg": ["heuristic_algorithm4"],
    #         "alg": ["FFk", "FFDk"],
    #         # "K": range(1,20),
    #         # "k": [1,5,10,15,20],
    #         "k_kbp": [50],
    #         "delta": [0.05],
    #         # "epsilon":[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #         # "delta": [round(0.5*0.5,4), 0.5, round(1.5*0.5,4), round(2*0.5,4), round(2.5*0.5,4), round(3*0.5,4) ],
    #         "simulations": [4],
    #         # only for alg 5
    #         # "epsilon" : [0.2],
    #         # "allowed_supply_diff":range(7,11) 
    #     }
    
    
    ed = ElectricityDistribution()
    ed.read_data()
    # ex1 = experiments_csv.Experiment("results/", "ed-sm_ha1-alg_sd0dot05.csv", "results/backup")
    # ex2 = experiments_csv.Experiment("results/", "ed-sm_ha2-alg_sd0dot05.csv", "results/backup")
    # ex3 = experiments_csv.Experiment("results/", "ed-sm_ha3-alg_sd0dot05.csv", "results/backup")
    # ex4 = experiments_csv.Experiment("results/", "ed-sm_ha4-alg_sd0dot05.csv", "results/backup")
    # ex1.run(ed.electricity_distribution_supply_algorithms, input_ranges1)
    # ex2.run(ed.electricity_distribution_supply_algorithms, input_ranges2)
    # ex3.run(ed.electricity_distribution_supply_algorithms, input_ranges3)
    # ex4.run(ed.electricity_distribution_supply_algorithms, input_ranges4)
    
    

def print_graph():
    '''
    Once the results are generated, you can call this function to draw graphs.
    

    Returns
    -------
    None.

    '''
    experiments_csv.multi_plot_results("./results/ed-sm_ha1-alg_sd0dot05.csv", filter={}, subplot_field="alg", subplot_rows=1, subplot_cols=2,
x_field="k_hd", y_field="Avg-Max_Supply-diff(Watts)", z_field="delta-sign" , sharex=True, sharey=True, mean=True, save_to_file=True)


# below are the calls to functions. Uncomment depending on whether you want to execute algorithm OR
# want to print the graph from the results
# call_ed()
# print_graph()    

if __name__ == "__main__":
    import doctest
    
    (failures, tests) = doctest.testmod(report=True, verbose=False)
    print(f"failures: {failures}, tests: {tests}")

