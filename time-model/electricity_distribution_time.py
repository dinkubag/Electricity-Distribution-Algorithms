#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:39:18 2022

@author: dineshkumarbaghel
"""

import json
import ffk
import numpy as np
import experiments_csv
import os
import logging
import datetime

def read_data(filename:str, test_data_bool:bool=False) -> tuple:
    '''
    It reads the data from the data directory.

    Returns
    -------
    tuple
        DESCRIPTION.

    '''
    if test_data_bool == False:
        # Data developing_country_household_consumption_data.csv is the original file.
        # See README.md to download this file.
        fr = open("./../data/developing_country_household_consumption_data.csv",'r')
    else:
        fr = open("./../data/"+filename+".csv",'r')
    
    agents_comfort_vector = []    #list of agents comfort vector.
    agents_consumption_vector = []   #list for agents consumption of electricity.
    agents_consumption_vector_byhour = []   # list of agents consumption vector by hour.
    agents_comfort_vector_byhour = []   # list of agents confort vector by hour
    
    agents_set = set()           #set of agents.
    hrs_set = set()          #set of hours.
    l = fr.readline()
    
    for l in fr:
        agents_set.add(int(l.rstrip('\n').split(',')[2]))
        hrs_set.add(int(l.rstrip('\n').split(',')[0]))
    
    tot_hrs = len(list(hrs_set)) #total number of hours
    numof_agents = len(list(agents_set))
    
    #creating list of empty lists for agent comfort and consumption.
    for i in range(numof_agents):    
        agents_comfort_vector.append([])
        agents_consumption_vector.append([]) 
       
    # initialize agents consumption by hour i.e. for each hour there is a vector of consumptions of agents
    # of length numof_agents
    for hr in range(tot_hrs):
        agents_consumption_vector_byhour.append([])
        agents_comfort_vector_byhour.append([])
        
        
    fr.seek(0)
    line = fr.readline()
    for line in fr:
        agent_ID = int(line.rstrip('\n').split(',')[2])
        hr_ID = int(line.rstrip('\n').split(',')[0])
        agents_comfort_vector[agent_ID-1].append(float(line.rstrip('\n').split(',')[3]))
        agents_comfort_vector_byhour[hr_ID-1].append(float(line.rstrip('\n').split(',')[3]))
        
        agents_consumption_vector[agent_ID-1].append(float(line.rstrip('\n').split(',')[1]))
        agents_consumption_vector_byhour[hr_ID - 1].append(float(line.rstrip('\n').split(',')[1]))
        
    
    if test_data_bool == False:
        if os.path.exists('./../data/data_comfort.csv') and os.path.exists('./../data/data_consumption.csv'):
            print("Files: data_comfort.csv and data_consumption.csv exists.")
        else:
            fw_comfort = open("./../data/data_comfort.csv",'w+')
            fw_consumption = open("./../data/data_consumption.csv",'w+') 
        
        
        
            for i in range(len(agents_comfort_vector)):
                json.dump(agents_comfort_vector[i],fw_comfort)
                json.dump(agents_consumption_vector[i],fw_consumption)
                if i < len(agents_comfort_vector)-1:
                    fw_comfort.write('\n')
                    fw_consumption.write('\n')
            
            
            fw_comfort.close()
            fw_consumption.close()
    else:
        if os.path.exists('./../data/'+filename+'_data_comfort.csv') and os.path.exists('./../data/'+filename+'_data_consumption.csv'):
            print(f"Files: {filename}_data_comfort.csv and {filename}_data_consumption.csv exists.")
        else:
            fw_comfort = open('./../data/'+filename+'_data_comfort.csv','w+')
            fw_consumption = open('./../data/'+filename+'_data_consumption.csv','w+') 
        
        
        
            for i in range(len(agents_comfort_vector)):
                json.dump(agents_comfort_vector[i],fw_comfort)
                json.dump(agents_consumption_vector[i],fw_consumption)
                if i < len(agents_comfort_vector)-1:
                    fw_comfort.write('\n')
                    fw_consumption.write('\n')
            
            
            fw_comfort.close()
            fw_consumption.close()
        
        
    hesc = sum([sum(vector) for vector in agents_consumption_vector])/tot_hrs
    avg_consumption = hesc/numof_agents
    
    return tot_hrs, numof_agents, hesc, avg_consumption, agents_comfort_vector, agents_consumption_vector, agents_consumption_vector_byhour, agents_comfort_vector_byhour


def electricity_distribution(alg: str, K: int, delta: float, simulations: int, hours_ina_day: int = 24, supplyof_test_data: list = [0], test_data_bool: bool = False, filename: str='') -> dict:
    '''
    

    Returns
    -------
    None.
    
    
    Test Cases:
    # >>> electricity_distribution("FFk", K=1, delta=0, simulations=1, hours_ina_day=1, supplyof_test_data=[2], test_data_bool = True, filename='test_data1')
    # >>> electricity_distribution("FFk", K=1, delta=0, simulations=1, hours_ina_day=1, supplyof_test_data=[2,3], test_data_bool = True, filename='test_data2')
    # >>> electricity_distribution("FFk", K=2, delta=0, simulations=1, hours_ina_day=1, supplyof_test_data=[2,3], test_data_bool = True, filename='test_data2')
    # >>> electricity_distribution("FFk", K=2, delta=0, simulations=1, hours_ina_day=2, supplyof_test_data=[2,3], test_data_bool = True, filename='test_data2')
    
    # >>> electricity_distribution("FFk", K=1, delta=0, simulations=1, test_data_bool = False, filename='')

    '''
    if __name__ == "__main__":
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename="time_model_electricity_distribution.log", encoding="utf-8", level=logging.DEBUG, force=True)
    else:
        logger = logging.getLogger("Time-Model: Electricity Distribution Algorithm")
    
    logger.info(datetime.datetime.now())
    
    data_vector = read_data(filename,test_data_bool)
    agents_consumption_vector_byhour = data_vector[6]
    agent_comfort_vector = data_vector[4]
    agent_consumption_vector = data_vector[5]
    agents_comfort_vector_byhour = data_vector[7]
    tot_hrs= data_vector[0]
    numof_agents = data_vector[1]    
    
    # below two lines are for logging. Uncomment if you are testing the function.
    # logger.debug(f"Agent Consumption Vector: {agent_consumption_vector} \nAgent Comfort Vector: {agent_comfort_vector}")
    # logger.debug(f"Total Hours: {tot_hrs}, Number of Agents: {numof_agents}")
    
    
    avg_total_connection = []
    avg_tot_hours = []
    avg_connection_EF = []
    
    avg_total_comfort = []
    avg_acs = []
    avg_comfort_EF = []
    
    avg_total_supply = []
    avg_asr = []
    avg_supply_EF = []
    
    alg = "online" if alg == "FFk" else "decreasing"
    
    for sml in range(simulations):
        total_connection = 0    # Sum of the total connection time of all the agents.
        tot_hour_connection = 0 #out of total number of hours for how many hours all agents are connected.
        agent_total_connection = np.zeros(numof_agents) # the total connection time for each agent
        
        total_comfort = 0   # total comfort received of all the agents during their connection time.
        agent_total_comfort = np.zeros(numof_agents)    #agent's total comfort in their connection time.
        acs_vector = np.zeros(numof_agents)     # agent's agent comfort rate. Defined as comfort-delivered/total-agent-comfort
        
        total_supply = 0        # sum of the total supply delivered of all the agents during their connection hrs.
        agent_total_supply = np.zeros(numof_agents) # agent's total supply during their connection time.
        asr_vector = np.zeros(numof_agents) # agent's agent supply rate. Defined as supply-delivered/total-agent-supply
        
        hours = 0   # define how many hours have passed.
        estimated_dayahead_agents_consumption_vector = []
        
        # derived agents consumption vector
        derived_agents_consumption_vector = []
        for each_agent in range(numof_agents):
            derived_agents_consumption_vector.append([])
            
        while hours < tot_hrs:
            # calculate the agents consumption vector a day ahead using the normal distribution.
            # For more details see page 11 in LINK to Olabambo's recent paper.
            for each_hour in range(hours, hours+hours_ina_day):
                dayahead_hourly_agents_consumption = []
                for each_agent in range(numof_agents):
                    eachhr_each_agent_consumption = abs(np.random.normal(agents_consumption_vector_byhour[each_hour][each_agent], delta))
                    derived_agents_consumption_vector[each_agent].append(eachhr_each_agent_consumption)
                    dayahead_hourly_agents_consumption.append(eachhr_each_agent_consumption)
                estimated_dayahead_agents_consumption_vector.append(dayahead_hourly_agents_consumption)
                
            # logging. Uncommnent the below line when testing this function.
            # logger.debug(f"Estimated Day Ahead Agents Consumption Vector: {estimated_dayahead_agents_consumption_vector}\n")
            
            # compute the day ahead supply
            binsize = sum([sum(estimated_dayahead_agents_consumption_vector[each_hour]) for each_hour in range(hours, hours + hours_ina_day)])/hours_ina_day
            
            for hr in range(hours, hours+hours_ina_day):    
                
                # When testing this function we require below two lines.
                # Test data may contain data for one hour or two hours etc. Thats why we need this.
                if test_data_bool == True:
                    binsize = supplyof_test_data[hr]
                
                # logging. Uncomment the below line when testing this function.
                # logger.debug(f"Supply at hour {hr} is {binsize}.")
                
                # use k-BP.
                if alg=='online':
                    if sum(estimated_dayahead_agents_consumption_vector[hr]) <= binsize:
                        bins = ffk.online_ffk(ffk.bkc_ffk(), binsize, estimated_dayahead_agents_consumption_vector[hr] , k = 1)
                        total_connection += (numof_agents)/len(bins[1])
                        tot_hour_connection += 1
                        for agent in range(numof_agents):
                            agent_total_connection[agent] += 1
                            
                            agent_total_comfort[agent] += agents_comfort_vector_byhour[hr][agent]
                            total_comfort += agents_comfort_vector_byhour[hr][agent]
                            
                            agent_total_supply[agent] += estimated_dayahead_agents_consumption_vector[hr][agent]
                            total_supply += estimated_dayahead_agents_consumption_vector[hr][agent]
                    else:
                        bins = ffk.online_ffk(ffk.bkc_ffk(), binsize, estimated_dayahead_agents_consumption_vector[hr] , k = K)
                        sums, lists = bins
                        total_connection += (numof_agents*K)/len(bins[1])
                        tot_hour_connection += K/len(bins[1])
                        for agent in range(numof_agents):
                            agent_total_connection[agent] += (K/len(bins[1]))
                            
                            agent_total_comfort[agent] += agents_comfort_vector_byhour[hr][agent]*(K/len(bins[1]))
                            total_comfort += agents_comfort_vector_byhour[hr][agent]*(K/len(bins[1]))
                            
                            agent_total_supply[agent] += estimated_dayahead_agents_consumption_vector[hr][agent]*(K/len(bins[1]))
                            total_supply += estimated_dayahead_agents_consumption_vector[hr][agent]*(K/len(bins[1]))
                            
                    # logging---------------START.
                    # Uncomment all the lines in this block to test this function.
                    # logging.debug(f"Total Connection: {total_connection}")
                    # logging.debug(f"Egalitarian Hour Connection: {tot_hour_connection}")
                    # logging.debug(f"Total Comfort: {total_comfort}")
                    # logging.debug(f"Total Supply: {total_supply}")
                    # logging.debug(f"Each Agent Total Hour Connection: {agent_total_connection}")
                    # for agent in range(numof_agents):
                        # logging.debug(f"Agent {agent}'s total comfort: {agent_total_comfort[agent]}")
                        # logging.debug(f"Agent {agent}'s total supply: {agent_total_supply[agent]}\n")
                    # logging----------------END
                else:
                    if sum(estimated_dayahead_agents_consumption_vector[hr]) <= binsize:
                        bins = ffk.decreasing_ffk(ffk.bkc_ffk(), binsize, estimated_dayahead_agents_consumption_vector[hr] , k = 1)
                        total_connection += (numof_agents)/len(bins[1])
                        tot_hour_connection += 1
                        for agent in range(numof_agents):
                            agent_total_connection[agent] += 1
                            
                            agent_total_comfort[agent] += agents_comfort_vector_byhour[hr][agent]
                            total_comfort += agents_comfort_vector_byhour[hr][agent]
                            
                            agent_total_supply[agent] += estimated_dayahead_agents_consumption_vector[hr][agent]
                            total_supply += estimated_dayahead_agents_consumption_vector[hr][agent]
                            
                    else:
                        bins = ffk.decreasing_ffk(ffk.bkc_ffk(), binsize, estimated_dayahead_agents_consumption_vector[hr] , k = K)
                        sums, lists = bins
                        total_connection += (numof_agents*K)/len(bins[1])
                        tot_hour_connection += K/len(bins[1])
                        for agent in range(numof_agents):
                            agent_total_connection[agent] += (K/len(bins[1]))
                            
                            agent_total_comfort[agent] += agents_comfort_vector_byhour[hr][agent]*(K/len(bins[1]))
                            total_comfort += agents_comfort_vector_byhour[hr][agent]*(K/len(bins[1]))
                            
                            agent_total_supply[agent] += estimated_dayahead_agents_consumption_vector[hr][agent]*(K/len(bins[1]))
                            total_supply += estimated_dayahead_agents_consumption_vector[hr][agent]*(K/len(bins[1]))
                            
                    # logging---------------START
                    # Uncomment all the lines in this block to test this function.
                    # logging.debug(f"Total Connection: {total_connection}")
                    # logging.debug(f"Egalitarian Hour Connection: {tot_hour_connection}")
                    # logging.debug(f"Total Comfort: {total_comfort}")
                    # logging.debug(f"Total Supply: {total_supply}")
                    # logging.debug(f"Each Agent Total Hour Connection: {agent_total_connection}")
                    # for agent in range(numof_agents):
                    #     logging.debug(f"Agent {agent}'s total comfort: {agent_total_comfort[agent]}")
                    #     logging.debug(f"Agent {agent}'s total supply: {agent_total_supply[agent]}")
                    # logging----------------END
                    
            hours += hours_ina_day
        avg_total_connection.append(total_connection)
        avg_tot_hours.append(tot_hour_connection)
        avg_connection_EF.append(max(agent_total_connection) - min(agent_total_connection))
        
        avg_total_comfort.append(total_comfort)
        for agent in range(numof_agents):
            acs_vector[agent] = agent_total_comfort[agent] / sum(agent_comfort_vector[agent])
        avg_acs.append(min(acs_vector))
        avg_comfort_EF.append(max(acs_vector) - min(acs_vector))
        
        avg_total_supply.append(total_supply)
        for agent in range(numof_agents):
            asr_vector[agent] = agent_total_supply[agent] / sum(derived_agents_consumption_vector[agent])
        avg_asr.append(min(asr_vector))
        avg_supply_EF.append(max(asr_vector) - min(asr_vector))
        
        
    # logging------START
    # comment all the lines in this block if executing the code on real data.
    # logging.info("Final Result:")
    # logging.debug(f"Utilitarian(Connection_Hours): {sum(avg_total_connection)/simulations}")
    # logging.debug(f"Utilitarian(Connection_Hours)_SD: {np.std(avg_total_connection)}")
    # logging.debug(f"Egalitarian(Connection_Hours): {sum(avg_tot_hours)/simulations}")
    # logging.debug(f"Egalitarian(Connection_Hours)_SD: {np.std(avg_tot_hours)}")
    # logging.debug(f"Max-Util-diff(Connection_Hours): {sum(avg_connection_EF)/simulations}")
    # logging.debug(f"Max-Util-diff(Connection_Hours)_SD: {np.std(avg_connection_EF)}")
    # logging.info("\n\n")
    # logging.debug(f"Utilitarian(Comfort): {sum(avg_total_comfort)/simulations}")
    # logging.debug(f"Utilitarian(Comfort)_SD: {np.std(avg_total_comfort)}")
    # logging.debug(f"Egalitarian(Comfort) ACS: {sum(avg_acs)/simulations}")
    # logging.debug(f"Egalitarian(Comfort)_SD ACS: {np.std(avg_acs)}")
    # logging.debug(f"Max-Util-diff(Comfort) ACS: {sum(avg_comfort_EF)/simulations}")
    # logging.debug(f"Max-Util-diff(Comfort)_SD ACS: {np.std(avg_comfort_EF)}")
    # logging.info("\n\n")
    # logging.debug(f"Utilitarian(Supply): {sum(avg_total_supply)/simulations}")
    # logging.debug(f"Utilitarian(Supply)_SD: {np.std(avg_total_supply)}")
    # logging.debug(f"Egalitarian(Supply) ASR: {sum(avg_asr)/simulations}")
    # logging.debug(f"Egalitarian(Supply)_SD ASR: {np.std(avg_asr)}")
    # logging.debug(f"Max-Util-diff(Supply) ASR: {sum(avg_supply_EF)/simulations}") 
    # logging.debug(f"Max-Util-diff(Supply)_SD ASR: {np.std(avg_supply_EF)}")
    # logging.info("\n\n\n\n")
    # logging ---------------END

    # whenever you are testing this function, comment the below return block
    return {
            "Connection_Hours": "Connection_Hours",
            "Utilitarian(Connection_Hours)": sum(avg_total_connection)/simulations,
            "Utilitarian(Connection_Hours)_SD": np.std(avg_total_connection),
            "Egalitarian(Connection_Hours)": sum(avg_tot_hours)/simulations,
            "Egalitarian(Connection_Hours)_SD": np.std(avg_tot_hours),
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
            "Max-Util-diff(Supply)_SD": np.std(avg_supply_EF)
        }

def call_ed():
    '''
    This is a function to call electricity distribution function.

    Returns
    -------
    None.

    '''
    input_ranges = {
            "alg": ["FFk", "FFDk"],
            # "K": range(1,35,2),
            "K": [100],
            # "delta": [round(0.5*0.5,4), 0.5, round(1.5*0.5,4), round(2*0.5,4), round(2.5*0.5,4), round(3*0.5,4) ],
            "delta": [0.05],
            "simulations": [9]
        }
    
    
    
    ex = experiments_csv.Experiment("results/", "final_results_delta0dot05_2.csv", "results/backup")
    ex.run(electricity_distribution, input_ranges)

def print_graph():
    '''
    Once the results are generated, you can call this function to draw graphs.
    

    Returns
    -------
    None.

    '''
    experiments_csv.multi_plot_results("./results/final_results_diff-delta_withsymbol.csv", filter={}, subplot_field="alg", subplot_rows=1, subplot_cols=2,
x_field="K", y_field="Max-Util-diff(Comfort)", z_field="delta(symbol)" , sharex=True, sharey=True, mean=True, save_to_file=True)
    

call_ed()
print_graph()

if __name__ == "__main__":
    import doctest
    
    (failures, tests) = doctest.testmod(report=True, verbose=True)
    print(f"Failures:{failures}, Tests:{tests}")
