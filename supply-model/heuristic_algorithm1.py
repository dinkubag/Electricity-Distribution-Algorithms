#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:24:47 2025

@author: dineshkumarbaghel
"""
import numpy as np
from typing import List, Tuple
import ffk 
import logging
import math
import time
import random
import copy
# required python3.7 or higher
from dataclasses import dataclass
from solution import Solution
import experiments_csv
from matplotlib import pyplot as plt
import datetime

# =============================================================================
# algorithmm v2 and the required functions    
# =============================================================================

def compute_solution(alg:str, remaining_new_demands: List[Tuple], sorted_std:List[Tuple], vsi_list:List[Tuple], rem_supply:float, g:int, k_ofhighest_demand:int) -> ffk.BinsArray :
    '''
    It returns a BinsArray.

    Parameters
    ----------
    remaining_new_demands : List[Tuple]
        DESCRIPTION.
    rem_supply : float
        DESCRIPTION.
    begin : int
        DESCRIPTION.

    Returns
    -------
    Solution
        DESCRIPTION.

    '''
    final_remaining_new_demands = []
    for demand in remaining_new_demands:
        # use deepcopy if the demand is not a tuple.
        final_remaining_new_demands.append(demand)
    # final_remaining_new_demands = copy.deepcopy(remaining_new_demands)
    
    # variables related to g items
    g_items_list = []
    g_items_sum = 0
    
    for i in range(g):
        # g_items_list.append(copy.deepcopy(sorted_smallest_tolargest_demand[i]))
        g_items_list.append(sorted_std[i])
        g_items_sum += sorted_std[i][0]

    # remove the smallest g items from final remaining agents
    for i in range(g):
        final_remaining_new_demands.remove(g_items_list[i])

    # Update the supply
    final_rem_supply = rem_supply - g_items_sum
    
    #  Compute the agents_k_vector: START HERE
    highest_demand= max(final_remaining_new_demands)
    # lowest_demand = min(final_remaining_new_demands)
    
    numof_agents = len(final_remaining_new_demands)
    
    # for a given k of highest demand we determine the agents k vector for the demands in final_remaining_new_demands
    agents_k_vector = [0 for i in range(numof_agents)]
    for idx in range(numof_agents):
        agents_k_vector[idx] = (round((k_ofhighest_demand*highest_demand[0])/final_remaining_new_demands[idx][0],0), final_remaining_new_demands[idx][1])
    
    # uncomment the below if you want to test the function.
    # logging.debug(f"packing final remaining demands: {final_remaining_new_demands}, updated supply: {final_rem_supply}")
    
    binner = ffk.bkc_ffk()
    if alg == "online":
        solution = ffk.online_ffk_supply_withvectork_test(binner, final_rem_supply, final_remaining_new_demands, agents_k_vector)
    elif alg == "online_ffk":
        solution = ffk.online_ffk(binner, final_rem_supply, final_remaining_new_demands, k_ofhighest_demand)
    elif alg == "decreasing_ffdk":
        solution = ffk.decreasing_ffk(binner, final_rem_supply, final_remaining_new_demands, k_ofhighest_demand)
    else:
        solution = ffk.decreasing_ffk_supply_withvectork_test(binner, final_rem_supply, final_remaining_new_demands, agents_k_vector)
    
    numof_bins = binner.numbins(solution)

    #  adding the vs items
    vsi_count = len(vsi_list)
    # numof_bins = len(lists)
    for ibin in range(numof_bins):
        for vs_item_idx in range(vsi_count):
            binner.add_item_to_bin(bins = solution, item=vsi_list[vs_item_idx], bin_index=ibin)
    
    # adding the g items
    for ibin in range(numof_bins):
        for s_item_idx in range(g):
            binner.add_item_to_bin(bins=solution, item=g_items_list[s_item_idx], bin_index=ibin)
    
    # compute the final agents_k_vector
    final_agents_k_vector = []
    for vsi in vsi_list:
        final_agents_k_vector.append((numof_bins, vsi[1]))
    
    for g_item in g_items_list:
        final_agents_k_vector.append((numof_bins, g_item[1]))
    
    if alg == "online_ffk" or alg == "decreasing_ffdk":
        for demand in final_remaining_new_demands:
            final_agents_k_vector.append((k_ofhighest_demand, demand[1]))
    else:
        for k_tuple in agents_k_vector:
            final_agents_k_vector.append(k_tuple)
    
    sums, lists = solution
    
    grouped_items_list = g_items_list + vsi_list
    
    return (sums, lists, final_agents_k_vector, grouped_items_list)

def compute_egal_alg1_v2(alg:str, demands: List[float], supply: int, demand_tomaxdemand_ratio:float = 0, k_ofhighest_demand:int=1):
    '''
    Implementing the original algorithm suggested by Erel sir on 3-Ddec-2024.

    Parameters
    ----------
    demands : List[Float]
        DESCRIPTION.
    supply : Int
        DECSRIPTION.

    Returns
    -------
    None.
    
    Test Cases
    Uncomment the below test cases if you want to test this main function.
    
    # >>> compute_egal_alg1_v2("FFk", [0.5, 1.5, 1], 3)

    # >>> compute_egal_alg1_v2("FFk", [0.0135, 0.4865, 1.5, 1], 3)
    
    # >>> compute_egal_alg1_v2("FFk", [0.5, 1.5, 1], 2)
    
    
    # >>> compute_egal_alg1_v1("FFk", [0.0135, 0.4865, 1.5, 1], 2)
    
    # >>> soln = compute_egal_alg1_v2("FFk", [0.2, 0.4, 0.8, 1.7, 3.0, 6.5, 14], 21, 0.01, 20)
    
    # >>> compute_egal_alg1_v2("FFk", [0.2,0.22, 0.4,0.42, 0.8,0.82, 1.7,1.7, 3,3.2, 6.5,6.7, 14,14.2], 21, 0.01, 3)
    
    # >>> soln = compute_egal_alg1_v2("FFk", [2, 4, 8, 17, 30, 30, 30, 30, 30, 30, 30, 30, 65, 65, 65, 65, 140], 210, 0.01, 20)
    
    # >>> soln = compute_egal_alg1_v2("FFk", [2.700083334, 2.540450001, 2.835183334, 0.599683334, 4.759250001, 1.026016668, 1.488833334, 2.600466668, 1.281950001, 0.386350001, 4.414050001, 0.801666668, 5.497566668, 2.798550001, 4.896533334, 5.330000001, 2.366533334, 6.063500001, 2.591333334, 3.726916668, 0.421733334, 4.301166668, 0.701066668, 2.625000001, 9.653816668, 1.633150001, 0.974733334, 2.966650001, 3.914716668, 4.019600001, 2.396966668, 1.905533334, 3.394583334, 2.845650001, 2.215733334, 5.022533334, 3.233833334, 2.472416668, 6.173300001, 3.058216668, 1.923483334, 1.768633334, 1.024283334, 0.611016668, 1.088533334, 1.091150001, 0.220150001, 2.017833334, 3.721483334, 0.258500001, 6.989550001, 2.143800001, 4.873550001, 2.968433334, 5.914333334, 1.666600001, 0.788833334, 6.582400001, 5.950000001, 1.257683334, 3.868466668, 5.023050001, 0.492733334, 0.584550001, 4.594050001, 3.682466668, 2.024916668, 0.638850001, 3.511950001, 3.666133334, 0.734533334, 0.154716668, 4.962216668, 1.431266668, 5.612550001, 3.322016668, 3.558416668, 2.931700001, 3.931666668, 1.035350001, 4.948133334, 2.737066668, 0.219333334, 4.456433334, 3.374966668, 0.541433334, 3.520800001, 3.399616668, 1.601066668, 1.778683334, 4.758666668, 2.744516668, 2.666716668, 3.454666668, 2.345183334, 0.507066668, 3.263533334, 2.752116668, 3.460716668, 3.773266668, 1.351950001, 4.432583334, 0.331816668, 2.011066668, 1.219316668, 1.138300001, 3.135950001, 3.216700001, 2.076983334, 1.724300001, 2.663700001, 2.922266668, 0.270050001, 9.054333334, 1.557033334, 5.678250001, 7.088883334, 1.458450001, 1.272633334, 2.057350001, 4.935533334, 1.072416668, 1.419366668, 8.141183334, 7.736866668, 2.401333334, 2.604866668, 2.684116668, 5.808483334, 2.994100001, 2.091250001, 2.556733334, 2.667600001, 2.443983334, 0.767616668, 5.081933334, 0.177483334, 4.400100001, 1.396216668, 4.765300001, 1.033133334, 1.218600001, 3.267900001, 0.748750001, 9.268383334, 5.649966668, 2.460616668, 0.615516668, 3.995733334, 1.907650001, 2.127016668, 0.774000001, 2.900166668, 0.404283334, 3.037816668, 1.175416668, 3.506466668, 4.286900001, 6.350866668, 0.195783334, 2.113950001, 1.905716668, 4.004766668, 2.124450001, 1.863483334, 0.175933334, 0.147650001, 4.258066668, 5.693633334, 1.382033332, 1.017883334, 1.089666668, 1.438516668, 0.651100001, 4.897416668, 2.609866668, 2.676466668, 4.596816668, 5.598816668, 5.596516668, 5.447650001, 4.999016668, 2.102650001, 4.104150001, 1.415866668, 0.918283334, 1.404550001, 4.084316668, 3.587783334, 6.275616668, 1.969466668, 5.760633334, 1.117166668, 1.610183334, 1.625866668, 1.741866668, 1.839116668, 1.158800001, 2.331433334, 1.417966668, 4.963250001, 0.569566668, 2.595233334, 2.337333334, 0.206216668, 5.051983334, 3.811800001, 3.539366668, 7.677366668, 3.387466668, 3.468900001, 3.917000001, 2.684683334, 2.613566668, 2.575000001, 1.745533334, 3.602150001, 0.330416668, 2.397616668, 3.583933334, 4.268433334, 4.255700001, 3.310383334, 7.184166668, 1.258500001, 0.582600001, 1.327133334, 4.632733334, 0.382666668, 4.833950001, 2.162116668, 5.959133334, 2.923566668, 3.260333334, 2.384300001, 3.294650001, 0.433216668, 2.178500001, 1.311050001, 3.893750001, 4.463366668, 0.523933334, 13.27948333, 0.136933334, 6.627116668, 0.387416668, 2.928850001, 3.550483334, 2.022250001, 6.503000001, 3.444166668, 3.752083334, 1.009683334, 3.237033334, 9.167716668, 3.846350001, 0.763133334, 2.858283334, 2.553733334, 4.055483334, 2.858883334, 4.162933334, 5.339766668, 3.233450001, 4.041200001, 3.248733334, 1.291400001, 1.857716668, 2.272500001, 1.684300001, 3.293533334, 3.300650001, 3.743083334, 1.371500001, 2.900583334, 3.144316668, 3.256966668, 3.145000001, 0.270100001, 5.199900001, 2.459200001, 2.117700001, 3.183166668, 3.298516668, 0.731283334, 3.950966668, 4.909916668, 3.034083334, 3.120800001, 6.647466668, 2.305516668, 8.798700001, 1.534800001, 1.882766668, 4.574450001, 2.531300001, 1.891200001, 4.558666668, 2.969766668, 1.841683334, 0.167566668, 0.737516668, 1.712850001, 1.053283334, 3.554833334, 2.290883334, 0.240633334, 2.364366668, 0.106516668, 2.306116668, 0.962483334, 1.607216668, 1.906916668, 3.057200001, 4.076983334, 0.124516668, 1.220200001, 2.197083334, 2.336966668, 2.329133334, 2.622883334, 0.250916668, 3.876083334, 5.640450001, 4.912633334, 2.310483334, 3.117700001, 2.524983334, 2.031700001, 2.688433334, 6.495916668, 1.223866668, 2.239433334, 2.424933334, 3.512416668, 1.479533334, 2.715183334, 3.839183334, 5.760183332, 4.131216668, 3.049183334, 3.143200001, 2.363666668, 8.577766668, 2.540766668, 1.348616668, 2.468366668, 7.243483334, 0.855066668, 14.9387, 3.336466668, 0.118050001, 0.595533334, 2.271900001, 2.039683334, 2.042950001, 2.459533334, 0.129083334, 0.275166668, 11.50521667, 1.665083334, 3.351716668, 2.763516668, 1.862533334, 3.913300001, 4.569900001, 6.006883334], 754, 0.01, 1)
    
    # >>> compute_egal_alg1_v2("FFk", [0.106516668,0.118050001,0.124516668,0.129083334,     0.136933334,     0.147650001,     0.154716668,     0.167566668,     0.175933334,0.177483334,     0.195783334,     0.206216668,     0.219333334,     0.220150001,0.240633334,     0.250916668,     0.258500001,0.270050001,0.270100001,0.275166668,     0.330416668,0.331816668,     0.382666668,     0.386350001,     0.387416668,0.404283334,     0.421733334,     0.433216668,     0.492733334,     0.507066668,     0.523933334,0.541433334,     0.569566668,     0.582600001,     0.584550001,     0.595533334,     0.599683334,     0.611016668,     0.615516668,0.638850001,     0.651100001,     0.701066668,     0.731283334,     0.734533334,     0.737516668,     0.748750001,     0.763133334,     0.767616668,0.774000001,     0.788833334,     0.801666668,     0.855066668,     0.918283334,     0.962483334,     0.974733334,     1.009683334,     1.017883334,     1.024283334,     1.026016668,     1.033133334,     1.035350001,     1.053283334,     1.072416668,     1.088533334,     1.089666668,     1.091150001,     1.117166668,     1.138300001,     1.158800001,     1.175416668,     1.218600001,     1.219316668,     1.220200001,     1.223866668,     1.257683334,     1.258500001,     1.272633334,     1.281950001,     1.291400001,     1.311050001,     1.327133334,     1.348616668,     1.351950001,     1.371500001,     1.382033332,     1.396216668,     1.404550001,     1.415866668,     1.417966668,     1.419366668,     1.431266668,     1.438516668,     1.458450001,     1.479533334,     1.488833334,     1.534800001,     1.557033334,     1.601066668,     1.607216668,     1.610183334,     1.625866668,     1.633150001,     1.665083334,     1.666600001,     1.684300001,     1.712850001,     1.724300001,     1.741866668,     1.745533334,     1.768633334,     1.778683334,     1.839116668,     1.841683334,     1.857716668,     1.862533334,     1.863483334,     1.882766668,     1.891200001,     1.905533334,     1.905716668,     1.906916668,     1.907650001,     1.923483334,     1.969466668,     2.011066668,     2.017833334,     2.022250001,     2.024916668,     2.031700001,     2.039683334,     2.042950001,     2.057350001,     2.076983334,     2.091250001,     2.102650001,     2.113950001,     2.117700001,     2.124450001,     2.127016668,     2.143800001,     2.162116668,     2.178500001,     2.197083334,     2.215733334,     2.239433334,     2.271900001,     2.272500001,     2.290883334,     2.305516668,     2.306116668,     2.310483334,     2.329133334,     2.331433334,     2.336966668,     2.337333334,     2.345183334,     2.363666668,     2.364366668,     2.366533334,     2.384300001,     2.396966668,     2.397616668,     2.401333334,     2.424933334,     2.443983334,     2.459200001,     2.459533334,     2.460616668,     2.468366668,     2.472416668,     2.524983334,     2.531300001,     2.540450001,     2.540766668,     2.553733334,     2.556733334,     2.575000001,     2.591333334,     2.595233334,     2.600466668,     2.604866668,     2.609866668,     2.613566668,     2.622883334,     2.625000001,     2.663700001,     2.666716668,     2.667600001,     2.676466668,     2.684116668,     2.684683334,     2.688433334,     2.700083334,     2.715183334,     2.737066668,     2.744516668,     2.752116668,     2.763516668,     2.798550001,     2.835183334,     2.845650001,     2.858283334,     2.858883334,     2.900166668,     2.900583334,     2.922266668,     2.923566668,     2.928850001,     2.931700001,     2.966650001,     2.968433334,     2.969766668,     2.994100001,     3.034083334,     3.037816668,     3.049183334,     3.057200001,     3.058216668,     3.117700001,     3.120800001,     3.135950001,     3.143200001,     3.144316668,     3.145000001,     3.183166668,     3.216700001,     3.233450001,     3.233833334,     3.237033334,     3.248733334,     3.256966668,     3.260333334,     3.263533334,     3.267900001,     3.293533334,     3.294650001,     3.298516668,     3.300650001,     3.310383334,     3.322016668,     3.336466668,     3.351716668,     3.374966668,     3.387466668,     3.394583334,     3.399616668,     3.444166668,     3.454666668,     3.460716668,     3.468900001,     3.506466668,     3.511950001,     3.512416668,     3.520800001,     3.539366668,     3.550483334,     3.554833334,     3.558416668,     3.583933334,     3.587783334,     3.602150001,     3.666133334,     3.682466668,     3.721483334,     3.726916668,     3.743083334,     3.752083334,     3.773266668,     3.811800001,     3.839183334,     3.846350001,     3.868466668,     3.876083334,     3.893750001,     3.913300001,     3.914716668,     3.917000001,     3.931666668,     3.950966668,     3.995733334,     4.004766668,     4.019600001,     4.041200001,     4.055483334,     4.076983334,     4.084316668,     4.104150001,     4.131216668,     4.162933334,     4.255700001,     4.258066668,     4.268433334,     4.286900001,     4.301166668,     4.400100001,     4.414050001,     4.432583334,     4.456433334,     4.463366668,     4.558666668,     4.569900001,     4.574450001,     4.594050001,     4.596816668,     4.632733334,     4.758666668,     4.759250001,     4.765300001,     4.833950001,     4.873550001,     4.896533334,     4.897416668,     4.909916668,     4.912633334,     4.935533334,     4.948133334,     4.962216668,     4.963250001,     4.999016668,     5.022533334,     5.023050001,     5.051983334,     5.081933334,     5.199900001,     5.330000001,     5.339766668,     5.447650001,     5.497566668,     5.596516668,     5.598816668,     5.612550001,     5.640450001,     5.649966668,     5.678250001,     5.693633334,     5.760183332,     5.760633334,     5.808483334,     5.914333334,     5.950000001,     5.959133334,     6.006883334,     6.063500001,     6.173300001,     6.275616668,     6.350866668,     6.495916668,     6.503000001,     6.582400001,     6.627116668,     6.647466668,     6.989550001,     7.088883334,     7.184166668,     7.243483334,     7.677366668,     7.736866668,     8.141183334,     8.577766668,     8.798700001,     9.054333334,     9.167716668,     9.268383334,     9.653816668,     11.50521667,     13.27948333,     14.9387], 754)
    
    
    '''
    
    # Uncomment the below line if you want to compute the execution time of this function. Below line starts the timer.
    # start = time.perf_counter()
    
    
    if __name__ == "__main__":
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename="heuristic_algorithm1_v2_github.log", encoding="utf-8", level=logging.DEBUG, force=True)
    else:
        logger = logging.getLogger("compute_egal_alg1_v2_github")
    
    
    if all(isinstance(item, tuple) for item in demands):
        if not all(len(item)==2 for item in demands):
            raise Exception("Some tuples are not in the format (item_value, index)")
        else:
            new_demands = demands
    else:
        if any(isinstance(item, tuple) for item in demands):
            raise Exception("Some items are not in the format (item_value, index)")
        else:
            new_demands = ffk.itemslist_to_tuplelist(demands)
    

    # logging--------START
    # uncomment if you want to test this function.
    # logger.info(f"ALGORITHM: {alg}, Agents: {new_demands}, Supply: {supply}") 
    # logger.debug("alg, g, k")
    # logging---------END
    alg = "online" if alg == "FFk" else "decreasing"
    
    max_demand_agent = max(new_demands)
    
    # numof_agents = len(new_demands)
    
    # computing the index 'i' of min_permanent_agent s.t. deman_of_i/max_demand >= 0.01
    smallest_demand = (math.inf, math.inf)
    for demand in new_demands:
        if demand[0]/max_demand_agent[0] >= demand_tomaxdemand_ratio and demand[0] < smallest_demand[0]:
            smallest_demand = demand
            
    # In a sorted_new_demands determine the largest_demand such that the sum of all the demands less than (or equal to)
    # largest_demand do not exceed the supply - max_demand
    sorted_new_demands = sorted(new_demands, key=lambda demand:demand[0])
    largest_demand = (None, None)
    sumof_demands = 0
    for demand in sorted_new_demands:
        if sumof_demands + demand[0] <= supply - max_demand_agent[0]:
            sumof_demands += demand[0]
            largest_demand = demand
        else:
            break
    
    # COMPUTING ALL THE ITEMS WHICH ARE LESS THAN smallest_demand
    # use append only if demand is a tuple. Otherwise, use deepcopy.
    remaining_new_demands = []
    for demand in new_demands:
        remaining_new_demands.append(demand)
        
    vs_items_sum =0  # sum of the very small items. Initially 0.
    
    vs_items_list = []  # list of very small items.
    # items in vs_items_list are such that the value of k is very very large.

    # compute the vs_items_sum and vs_items_list
    for demand in sorted_new_demands:
        if demand[0] < smallest_demand[0]:
            # item is too small so add it to the vs_items_list.
            vs_items_list.append(demand)
            # remove demand from the remaining_new_demands list
            remaining_new_demands.remove(demand)
            # update vs_items_sum
            vs_items_sum += demand[0]
        else:
            break
    
    # rem_supply is the remaining supply after all small items are connected all the time.
    rem_supply = supply - vs_items_sum  
    
    # uncomment if you want to test this function
    # logger.debug(f"rem_supply = {rem_supply}")
    
    # the number of very small items
    vs_items_count = len(vs_items_list)
    
    # determine the set of demands between smallest_demand and largest_demand
    smallest_tolargest_demand = []
    sum_smallest_tolargest_demand = 0
    for demand in remaining_new_demands:
        if demand[0] >= smallest_demand[0] and demand[0] <= largest_demand[0] and sum_smallest_tolargest_demand + demand[0] <= rem_supply - max_demand_agent[0] :
            smallest_tolargest_demand.append(demand)
            sum_smallest_tolargest_demand += demand[0]
     
    # sorting the smallest_tolargest_demand
    sorted_smallest_tolargest_demand = sorted(smallest_tolargest_demand, key = lambda demand: demand[0])
    
    # uncomment if you want to test this function
    # logger.debug(f"sum_std={sum_smallest_tolargest_demand},rem supply={rem_supply}, final rem sup={rem_supply - sum_smallest_tolargest_demand}\n")
    # logger.debug(f"Demands that can be grouped: {sorted_smallest_tolargest_demand}")
    
    # number of items in sorted_smallest_tolargest_demand
    numof_items_instd = len(smallest_tolargest_demand)
    
    
    # uncomment if you want to test this function
    # logger.debug(f"num of items in instd: {numof_items_instd}")
    
    best_solution = None
    
    # checking if all the items can be packed into a single bin
    if sum([item[0] for item in remaining_new_demands]) <= rem_supply:
    
        final_rem_supply = rem_supply
        
        each_agents_k_vector = [(1, item[1]) for item in remaining_new_demands]
        
        binner = ffk.bkc_ffk()
        if alg == "online":
            solution = ffk.online_ffk_supply_withvectork_test(binner, final_rem_supply, remaining_new_demands, each_agents_k_vector)
        else:
            solution = ffk.decreasing_ffk_supply_withvectork_test(binner, final_rem_supply, remaining_new_demands, each_agents_k_vector)
        sums, lists = solution
        
        #  adding the vs items
        numof_bins = len(lists)
        for ibin in range(numof_bins):
            for vs_item_idx in range(vs_items_count):
                binner.add_item_to_bin(bins = solution, item=vs_items_list[vs_item_idx], bin_index=ibin)
        sums, lists = solution
        
        # Now we will add this solution to the list of different solutions.
        # sums, lists = solution
        soln  = Solution(sums, lists, new_demands, each_agents_k_vector, group_size = vs_items_count, k=1)
    
        best_solution = soln
        # best_solution_agents_k_vector = each_agents_k_vector
        
        best_solution.compute_agents_connection_vector()
        best_solution.compute_agents_egal_vector()
        best_solution.solution_egalconn_val(0)
        best_solution.solution_egalsupply_val(0)
        best_solution.compute_asr_vector()
        
        # uncomment if you want to test this function
        # logger.debug(f"{alg}, {1}, {best_solution.min_asr[0]}")
        # return best_solution_sofar, different_solutions

        # uncomment the below line if using the main algorithm on the electricity data, and comment it if testing this function.
        return best_solution
    
        # uncomment the below line if testing this function
        # return
        
    # we use the ternary search here.
    begin = 0
    end = numof_items_instd
    
    while end - begin > 3:
        # compute the solution for g = begin
        soln_begin = compute_solution(alg, remaining_new_demands, sorted_smallest_tolargest_demand, vs_items_list, rem_supply, begin, k_ofhighest_demand)
        # compute the solution for g = end
        soln_end = compute_solution(alg, remaining_new_demands, sorted_smallest_tolargest_demand, vs_items_list, rem_supply, end, k_ofhighest_demand)
        
        sums_begin, lists_begin, agents_k_vector_begin, grouped_items_list_begin = soln_begin
        sums_end, lists_end, agents_k_vector_end, grouped_items_list_end = soln_end
        
        # uncomment the below log statement if you are testing this function.
        # logger.debug(f"AFTER ADDING VS ITEMS: {vs_items_list}")
        # logger.debug(f"sums: {sums}, lists: {lists}")
        
        # uncommen the below logging statements if you are testing this function.
        # logger.debug(f"After adding g items: {g_items_list}")
        # logger.debug(f"sums: {sums}, lists: {lists}")
        
        # Now we will add this solution to the list of different solutions.
        solution_begin  = Solution(sums_begin, lists_begin, new_demands, agents_k_vector_begin, group_size = begin + vs_items_count, vs=vs_items_count, g=begin, group_list=grouped_items_list_begin)
        solution_end  = Solution(sums_end, lists_end, new_demands, agents_k_vector_end, group_size = end + vs_items_count, vs=vs_items_count, g=end, group_list=grouped_items_list_end)
        
        solution_begin.compute_agents_connection_vector()
        solution_begin.compute_agents_egal_vector()
        solution_begin.solution_egalconn_val(0)
        solution_begin.solution_egalsupply_val(0)
        solution_begin.compute_asr_vector()
        
        solution_end.compute_agents_connection_vector()
        solution_end.compute_agents_egal_vector()
        solution_end.solution_egalconn_val(0)
        solution_end.solution_egalsupply_val(0)
        solution_end.compute_asr_vector()
        
        # uncomment the below logging statements if you are testing this function.
        # logger.debug(f"FINAL SOLUTION:")
        # logger.debug(f"sums: {sums}, lists: {lists}")
        
        if solution_end > solution_begin and (best_solution == None or solution_end > best_solution) :
            best_solution = solution_end
            
        elif best_solution == None or solution_begin > best_solution :
            best_solution = solution_begin
            
        # update begin and end
        begin = int(round((begin*2 + end)/3, 0))
        end = int(round((begin + end*2)/3, 0))
            
        #  Now we will add this best solution for the current iteration g to the different solutions list

    for g in range(begin, end+1):
        soln_g = compute_solution(alg, remaining_new_demands, sorted_smallest_tolargest_demand, vs_items_list, rem_supply, g, k_ofhighest_demand)
    
        sums, lists, agents_k_vector, grouped_items_list = soln_g
        
        solution_g = Solution(sums, lists, new_demands, agents_k_vector, group_size = g + vs_items_count, vs=vs_items_count, g=g, group_list=grouped_items_list)
        solution_g.compute_agents_connection_vector()
        solution_g.compute_agents_egal_vector()
        solution_g.solution_egalconn_val(0)
        solution_g.solution_egalsupply_val(0)
        solution_g.compute_asr_vector()
        
        if best_solution == None or solution_g > best_solution:
            best_solution = solution_g
    
    # logging ---------- START
    # uncomment the below logging statements if you are testing this function.
    # logger.info("****FINAL SOLUTION****")
    # logger.debug(f"Sums: {best_solution.sums}, Lists: {best_solution.lists}")
    # logger.debug(f"Egal Connection List: {best_solution.agents_conn_vector}")
    # logger.debug(f"Egal Supply List: {best_solution.agents_egal_vector}")
    # logger.debug(f"ASR List: {best_solution.asr_vector}")
    # logger.debug(f"Total Watts={sum([item[0] for item in best_solution.agents_egal_vector])}")
    # logger.debug(f"Total Connections(time)={sum([item[0] for item in best_solution.agents_conn_vector])}")
    # logger.debug(f"Min ASR: {best_solution.min_asr}")
    # max_sup, min_sup = best_solution.determine_max_and_min()
    # logger.debug(f"g: {best_solution.g}")
    # logger.debug(f"max supply: {max_sup}, min supply: {min_sup}, max supply diff: {max_sup - min_sup}")
    # logger.info("\n\n\n\n")
    # end = time.perf_counter()
    # commenting the below line as I am running the main algorithm on electricity data
    # logger.info(f"Time taken for the entire function: {end - start:0.4f} seconds")
    # logging ---------- END

    # comment the below statement if testing this function.
    return best_solution

    # uncomment the below line if testing this function
    # return

if __name__ == "__main__":
    import doctest
    
    (failures, tests) = doctest.testmod(report=True, verbose=False)
    print(f"Failures: {failures}, Tests: {tests}")