#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:32:51 2025

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
from dataclasses import dataclass, field
from solution import Solution

def vs_items(demands: List[Tuple], largest_item: Tuple, epsilon:float = 0) -> Tuple:
    '''
    This function computes the list of very small items.
    Very small items are those whose values < epsilon*largest item

    Parameters
    ----------
    demands : List[Tuple]
        DESCRIPTION.
    largest_item : Tuple
        DESCRIPTION.
    epsilon : float, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    Tuple
        DESCRIPTION.

    >>> vs_items([(0.5,0), (1.5,1), (1, 2)], (1.5,1), 0.01)
    ([], 0)
    
    >>> vs_items([(0.0135,0), (0.4865,1), (1.5,2), (1, 3)], (1.5,2), 0.01)
    ([(0.0135, 0)], 0.0135)
    
    >>> vs_items([(14.9387, 0), (0.106516668, 1), (0.118050001, 2), (0.124516668, 3), (0.129083334, 4), (0.136933334, 5), (0.147650001, 6), (0.154716668, 7)], (14.9387, 0), 0.01)
    ([(0.106516668, 1), (0.118050001, 2), (0.124516668, 3), (0.129083334, 4), (0.136933334, 5), (0.147650001, 6)], 0.762750006)

    '''
    vs_items_list = []
    
    vs_items_sum = 0
    
    threshold_value = epsilon*largest_item[0]
    
    for item in demands:
        if item[0] < threshold_value:
            vs_items_list.append(item)
            vs_items_sum += item[0]
    
    return vs_items_list, vs_items_sum
    

def compute_group_ofitems(demands:List[Tuple], r:int, max_demand:Tuple) -> List[Tuple]:
    '''
    Function computes the group of items whose size is between (max_demand/2^{r+1}, max_demand/2^r].

    Parameters
    ----------
    demands : List[Tuple]
        DESCRIPTION.
    r : int
        DESCRIPTION.
    max_demand : Tuple
        DESCRIPTION.

    Returns
    -------
    List[Tuple]
        DESCRIPTION.

    >>> compute_group_ofitems([(0.0135,0), (0.4865,1), (1.5,2), (1, 3)], 6, (1.5,2))
    [(0.0135, 0)]
    
    >>> compute_group_ofitems([(0.0135,0), (0.4865,1), (1.5,2), (1, 3)], 5, (1.5,2))
    []
    
    >>> compute_group_ofitems([(0.0135,0), (0.4865,1), (1.5,2), (1, 3)], 4, (1.5,2))
    []
    
    >>> compute_group_ofitems([(0.0135,0), (0.4865,1), (1.5,2), (1, 3)], 3, (1.5,2))
    []
    
    >>> compute_group_ofitems([(0.0135,0), (0.4865,1), (1.5,2), (1, 3)], 2, (1.5,2))
    []
    
    >>> compute_group_ofitems([(0.0135,0), (0.4865,1), (1.5,2), (1, 3)], 1, (1.5,2))
    [(0.4865, 1)]
    
    >>> compute_group_ofitems([(0.0135,0), (0.4865,1), (1.5,2), (1, 3)], 0, (1.5,2))
    [(1.5, 2), (1, 3)]
    '''
    lower_value = 1/(2**(r+1))
    higher_value = 1/(2**r)
    
    group_ofitems = []
    
    for item in demands:
        if item[0] > max_demand[0]*lower_value and item[0] <= max_demand[0]*higher_value:
            group_ofitems.append(item)

    return group_ofitems


# creating a C like structure 'group_solution'
@dataclass
class group_solution:
    r:int = field(default =None)
    group: List[Tuple] = field(default_factory = [])
    group_items_sum : float = field(default = 0)
    sums : List[np.ndarray] = field(default_factory= list)
    lists : List[List] = field(default_factory= list)
    numbins : int = field(default = 0)
    k : int = field(default = 0)
    eachbin_time_alloc:float = field(default = 0)
    eachagent_time_alloc:float = field(default = 0)
    final_eachbin_time_alloc:float = field(default = 0)
    final_eachagent_time_alloc:float = field(default = 0)
    final_group_time_alloc:float = field(default = 0)

def compute_large_demands_group_eachbin_time_alloc(sorted_groups_solution: List[group_solution]) ->float:
    '''
    Computes the time for group containing the large demands.
    The group_solution having least 'r' will be the group of large demands
    
    Algorithm:
        1 find the group having least r. Let this group is large_group
        2 Compute the 't' as follows
        sum = 0; large_group_k = large_group.k
        for grp_soln in groups_solution:
            sum += grp_soln.numbins * 2**(grp_soln.r) * (large_group_k/grp_soln.k) 
            
        3 t = 1/sum
        4 return t
    Parameters
    ----------
    groups_solution : List[group_solution]
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    '''
    large_group_time = 0
    large_group_k = sorted_groups_solution[0].k
    
    sum = 0
    for grp_soln in sorted_groups_solution:
        sum += grp_soln.numbins * 2**(grp_soln.r) * (large_group_k/grp_soln.k)
    
    large_group_time = 1/sum
    return large_group_time

def compute_possible_groups_tocombine(groups: List[group_solution], updated_supply:float, max_item:tuple, epsilon:float) -> Tuple:
    '''
    This function computes the possible groups that can be combined.
    It also computes the r value for such groups.

    Parameters
    ----------
    groups : List[group_solution]
        DESCRIPTION.
    updated_supply : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    Test cases:-
    
    >>> R = 6
    >>> groups_solution = []
    >>> for i in range(R+1):
    ...     grp = compute_group_ofitems([(0.2,0), (0.4,1), (0.8,2), (1.7,3), (3,4), (6.5, 5), (14,6)], i, (14,6)) 
    ...     if len(grp) != 0:
    ...         grp_items_sum = sum([demand[0] for demand in grp])
    ...         grp_soln = group_solution(r=i, group=grp, group_items_sum=grp_items_sum)
    ...         groups_solution.append(grp_soln)

    >>> groups_solution = sorted(groups_solution, key=lambda grp: grp.r, reverse=True)
    >>> groups, r_list = compute_possible_groups_tocombine(groups=groups_solution , updated_supply=21.0, max_item=(14,6), epsilon=1/2)
    >>> print(r_list)
    [6, 5, 4, 3, 2]
    
    >>> for grp in groups:
    ...     print(grp.group)
    [(0.2, 0)]
    [(0.4, 1)]
    [(0.8, 2)]
    [(1.7, 3)]
    [(3, 4)]
    '''
    small_item_threshold = epsilon*max_item[0]    #threshold to check of an item is small.
    
    valid_sum = 0   # denotes the total sum of all the items in all the groups
    possible_groups_tocombine_r_list = []  # containd the r value of the groups that can be combined.
    
    possible_groups_tocombine = []
    # sum_possible_groups_tocombine = 0
    
    for grp in groups:
        # check if all the item sizes in the group are at most the small_item_threshold
        if (max(grp.group)[0]) <= small_item_threshold:
            if valid_sum + grp.group_items_sum <= updated_supply - max_item[0]:
                valid_sum += grp.group_items_sum
                possible_groups_tocombine_r_list.append(grp.r)
                # add this group to the possible list of groups that can be combined
                possible_groups_tocombine.append(grp)
    
    
    # return the tuple possible_groups_tocombine, possible_groups_tocombine_r_list
    return possible_groups_tocombine, possible_groups_tocombine_r_list

def aggregate_bins(groups_solution: List[group_solution]) :
    '''
    Aggregate all the sums, lists in each group_solution in groups_solution.

    Parameters
    ----------
    groups_solution : List[group_solution]
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    final_sums = np.array([])
    final_lists = []    
    for grp_soln in groups_solution:
        final_sums = np.concatenate((final_sums, grp_soln.sums), axis=0)
        final_lists += grp_soln.lists
        
    return final_sums, final_lists

def compute_agents_time_alloc_list(groups_solution: List[group_solution], tocombine_groups_list:List[group_solution], vs_items_list: List[Tuple]) -> List:
    '''
    It computes the agents allocation of time from the groups_solution, tocombine_groups_list and vs_items_list.

    Returns
    -------
    None.

    '''
    agents_time_alloc_list = []
    
    flat_list = []
    unique_list = []
    for grp_soln in groups_solution:
        # finding all the items in the grp_soln bins.
        flat_list += [item for abin in grp_soln.lists for item in abin]
    
    # logging for testing
    # logger.debug(f"Flat list: {flat_list}")
    # logger.info("\n")
    
    # finding unique items
    unique_list = list(set(flat_list))
    
    # find the items from the tocombine_groups_list
    allitemsin_tocombine_groups_list = []
    for grp in tocombine_groups_list:
        for item in grp.group:
            if item not in allitemsin_tocombine_groups_list:
                allitemsin_tocombine_groups_list.append(item)
    
    # logging for testing
    # logger.debug(f"Unique list: {unique_list}")
    # logger.info("\n")
    
    for item in unique_list:
        # logging for testing
        # logger.debug(f"Final group connection: {grp_soln.final_alloc_time} \
                     # Each agent connection: {grp_soln.eachagent_alloc_fraction}")
        for grp_soln in groups_solution:
            if item in grp_soln.group and item not in allitemsin_tocombine_groups_list and item not in vs_items_list :
                agents_time_alloc_list.append((grp_soln.final_eachagent_time_alloc, item[1]))
        
        if item in allitemsin_tocombine_groups_list:
            agents_time_alloc_list.append((1, item[1]))
        
        if item in vs_items_list:
            agents_time_alloc_list.append((1, item[1]))
    
    return agents_time_alloc_list

def compute_egal_alg2_v2(alg:str, demands: List[float], supply: int, k:int, demand_tomaxdemand_ratio:float, e:float) -> Tuple:
    '''
    Implements the HA2 algorithm.
      
    Returns
    -------
    Tuple(Solution, List[Solution]
        DESCRIPTION.

    # >>> compute_egal_alg2_v2("FFk", [0.5, 1.5, 1], 3, k=20, demand_tomaxdemand_ratio=0.01, e=1/2)
    
    # >>> compute_egal_alg2_v2("FFk", [0.0135, 0.4865, 1.5, 1], 3, k=20, demand_tomaxdemand_ratio=0.01, e=1/2)
    
    # >>> compute_egal_alg2_v2("FFk", [0.5, 1.5, 1], 2, k=20, demand_tomaxdemand_ratio=0.01, e=1/2)
    
    
    # >>> compute_egal_alg2_v2("FFk", [0.0135, 0.4865, 1.5, 1], 2, k=20, demand_tomaxdemand_ratio=0.01, e=1/2)
    
    # >>> compute_egal_alg2_v2("FFk", [10, 9, 1], 11, k=1, demand_tomaxdemand_ratio=0.01, e=1/2)
    
    # >>> compute_egal_alg2_v2("FFk", [10, 9, 1], 11, k=1, demand_tomaxdemand_ratio=0.01, e=1)
    
    # >>> compute_egal_alg2_v2("FFk", [0.2, 0.4, 0.8, 1.7, 3, 6.5, 14], 21, k=20, demand_tomaxdemand_ratio=0.01, e=1/2)
    
    # >>> compute_egal_alg2_v2("FFk", [0.2,0.22, 0.4,0.42, 0.8,0.82, 1.7,1.7, 3,3.2, 6.5,6.7, 14,14.2], 21, k=3, demand_tomaxdemand_ratio=0.01, e=1/2)
    
    # >>> compute_egal_alg2_v2("FFk", [2, 4, 8, 17, 30, 30, 30, 30, 30, 30, 30, 30, 65, 65, 65, 65, 140], 210, k=20, demand_tomaxdemand_ratio=0.01, e=1/2)
    
    
    # >>> compute_egal_alg2_v2("FFk", [2.700083334, 2.540450001, 2.835183334, 0.599683334, 4.759250001, 1.026016668, 1.488833334, 2.600466668, 1.281950001, 0.386350001, 4.414050001, 0.801666668, 5.497566668, 2.798550001, 4.896533334, 5.330000001, 2.366533334, 6.063500001, 2.591333334, 3.726916668, 0.421733334, 4.301166668, 0.701066668, 2.625000001, 9.653816668, 1.633150001, 0.974733334, 2.966650001, 3.914716668, 4.019600001, 2.396966668, 1.905533334, 3.394583334, 2.845650001, 2.215733334, 5.022533334, 3.233833334, 2.472416668, 6.173300001, 3.058216668, 1.923483334, 1.768633334, 1.024283334, 0.611016668, 1.088533334, 1.091150001, 0.220150001, 2.017833334, 3.721483334, 0.258500001, 6.989550001, 2.143800001, 4.873550001, 2.968433334, 5.914333334, 1.666600001, 0.788833334, 6.582400001, 5.950000001, 1.257683334, 3.868466668, 5.023050001, 0.492733334, 0.584550001, 4.594050001, 3.682466668, 2.024916668, 0.638850001, 3.511950001, 3.666133334, 0.734533334, 0.154716668, 4.962216668, 1.431266668, 5.612550001, 3.322016668, 3.558416668, 2.931700001, 3.931666668, 1.035350001, 4.948133334, 2.737066668, 0.219333334, 4.456433334, 3.374966668, 0.541433334, 3.520800001, 3.399616668, 1.601066668, 1.778683334, 4.758666668, 2.744516668, 2.666716668, 3.454666668, 2.345183334, 0.507066668, 3.263533334, 2.752116668, 3.460716668, 3.773266668, 1.351950001, 4.432583334, 0.331816668, 2.011066668, 1.219316668, 1.138300001, 3.135950001, 3.216700001, 2.076983334, 1.724300001, 2.663700001, 2.922266668, 0.270050001, 9.054333334, 1.557033334, 5.678250001, 7.088883334, 1.458450001, 1.272633334, 2.057350001, 4.935533334, 1.072416668, 1.419366668, 8.141183334, 7.736866668, 2.401333334, 2.604866668, 2.684116668, 5.808483334, 2.994100001, 2.091250001, 2.556733334, 2.667600001, 2.443983334, 0.767616668, 5.081933334, 0.177483334, 4.400100001, 1.396216668, 4.765300001, 1.033133334, 1.218600001, 3.267900001, 0.748750001, 9.268383334, 5.649966668, 2.460616668, 0.615516668, 3.995733334, 1.907650001, 2.127016668, 0.774000001, 2.900166668, 0.404283334, 3.037816668, 1.175416668, 3.506466668, 4.286900001, 6.350866668, 0.195783334, 2.113950001, 1.905716668, 4.004766668, 2.124450001, 1.863483334, 0.175933334, 0.147650001, 4.258066668, 5.693633334, 1.382033332, 1.017883334, 1.089666668, 1.438516668, 0.651100001, 4.897416668, 2.609866668, 2.676466668, 4.596816668, 5.598816668, 5.596516668, 5.447650001, 4.999016668, 2.102650001, 4.104150001, 1.415866668, 0.918283334, 1.404550001, 4.084316668, 3.587783334, 6.275616668, 1.969466668, 5.760633334, 1.117166668, 1.610183334, 1.625866668, 1.741866668, 1.839116668, 1.158800001, 2.331433334, 1.417966668, 4.963250001, 0.569566668, 2.595233334, 2.337333334, 0.206216668, 5.051983334, 3.811800001, 3.539366668, 7.677366668, 3.387466668, 3.468900001, 3.917000001, 2.684683334, 2.613566668, 2.575000001, 1.745533334, 3.602150001, 0.330416668, 2.397616668, 3.583933334, 4.268433334, 4.255700001, 3.310383334, 7.184166668, 1.258500001, 0.582600001, 1.327133334, 4.632733334, 0.382666668, 4.833950001, 2.162116668, 5.959133334, 2.923566668, 3.260333334, 2.384300001, 3.294650001, 0.433216668, 2.178500001, 1.311050001, 3.893750001, 4.463366668, 0.523933334, 13.27948333, 0.136933334, 6.627116668, 0.387416668, 2.928850001, 3.550483334, 2.022250001, 6.503000001, 3.444166668, 3.752083334, 1.009683334, 3.237033334, 9.167716668, 3.846350001, 0.763133334, 2.858283334, 2.553733334, 4.055483334, 2.858883334, 4.162933334, 5.339766668, 3.233450001, 4.041200001, 3.248733334, 1.291400001, 1.857716668, 2.272500001, 1.684300001, 3.293533334, 3.300650001, 3.743083334, 1.371500001, 2.900583334, 3.144316668, 3.256966668, 3.145000001, 0.270100001, 5.199900001, 2.459200001, 2.117700001, 3.183166668, 3.298516668, 0.731283334, 3.950966668, 4.909916668, 3.034083334, 3.120800001, 6.647466668, 2.305516668, 8.798700001, 1.534800001, 1.882766668, 4.574450001, 2.531300001, 1.891200001, 4.558666668, 2.969766668, 1.841683334, 0.167566668, 0.737516668, 1.712850001, 1.053283334, 3.554833334, 2.290883334, 0.240633334, 2.364366668, 0.106516668, 2.306116668, 0.962483334, 1.607216668, 1.906916668, 3.057200001, 4.076983334, 0.124516668, 1.220200001, 2.197083334, 2.336966668, 2.329133334, 2.622883334, 0.250916668, 3.876083334, 5.640450001, 4.912633334, 2.310483334, 3.117700001, 2.524983334, 2.031700001, 2.688433334, 6.495916668, 1.223866668, 2.239433334, 2.424933334, 3.512416668, 1.479533334, 2.715183334, 3.839183334, 5.760183332, 4.131216668, 3.049183334, 3.143200001, 2.363666668, 8.577766668, 2.540766668, 1.348616668, 2.468366668, 7.243483334, 0.855066668, 14.9387, 3.336466668, 0.118050001, 0.595533334, 2.271900001, 2.039683334, 2.042950001, 2.459533334, 0.129083334, 0.275166668, 11.50521667, 1.665083334, 3.351716668, 2.763516668, 1.862533334, 3.913300001, 4.569900001, 6.006883334], 754, demand_tomaxdemand_ratio=0.01)
    
    
    '''
    if __name__ == "__main__":
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename = "heuristic_algorithm2_v2_github.log", encoding = "utf-8", level=logging.DEBUG, force=True)
    else:
        logger = logging.getLogger("heuristic_algorithm8.compute_egal_alg8_v2")
    
    
    if all(isinstance(item, tuple) for item in demands):
        if not all(len(item)==2 for item in demands):
            raise Exception("Some tuples are not in the format (item_value, index)")
        else:
            new_demands = demands
            # print("Input is in the required format.")
    else:
        if any(isinstance(item, tuple) for item in demands):
            raise Exception("Some items are not in the format (item_value, index)")
        else:
            new_demands = ffk.itemslist_to_tuplelist(demands)
    
    alg = "online" if alg == "FFk" else "decreasing"        
    # uncomment while testing this function
    logger.info(f"Algorithm: {alg}, Demands: {new_demands}, Supply: {supply}, k: {k}")
    
    # groups_solution: keeps the solution for each group
    groups_solution = []
    
    # if the sum of all the demands is at most the supply then pack them in a single bin.
    if sum([item[0] for item in new_demands]) <= supply:
        binner = ffk.bkc_ffk()
        if alg == "online":
            bins = ffk.online_ffk(binner, supply, new_demands, k=1)
            numbins_ingrp = binner.numbins(bins)
            groups_solution.append(group_solution(r=0, group=new_demands, sums=bins[1], lists=bins[1], numbins=numbins_ingrp, k=1, eachbin_time_alloc=1/numbins_ingrp, eachagent_time_alloc=1, final_eachbin_time_alloc=1, final_eachagent_time_alloc=1, final_group_time_alloc=1))
        else:
            bins = ffk.decreasing_ffk(binner, supply, new_demands, k=1)
            numbins_ingrp = binner.numbins(bins)
            groups_solution.append(group_solution(r=0, group=new_demands, sums=bins[1], lists=bins[1], numbins=numbins_ingrp, k=1, eachbin_time_alloc=1/numbins_ingrp, eachagent_time_alloc=1, final_eachbin_time_alloc=1, final_eachagent_time_alloc=1, final_group_time_alloc=1))
            groups_solution.append(group_solution(0, new_demands, bins[0], bins[1], numbins_ingrp , 1, 1/numbins_ingrp, eachagent_time_alloc=1, final_eachbin_time_alloc=1, final_eachagent_time_alloc=1, final_group_time_alloc=1))
        
        agents_time_alloc_list = []
        for item in new_demands:
            agents_time_alloc_list.append((1, item[1]))
            
        final_soln = Solution(groups_solution[0].sums, groups_solution[0].lists, new_demands, agents_time_alloc_list = agents_time_alloc_list, supply=supply)
        final_soln.compute_agents_connection_vector_fromagentstime()
        final_soln.compute_agents_egal_vector()
        final_soln.solution_egalconn_val(0)
        final_soln.solution_egalsupply_val(0)
        final_soln.compute_asr_vector()
        
        # logging -------- START
        # uncomment if you are testing this function
        # logger.info("****FINAL SOLUTION****")
        # logger.debug(f"Group: {final_soln.agents_list}")
        # logger.debug(f"Sums: {final_soln.sums},\nLists: {final_soln.lists}")
        # logger.debug(f"Egal Connection List: {    final_soln.agents_conn_vector}")
        # logger.debug(f"Egal Supply List: {    final_soln.agents_egal_vector}")
        # logger.debug(f"ASR List: {final_soln.asr_vector}")
        # logger.debug(f"Min ASR: {final_soln.min_asr}")
        # logger.debug(f"Total Watts: {sum([item[0] for item in final_soln.agents_egal_vector])}")
        # logger.debug(f"Total Connections(time): {sum([item[0] for item in final_soln.agents_conn_vector])}")
        # logger.info("\n\n\n\n")   
        # logging --------- END
        
        # uncomment if you are testing this function
        return final_soln
    
    # Determine the largest demand new_demands
    largest_demand = max(new_demands)
    
    # Finding the vs_items_list and vs_items_sum
    vs_items_list, vs_items_sum = vs_items(new_demands, largest_demand, epsilon=demand_tomaxdemand_ratio)
    
    # compute the remaining supply and remaining demand
    remaining_supply = supply - vs_items_sum
    remaining_new_demands = copy.deepcopy(new_demands)
    for item in vs_items_list:
        remaining_new_demands.remove(item)
    
    # for debugging
    # logger.debug(f"Very Small Items: {vs_items_list}, Very Small Items Sum: {vs_items_sum}, Updated Supply: {remaining_supply}")
    # logger.info("\n")
    
    # determine the smallest demand in remaining_new_demands.
    smallest_demand = min(remaining_new_demands)
    R = math.floor(math.log(largest_demand[0]/smallest_demand[0], 2))
    
    # for debugging
    # logger.debug(f"Largest demand: {largest_demand}, Smallest demand in remaining demands: {smallest_demand}, R : {R}")
    # logger.info("\n")
    
    # here I am also adding group to an instance of the group_solution
    for i in range(R+1):
       grp = compute_group_ofitems(remaining_new_demands, i, largest_demand) 
       if len(grp) != 0:
           grp_items_sum = sum([demand[0] for demand in grp])
           groups_solution.append(group_solution(r=i, group=grp, group_items_sum=grp_items_sum))
    
    # for logging purpose
    # for grp in groups_solution:
        # logger.debug(f"r={grp.r}, group={grp.group}, group items sum={grp.group_items_sum}") 
    # logger.info("\n")

    best_solution = None
    
    # check if there is only one group in groups_solution
    if len(groups_solution) ==1:
        
        binner  = ffk.bkc_ffk()
        if alg == "online":
            bins = ffk.online_ffk(binner, remaining_supply, groups_solution[0].group, k)
        else:
            bins = ffk.online_ffk(binner, remaining_supply, groups_solution[0].group, k)
        
        numbins_ingrp = binner.numbins(bins)
        groups_solution[0].sums = bins[0]
        groups_solution[0].lists = bins[1]
        groups_solution[0].numbins = numbins_ingrp
        groups_solution[0].k =k
        groups_solution[0].eachagent_time_alloc = k/numbins_ingrp
        groups_solution[0].eachbin_time_alloc = 1/numbins_ingrp
        
        #  adding the vs items
        vs_items_count = len(vs_items_list)
        for grp_soln in groups_solution :
            for ibin in range(grp_soln.numbins):
                for idx in range(vs_items_count):
                    binner.add_item_to_bin(bins=(grp_soln.sums, grp_soln.lists), item=vs_items_list[idx], bin_index=ibin)
        
            # adding the vs items to the groups_solution.group
            for item in vs_items_list:
                grp_soln.group.append(item)
        
        # determine the items which are connected all the time.
        g_items_list = []
        # add the vs_items_list items as they are connected all the time
        for item in vs_items_list:
            g_items_list.append(item)
        
        agents_time_alloc_vector = compute_agents_time_alloc_list(groups_solution, [], vs_items_list)
        
        best_solution = Solution(groups_solution[0].sums, groups_solution[0].lists, new_demands, agents_time_alloc_list = agents_time_alloc_vector, supply=supply, group_list=g_items_list)
        best_solution.compute_agents_connection_vector_fromagentstime()
        best_solution.compute_agents_egal_vector()
        best_solution.solution_egalconn_val(0)
        best_solution.solution_egalsupply_val(0)
        best_solution.compute_asr()
        
        return best_solution
        
    else:
        # logging for debugging
        # logger.debug(f"Small items are less than {e*largest_demand[0]}")
        # logger.info("\n")
        groups_solution = sorted(groups_solution, key=lambda grp: grp.r, reverse=True)
        possible_groups_tocombine, possible_groups_tocombine_r_list = compute_possible_groups_tocombine(groups=groups_solution, updated_supply=remaining_supply , max_item = largest_demand , epsilon = e)
        
        # for logging
        # logger.debug(f"Possible groups to combine={possible_groups_tocombine}")
        
        # sort groups_r_list in non-increasing order
        possible_groups_tocombine_r_list = sorted(possible_groups_tocombine_r_list , reverse=True)
        # logger.debug(f"Sorted r list for possible groups to combine={possible_groups_tocombine_r_list}")
        # logger.info("\n")
        
        # compute max_numof_groups_tojoin
        max_numof_groups_tojoin = len(possible_groups_tocombine_r_list)
        # for logging
        # logger.debug(f"Number of groups to combine={max_numof_groups_tojoin}")
        # logger.info("\n")

        # for each possible numof_groups_tojoin determine the solution for the remaining groups.
        # select the solution which gives better equal supply.
        for numof_groups_tojoin in range(max_numof_groups_tojoin+1):
            # compute the solution for the remaining groups and also compute the agents connection time
            # for logging
            # for grp in possible_groups_tocombine:
                # logger.debug(f"group to combine={grp.group}")
            
            tocombine_groups_list = possible_groups_tocombine[:numof_groups_tojoin]
            # for logging
            # groups_to_combine = []
            # for grp in tocombine_groups_list:
                # groups_to_combine.append(grp.group)
            # logger.debug(f"To combine groups list={groups_to_combine}")
            # logger.debug(f"To combine groups list={[grp.group for grp in tocombine_groups_list]}")
            # compute the sum of the tocombine_groups_list
            sum_tocombine_groups_list = 0 if len(tocombine_groups_list) == 0 else sum([grp.group_items_sum for grp in tocombine_groups_list])
            # for logging
            # logger.debug(f"Sum of to combine groups list={sum_tocombine_groups_list}")
            # compute the updated supply
            temp_rem_supply = remaining_supply - sum_tocombine_groups_list
            # for logging
            # logger.debug(f"Updated supply={temp_rem_supply}")
            # determine the remaining set of groups for which the solution has to be computed
            remaining_groups_solution = []
            for grp in groups_solution:
                if grp not in tocombine_groups_list:
                    remaining_groups_solution.append(copy.deepcopy(grp))
            
            # for logging
            # logger.debug(f"Remaining groups to kBP={[grp.group for grp in remaining_groups_solution]}")
            
            # compute the solution for remaining_groups_solution
            for agroup in remaining_groups_solution:
                # compute the ffk solution.
                binner = ffk.bkc_ffk()
                if alg == "online":
                   bins = ffk.online_ffk(binner, temp_rem_supply, agroup.group, k)
                   numbins_ingrp = binner.numbins(bins)
                   agroup.sums = bins[0]
                   agroup.lists = bins[1]
                   agroup.numbins = numbins_ingrp
                   agroup.k = k
                   agroup.eachagent_time_alloc = k/numbins_ingrp
                   agroup.eachbin_time_alloc = 1/numbins_ingrp
                else:
                   bins = ffk.decreasing_ffk(binner, temp_rem_supply, agroup.group, k)
                   numbins_ingrp = binner.numbins(bins)
                   agroup.sums = bins[0]
                   agroup.lists = bins[1]
                   agroup.numbins = numbins_ingrp
                   agroup.k = k
                   agroup.eachagent_time_alloc = k/numbins_ingrp
                   agroup.eachbin_time_alloc = 1/numbins_ingrp
            
            # Compute time for group of large demands
            # sort the groups_solution according to attribute r.
            remaining_groups_solution = sorted(remaining_groups_solution, key = lambda g: g.r)
            large_demands_group_eachbin_time = compute_large_demands_group_eachbin_time_alloc(remaining_groups_solution)
            
            remaining_groups_solution[0].final_eachbin_time_alloc = large_demands_group_eachbin_time
            remaining_groups_solution[0].final_eachagent_time_alloc = remaining_groups_solution[0].final_eachbin_time_alloc*remaining_groups_solution[0].k
            remaining_groups_solution[0].final_group_time_alloc = remaining_groups_solution[0].final_eachbin_time_alloc*remaining_groups_solution[0].numbins
         
            # for logging
            # logger.debug(f"group={remaining_groups_solution[0].group}")
            # logger.debug(f"group={remaining_groups_solution[0].group}, each agent time allocation={remaining_groups_solution[0].final_eachagent_time_alloc}")
            # logger.debug(f"group time allocation={remaining_groups_solution[0].final_group_time_alloc}")
         
            # compute the time for each bin, each agent for all other groups in the remaining_groups_solution
            for i in range(1,len(remaining_groups_solution)):
                remaining_groups_solution[i].final_eachagent_time_alloc = (2**remaining_groups_solution[i].r)*remaining_groups_solution[0].final_eachagent_time_alloc
                remaining_groups_solution[i].final_eachbin_time_alloc = remaining_groups_solution[i].final_eachagent_time_alloc/remaining_groups_solution[i].k 
                remaining_groups_solution[i].final_group_time_alloc = remaining_groups_solution[i].final_eachbin_time_alloc*remaining_groups_solution[i].numbins               
            
                # logging
                # logger.debug(f"group={remaining_groups_solution[i].group}, each agent time allocation={remaining_groups_solution[i].final_eachagent_time_alloc}")
            
            # add the agents in the tocombine_groups_list to all the bins in the solution to each group in remaining_groups_solution
            for grp_soln in remaining_groups_solution:
                for ibin in range(grp_soln.numbins):
                    for grp2 in tocombine_groups_list:
                        for demand in grp2.group:
                            binner.add_item_to_bin(bins=(grp_soln.sums, grp_soln.lists), item=demand, bin_index=ibin)
                    
                    # for logging
                    # logger.debug(f"Groups {[grp2.group for grp2 in tocombine_groups_list]} has been added to bin {grp_soln.lists[ibin]}")
                                
                
                # adding demand in tocombine_groups_list.group to the grp_soln.group
                for grp2 in tocombine_groups_list:
                    for demand in grp2.group:
                        grp_soln.group.append(demand)
            
            
            #  adding the vs items
            vs_items_count = len(vs_items_list)
            for grp_soln in remaining_groups_solution :
                for ibin in range(grp_soln.numbins):
                    for idx in range(vs_items_count):
                        binner.add_item_to_bin(bins=(grp_soln.sums, grp_soln.lists), item=vs_items_list[idx], bin_index=ibin)
            
                # adding the vs items to the groups_solution.group
                for item in vs_items_list:
                    grp_soln.group.append(item)
            
            # determine the demands which are connected all the time. We call this group_list. It is required in solution and will be used in 
            # determine max min function.
            g_items_list = []
            # first we add the items in the vs_items_list, as they are connected all the time.
            for item in vs_items_list:
                g_items_list.append(item)
            # now we add the items in the tocombine_groups_list, as these items are also connected all the time
            for grp in tocombine_groups_list:
                for item in grp.group:
                    g_items_list.append(item)
            
            # Now compute the solution using the Solution class
            # A: aggregate the bins in remaining_groups_solution
            final_sums, final_lists = aggregate_bins(remaining_groups_solution)
            # B: first we compute the agents allocaiton vector
            agents_time_alloc_vector = compute_agents_time_alloc_list(remaining_groups_solution, tocombine_groups_list, vs_items_list)
            # C: then,we create a solution object.
            cur_solution = Solution(final_sums, final_lists, new_demands, agents_time_alloc_list = agents_time_alloc_vector, supply=supply, group_list=g_items_list)
            cur_solution.compute_agents_connection_vector_fromagentstime()
            cur_solution.compute_agents_egal_vector()
            cur_solution.solution_egalconn_val(0)
            cur_solution.solution_egalsupply_val(0)
            cur_solution.compute_asr_vector()
            
            # logging ---------- START
            # uncomment while testing this function
            # logger.debug(f"Agents connection vector={cur_solution.agents_conn_vector}")
            # logger.debug(f"Agents egal vector={cur_solution.agents_egal_vector}")
            # logger.debug(f"Agents asr vector={cur_solution.asr_vector}")
            # logger.debug(f"Min asr={cur_solution.min_asr}")
            # logger.debug(f"Total Watts={sum([item[0] for item in cur_solution.agents_egal_vector])}")
            # logger.debug(f"Total Connections(time)={sum([item[0] for item in cur_solution.agents_conn_vector])}")
            # max_sup, min_sup = cur_solution.determine_max_and_min() 
            # logger.debug(f"max supply: {max_sup}, min supply: {min_sup}, max supply diff: {max_sup - min_sup}")
            # logger.info("\n\n")
            # logging -------- END
            
            # compare the cur_solution with the best_solution and update best_solution
            if best_solution ==None:
                best_solution = cur_solution
            elif cur_solution > best_solution:
            # elif cur_solution.min_asr > best_solution.min_asr:
                best_solution = cur_solution
    
    max_sup, min_sup = best_solution.determine_max_and_min()    
    
    
    # logging --------- START
    # uncomment while testing this function
    # logger.info("Best Solution")
    # logger.debug(f"Agents connection vector={best_solution.agents_conn_vector}")
    # logger.debug(f"Agents egal vector={best_solution.agents_egal_vector}")
    # logger.debug(f"Agents asr vector={best_solution.asr_vector}")
    # logger.debug(f"Min asr={best_solution.min_asr}")
    # logger.debug(f"Total Watts={sum([item[0] for item in best_solution.agents_egal_vector])}")
    # logger.debug(f"Total Connections(time)={sum([item[0] for item in best_solution.agents_conn_vector])}")
    # logger.debug(f"max supply: {max_sup}, min supply: {min_sup}, max supply diff: {max_sup - min_sup}")
    # logger.info("\n\n\n\n")        
    # logging ------- END
    
    # comment if testing this function
    return best_solution     
     
if __name__ == "__main__":
    import doctest
    
    (failures, tests) = doctest.testmod(report=True, verbose=False)
    print(f"Failures: {failures}, Tests: {tests}")