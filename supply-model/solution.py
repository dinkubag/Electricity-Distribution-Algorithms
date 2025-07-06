#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:35:02 2025

@author: dineshkumarbaghel
"""
# import numpy as np
from typing import List, Tuple
# import ffk 
import logging
import math

class Solution:
    '''
    This class represents a soluion. A solution consists of the following things:
    
    sums:= this is list of the sum of the items in bins.
    lists:= List is a list of lsits of items.
    agent_connection_vector:= it denotes the connection time for each agent.
    agent_egal_vector:= it denotes the egalitarian value for each agent.
    egalsupply_value:= denotes the egalitarian value for the packing in lists.
    k:= this is the maximum copies an item can have. in a solution there can be differnt k for different items.
    For a given set of items there is a subset which appears in every bin. Now, for the remaining subset of items, an item can appear
    a different number of times in different bins. Each item appears at most in a bin.
    NOTE: it is possible to use 'k' in a way differnet than we used. If you do not want to use it at all then specify k as 1.
    
    '''
    
    def __init__(self, sums:List, lists: List[List], agents: List[Tuple], agentskvector: List[Tuple] = None, agents_time_alloc_list: List[Tuple]=None, supply:float =0, k: int=1, group_size: int = 0, vs:int = 0, g:int = 0, group_list: List[Tuple] = []):
        self.sums = sums
        self.lists = lists
        self.agents_conn_vector = []
        self.agents_egal_vector = []
        self.egalsupply_val = 0
        self.ith_egalsupply_val = 0
        self.agents_list = agents
        self.k = k
        self.egalconn_val = 0
        self.ith_egalconn_val = 0
        self.agents_k_vector = agentskvector
        self.gs = group_size
        self.g = g
        self.vs = vs
        self.num_bins = len(lists)
        self.agents_time_alloc_list = agents_time_alloc_list
        self.supply = supply
        self.min_asr = 0
        self.min_acr = 0
        self.asr_vector = []
        self.acr_vector = []
        self.grouped_demands_list = group_list
        
        
    def compute_agents_connection_vector_fromagentstime(self):
        '''
        

        Returns
        -------
        None.

        '''
        self.agents_conn_vector = self.agents_time_alloc_list
        self.agents_conn_vector[:] = sorted(self.agents_conn_vector, key=lambda conn: conn[1])
    
    def compute_agents_connection_vector(self):
        '''
        

        Parameters
        ----------
        bins : List[List]
            DESCRIPTION.

        Returns
        -------
        None.
        
        Test Cases:
        
        >>> from ffk import online_ffk_supply, bkc_ffk
        
        >>> soln = Solution([3., 4., 5., 3., 1., 1., 1.], [[(1, 0), (2, 1)], [(3, 2), (1, 0)], [(2, 1), (3, 2)], [(1, 0), (2, 1)], [(1, 0)], [(1, 0)], [(1, 0)]], [(1, 0), (2, 1), (3, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.agents_conn_vector
        [(0.85714, 0), (0.42857, 1), (0.28571, 2)]
        
        >>> soln = Solution([3., 3.], [[(1, 0), (2, 1)], [(2, 2), (1, 0)]], [(1, 0), (2, 1), (2, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.agents_conn_vector
        [(1.0, 0), (0.5, 1), (0.5, 2)]
        
        >>> soln = Solution([33., 44., 55., 33., 11., 11., 11.], [[(11, 0), (22, 1)], [(33, 2), (11, 0)], [(22, 1), (33, 2)], [(11, 0), (22, 1)], [(11, 0)], [(11, 0)], [(11, 0)]], [(11, 0), (22, 1), (33, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.agents_conn_vector
        [(0.85714, 0), (0.42857, 1), (0.28571, 2)]
        
        >>> soln = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0), (2, 1), (1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.agents_conn_vector
        [(0.5, 0), (0.5, 1), (0.5, 2)]
         
        >>> soln = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0), (1, 2)]], [(3, 0), (2, 1), (1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.agents_conn_vector
        [(0.5, 0), (0.5, 1), (1.0, 2)]
        
        >>> soln = Solution([3.], [[(2, 0), (1, 1)]], [(2, 0), (1, 1)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.agents_conn_vector
        [(1.0, 0), (1.0, 1)]
        

        '''
        # logger = logging.getLogger("compute agents connection vector")
        numof_bins = len(self.lists)
        
        
        flat_binslist = [item for bin in self.lists for item in bin]
        
        unique_flat_binslist = list(set(flat_binslist))
        
        sorted_unique_flat_binslist = sorted(unique_flat_binslist, key = lambda x: x[1])
        
        
        for item in sorted_unique_flat_binslist:
            
            count_item = flat_binslist.count(item)
            connection_hr = round(count_item/numof_bins,5)
            
            self.agents_conn_vector.append((connection_hr, item[1]))
     
        for item in self.agents_list:
            if item not in sorted_unique_flat_binslist:
                self.agents_conn_vector.append((0, item[1]))
        
        self.agents_conn_vector[:] = sorted(self.agents_conn_vector, key=lambda conn: conn[1])
        # logger.debug(f"Agents connection vector: {self.agents_conn_vector}\n")
        
    
    def compute_agents_egal_vector(self):
        '''
        

        Parameters
        ----------
        agents : List[Tuple]
            DESCRIPTION.

        Returns
        -------
        None.

        Test Cases-
        
        >>> soln = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.compute_agents_egal_vector()
        >>> print(soln.agents_egal_vector)
        [(1.5, 0), (1.0, 1), (0.5, 2)]
        
        
        >>> soln = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0), (1, 2)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.compute_agents_egal_vector()
        >>> print(soln.agents_egal_vector)
        [(1.5, 0), (1.0, 1), (1.0, 2)]
        
        
        
        '''
        for agent in self.agents_list:
            for agent_connection in self.agents_conn_vector:
                if agent[1]== agent_connection[1]:
                    self.agents_egal_vector.append((round(agent[0]*agent_connection[0],5), agent[1]))
        
        self.agents_egal_vector[:] = sorted(self.agents_egal_vector, key = lambda egl: egl[1])
        
    def solution_egalsupply_val(self, ith_idx: int):
        '''
        

        Returns
        -------
        None.

        Test Cases-
        
        >>> soln = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.compute_agents_egal_vector()
        >>> soln.solution_egalsupply_val(0)
        >>> print(soln.egalsupply_val)
        (0.5, 2)
        
        >>> soln = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0), (1, 2)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.compute_agents_egal_vector()
        >>> soln.solution_egalsupply_val(0)
        >>> print(soln.egalsupply_val)
        (1.0, 1)
        
        >>> soln = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.compute_agents_egal_vector()
        >>> soln.solution_egalsupply_val(1)
        >>> print(soln.ith_egalsupply_val)
        (1.0, 1)
        
        >>> soln = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0), (1, 2)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.compute_agents_egal_vector()
        >>> soln.solution_egalsupply_val(2)
        >>> print(soln.ith_egalsupply_val)
        (1.5, 0)


        '''
        # logger = logging.getLogger("Solution.solution_egalsupply_val")
        
        sorted_agents_egal_vector = sorted(self.agents_egal_vector, key = lambda x: x[0])
        
        # logger.debug(f"ith index: {ith_egalsupply_val}, egal vector len: {len(sorted_agents_egal_vector)}")

        self.ith_egalsupply_val = sorted_agents_egal_vector[ith_idx]
        
        self.egalsupply_val = min(self.agents_egal_vector)
        
        # uncomment if you are testing this function while executing this file
        # return self.egalsupply_val
        
    def solution_egalconn_val(self, ith_idx: int):
        '''
        

        Returns
        -------
        None.

        Test Cases-
        
        >>> soln = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.solution_egalconn_val(0)
        >>> print(soln.egalconn_val)
        (0.5, 0)
        
        >>> soln = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0), (1, 2)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.solution_egalconn_val(0)
        >>> print(soln.egalconn_val)
        (0.5, 0)
        
        >>> soln = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.solution_egalconn_val(1)
        >>> print(soln.ith_egalconn_val)
        (0.5, 1)
        
        >>> soln = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0), (1, 2)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln.compute_agents_connection_vector()
        >>> soln.solution_egalconn_val(2)
        >>> print(soln.ith_egalconn_val)
        (1.0, 2)


        '''
        sorted_agents_conn_vector = sorted(self.agents_conn_vector, key = lambda x: x[0])

        self.ith_egalconn_val = sorted_agents_conn_vector[ith_idx]
        
        self.egalconn_val = min(self.agents_conn_vector)
    
    
    def __gt__(self, soln2):
        '''
        Compares two solutions and returns True if soln1 is greater than soln2; false otherwise.

        Parameters
        ----------
        soln2 : TYPE
            DESCRIPTION.

        Returns
        -------
        Bool
            DESCRIPTION.

        >>> soln1 = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0), (1, 2)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln2 = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln1 > soln2
        True
        
        >>> soln1 = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0), (1, 2)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln2 = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln2 > soln1
        False
        
        >>> soln1 = Solution([3., 4.], [[(2, 1), (1, 2)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln2 = Solution([3., 3.], [[(1, 2), (2, 1)], [(3, 0)]], [(3, 0),(2, 1),(1, 2)], k=1)
        >>> soln1 > soln2
        False
        
        '''
        logger = logging.getLogger("Solution.__gt__")
        numof_items = len(self.agents_list)
        
        
        ith_idx = 0
        if len(self.agents_conn_vector) == 0:
            self.compute_agents_connection_vector()
            self.compute_agents_egal_vector()
            self.solution_egalsupply_val(ith_idx)
            self.solution_egalconn_val(ith_idx)
            self.compute_asr_vector()
        
        if len(soln2.agents_conn_vector) == 0:        
            soln2.compute_agents_connection_vector()
            soln2.compute_agents_egal_vector()
            soln2.solution_egalsupply_val(ith_idx)
            soln2.solution_egalconn_val(ith_idx)
            soln2.compute_asr_vector()
        
        # for testing purposes
        # self.soln1_iegalvector.append(self.egalsupply_val)
        # self.soln2_iegalvector.append(soln2.egalsupply_val)
        
        if self.egalsupply_val[0] > soln2.egalsupply_val[0]:
            # for logging
            # logger.debug(f"Current solution {sorted(self.agents_egal_vector)} \nis greater than \nbest solution {sorted(soln2.agents_egal_vector)}") 
            return True
            
            
        elif self.egalsupply_val[0] == soln2.egalsupply_val[0]:
            # since the egal value of two solutons are same. We will compare the second best egal value, and if it is same then we
            # will comapre the third best egal value, and so on.
            
            #  now we will compare each egalitarian value in the cur_soln against each egalitarian value in the best_solution_sofar.
            for idx in range(1, numof_items):
                ith_idx = idx    # we will compare the ith_egalsupply_val in the current and the best solution.
                
                # compute the ith_egalsupply_val in the current and the best solution so far.
                # logger.debug(f"soln1: {self.agents_egal_vector}\nsoln2: {soln2.agents_egal_vector}\n")
                self.solution_egalsupply_val(ith_idx)
                soln2.solution_egalsupply_val(ith_idx)
                
                
                # compare the above computed values.
                if self.ith_egalsupply_val[0] > soln2.ith_egalsupply_val[0]:
                    # current solution sofar is better than the best solution.
                    # There is no need to do anything except come out of the loop.
                    
                    # for testing purpose
                    # self.soln1_iegalvector.append(self.ith_egalsupply_val)
                    # self.soln2_iegalvector.append(soln2.ith_egalsupply_val)
                    
                    # below line only for testing.
                    # logger.debug(f"ith val: {self.ith_egalsupply_val}")
                    
                    # for logging
                    # logger.debug(f"Current solution {sorted(self.agents_egal_vector)} \nis greater than \nbest solution {sorted(soln2.agents_egal_vector)}") 
                    
                    return True
                elif self.ith_egalsupply_val[0] < soln2.ith_egalsupply_val[0]:
                    # current solution is not better than the best solution so far.
                    # logger.debug(f"Current solution {sorted(self.agents_egal_vector)} \nis not greater than \nbest solution {sorted(soln2.agents_egal_vector)}") 
                    return False
        
                # for testing purpose
                # self.soln1_iegalvector.append(self.ith_egalsupply_val)
                # self.soln2_iegalvector.append(soln2.ith_egalsupply_val)
                
        
        return False
    
    def determine_max_and_min(self) -> Tuple:
        '''
        Determines the maximum and minimum value in the agents_egal_vector for the agents having conn_val less than 1
        If all the conn values are 1 then return simply max and min.

        Returns
        -------
        Tuple
            DESCRIPTION.

        '''
        # logger = logging.getLogger("determine_max_and_min")
        if all([True if conn[0] == 1 else False for conn in self.agents_conn_vector]):
            # return max(self.agents_egal_vector)[0], min(self.agents_egal_vector)[0]
            return 0,0
        else:
            min_sup = math.inf
            max_sup = 0
            
            # new version
            grouped_demands_index_list = [demand[1] for demand in self.grouped_demands_list]
            
            for egal_val in self.agents_egal_vector:
                if egal_val[1] not in grouped_demands_index_list:
                    if egal_val[0] < min_sup :
                        min_sup = egal_val[0]
                    if egal_val[0] > max_sup:
                        max_sup = egal_val[0]
            
            
            return max_sup, min_sup
                        
        
    
    def compute_asr_vector(self) -> float:
        '''
        Computes the asr for an agent as = allocated_demand/demand

        Returns
        -------
        float
            DESCRIPTION.

        >>> soln = Solution(sums = [3., 5., 4.], lists = [[(1,0), (2,1)], [(2,1), (3,2)], [(3,2), (1,0)]], agents = [(1,0), (2,1), (3,2)])
        >>> soln.agents_egal_vector = [(1,1), (1,0), (1,2)]
        >>> soln.compute_asr_vector()
        >>> soln.min_asr
        (0.3333333333333333, 2)

        '''
        
        
        for item in self.agents_egal_vector:
            for demand in self.agents_list:
                if demand[1] == item[1]:
                    self.asr_vector.append((item[0]/demand[0], demand[1]))
                    break
                
        self.min_asr = min(self.asr_vector)


if __name__ == "__main__":
    import doctest
    
    (failures, tests) = doctest.testmod(report=True, verbose=False)
    
    print(f"Failures: {failures}, Tests: {tests}")