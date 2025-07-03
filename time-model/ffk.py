#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:37:57 2022

@author: dineshkumarbaghel
Since: 06-2022
"""

"""
    Implements the k-times bin packing. It packs the numbers using the variant of the first-fit bin-packing algorithms:
"""
from typing import Callable, List, Any, Tuple, Iterator
import itertools
from prtpy import outputtypes as out
from prtpy import BinsArray, Binner, BinnerKeepingContents, printbins
import numpy as np
from copy import deepcopy
from functools import reduce
import logging
 
class bkc_ffk(BinnerKeepingContents):
    def __init__(self, valueof: Callable = lambda x: x[0], indexof: Callable = lambda x: x[1]):
        super().__init__(valueof)
        
        
    def add_item_to_bin(self, bins: BinsArray, item: tuple, bin_index: int) -> BinsArray:
        '''

        Parameters
        ----------
        bins : BinsArray
            DESCRIPTION.
        item : Tuple(Any, int)
            DESCRIPTION.
        bin_index : int
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        BinsArray
            DESCRIPTION.
            
        Test Cases:
        -----------
        
        >>> values = {"a":3, "b":4, "c":5, "d":5, "e":5}
        >>> binner = bkc_ffk(lambda x: values[x])
        >>> bins = binner.new_bins(3)
        >>> #binner.add_item_to_bin(bins, item=(1,1.0), bin_index=0)
        #Item index is not integer.
        >>> #binner.add_item_to_bin(bins, item="a", bin_index=0)
        #item is not of the form: (Any, int)
        '''
        if len(item) != 2:
            raise Exception(f"item is not of the form: (Any, int).")
        elif type(item[1]) != int:
            raise ValueError(f"Item index is not integer.")
        return super().add_item_to_bin(bins, item, bin_index)
    
    def all_combinations(self, bins1: BinsArray, bins2: BinsArray)->Iterator[BinsArray]:
        """
        >>> binner = bkc_ffk()
        >>> b1 = ([1, 20, 300],  [[(1, 0)], [(20, 1)], [(300, 2)]])
        >>> b2 = ([4, 50, 600],  [[(1, 3), (3, 4)], [(4, 0), (46, 1)], [(600, 2)]])
        >>> for perm in binner.all_combinations(b1,b2): perm[1]
        [[(1, 0), (1, 3), (3, 4)], [(4, 0), (20, 1), (46, 1)], [(300, 2), (600, 2)]]
        [[(1, 0), (1, 3), (3, 4)], [(4, 0), (46, 1), (300, 2)], [(20, 1), (600, 2)]]
        [[(1, 3), (3, 4), (20, 1)], [(1, 0), (4, 0), (46, 1)], [(300, 2), (600, 2)]]
        [[(1, 3), (3, 4), (20, 1)], [(4, 0), (46, 1), (300, 2)], [(1, 0), (600, 2)]]
        [[(1, 0), (4, 0), (46, 1)], [(1, 3), (3, 4), (300, 2)], [(20, 1), (600, 2)]]
        [[(4, 0), (20, 1), (46, 1)], [(1, 3), (3, 4), (300, 2)], [(1, 0), (600, 2)]]
        """
        yielded = set() # to avoid duplicates
        sums1, lists1 = bins1
        sums2, lists2 = bins2
        numbins = len(sums1)
        if len(sums2)!=numbins:
            raise ValueError(f"Inputs should have the same number of bins, but they have {numbins} and {len(sums2)} bins.")
        for perm in itertools.permutations(range(numbins)):
            new_sums =  [sums1[perm[i]] + sums2[i] for i in range(numbins)]
            new_lists = [sorted(lists1[perm[i]] + lists2[i], key=lambda x:x[0]) for i in range(numbins)]  # sorting to avoid duplicates
            new_bins = (new_sums, new_lists)
            self.sort_by_ascending_sum(new_bins)
            new_lists_tuple = tuple(map(tuple,new_bins[1]))
            if new_lists_tuple not in yielded:
                yielded.add(new_lists_tuple)
                yield new_bins
    
    def check(self, item, bin: list) -> bool:
        '''
        It checks if item is in bin.

        Parameters
        ----------
        item : TYPE
            DESCRIPTION.
        bin : list
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.
        
        '''
        return item in bin
        
        
        
        
                
def itemslist_to_tuplelist(items: Any) -> List[Tuple]:
    '''
    It converts the list of items to list of tuples where each tuple is (item, index_of_item).

    Parameters
    ----------
    items : Any
        DESCRIPTION.

    Returns
    -------
    List[Tuple]
        DESCRIPTION.
        
    Test Cases:
    
    >>> itemslist_to_tuplelist([1,2,3,7,5])
    [(1, 0), (2, 1), (3, 2), (7, 3), (5, 4)]
    
    >>> itemslist_to_tuplelist([])
    []
    
    >>> itemslist_to_tuplelist([5])
    [(5, 0)]
    '''
    tl = []
    items_len = len(items)
    
    for i in range(items_len):
        t = (items[i], i)
        tl.append(t)
    
    return tl
                
def online_ffk(
    binner: Binner,
    binsize: float,
    items: List[any],
    k: int = 1
):
    """
    Pack the given items into bins using the online first-fit algorithm.
    The online algorithm handles the items in the order they are given.

    >>> from prtpy.binners import Binner, BinnerKeepingContents, BinnerKeepingSums
    >>> printbins(online_ffk(bkc_ffk(), binsize=9, items=[1,2,3,3,5,9,9], k=2))
    Bin #0: [(1, 0), (2, 1), (3, 2), (3, 3)], sum=9.0
    Bin #1: [(5, 4), (1, 0), (2, 1)], sum=8.0
    Bin #2: [(9, 5)], sum=9.0
    Bin #3: [(9, 6)], sum=9.0
    Bin #4: [(3, 2), (3, 3)], sum=6.0
    Bin #5: [(5, 4)], sum=5.0
    Bin #6: [(9, 5)], sum=9.0
    Bin #7: [(9, 6)], sum=9.0
    >>> printbins(online_ffk(bkc_ffk(), binsize=9, items=[1,2,3,5,9,3,9], k=2))
    Bin #0: [(1, 0), (2, 1), (3, 2), (3, 5)], sum=9.0
    Bin #1: [(5, 3), (1, 0), (2, 1)], sum=8.0
    Bin #2: [(9, 4)], sum=9.0
    Bin #3: [(9, 6)], sum=9.0
    Bin #4: [(3, 2), (5, 3)], sum=8.0
    Bin #5: [(9, 4)], sum=9.0
    Bin #6: [(3, 5)], sum=3.0
    Bin #7: [(9, 6)], sum=9.0
    >>> printbins(online_ffk(bkc_ffk(), binsize = 100, items = [26, 31, 17, 6, 12, 35, 22, 24, 13, 14], k=2))     
    Bin #0: [(26, 0), (31, 1), (17, 2), (6, 3), (12, 4)], sum=92.0
    Bin #1: [(35, 5), (22, 6), (24, 7), (13, 8), (6, 3)], sum=100.0
    Bin #2: [(14, 9), (26, 0), (31, 1), (17, 2), (12, 4)], sum=100.0
    Bin #3: [(35, 5), (22, 6), (24, 7), (13, 8)], sum=94.0
    Bin #4: [(14, 9)], sum=14.0
    
    """
    
    logger = logging.getLogger(__name__)
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception(f"Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    bins = binner.new_bins(1)
    numbins = 1
   
    for times in range(k):
        for item in new_items:
            value = binner.valueof(item)
            # uncomment the below one line when testing the function
            # logging.info(f"FFk: value: {value}")
            if value > binsize:
                raise ValueError(f"Item {item} has size {value} which is larger than the bin size {binsize}.")
            ibin = 0
            while ibin  < numbins:
                if binner.sums(bins)[ibin] + value <= binsize and not binner.check(item, bins[1][ibin]):
                    binner.add_item_to_bin(bins, item, ibin)
                    break
                ibin += 1
            else:  # if not added to any bin
                bins = binner.add_empty_bins(bins, 1)
                numbins += 1
                binner.add_item_to_bin(bins, item, ibin)
    return bins
    

    
def decreasing_ffk(
    binner: Binner,
    binsize: float,
    items: List[any],
    k: int  = 1
) -> BinsArray:
    
    '''
    Pack the given items into bins using the first-fit-decreasing algorithm.
    It sorts the items by descending value, and then runs first-fit.

    >>> from prtpy import BinnerKeepingContents, BinnerKeepingSums
    >>> decreasing_ffk(bkc_ffk(), binsize=9, items=[1,2,3,3,5,9,9])
    (array([9., 9., 9., 5.]), [[(9, 5)], [(9, 6)], [(5, 4), (3, 2), (1, 0)], [(3, 3), (2, 1)]])
    >>> decreasing_ffk(bkc_ffk(), binsize=18, items=[1,2,3,3,5,9,9])
    (array([18., 14.]), [[(9, 5), (9, 6)], [(5, 4), (3, 2), (3, 3), (2, 1), (1, 0)]])

    Non-monotonicity examples from Wikipedia:
    >>> example1 = [44, 24, 24, 22, 21, 17, 8, 8, 6, 6]
    >>> printbins(decreasing_ffk(bkc_ffk(), binsize=60, items=example1, k=2)) # 6 bins
    Bin #0: [(44, 0), (8, 6), (8, 7)], sum=60.0
    Bin #1: [(24, 1), (24, 2), (6, 8), (6, 9)], sum=60.0
    Bin #2: [(22, 3), (21, 4), (17, 5)], sum=60.0
    Bin #3: [(44, 0), (8, 6), (8, 7)], sum=60.0
    Bin #4: [(24, 1), (24, 2), (6, 8), (6, 9)], sum=60.0
    Bin #5: [(22, 3), (21, 4), (17, 5)], sum=60.0
    
    >>> printbins(decreasing_ffk(bkc_ffk(), binsize=61, items=example1, k=2)) # 7 bins
    Bin #0: [(44, 0), (17, 5)], sum=61.0
    Bin #1: [(24, 1), (24, 2), (8, 6)], sum=56.0
    Bin #2: [(22, 3), (21, 4), (8, 7), (6, 8)], sum=57.0
    Bin #3: [(6, 9), (44, 0), (8, 6)], sum=58.0
    Bin #4: [(24, 1), (24, 2), (8, 7)], sum=56.0
    Bin #5: [(22, 3), (21, 4), (17, 5)], sum=60.0
    Bin #6: [(6, 8), (6, 9)], sum=12.0
    
    >>> example2 = [51, 27.5, 27.5, 27.5, 27.5, 25, 12, 12, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    >>> printbins(decreasing_ffk(bkc_ffk(), binsize=75, items=example2, k = 2)) # 8 bins
    Bin #0: [(51, 0), (12, 6), (12, 7)], sum=75.0
    Bin #1: [(27.5, 1), (27.5, 2), (10, 8), (10, 9)], sum=75.0
    Bin #2: [(27.5, 3), (27.5, 4), (10, 10), (10, 11)], sum=75.0
    Bin #3: [(25, 5), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16)], sum=75.0
    Bin #4: [(51, 0), (12, 6), (12, 7)], sum=75.0
    Bin #5: [(27.5, 1), (27.5, 2), (10, 8), (10, 9)], sum=75.0
    Bin #6: [(27.5, 3), (27.5, 4), (10, 10), (10, 11)], sum=75.0
    Bin #7: [(25, 5), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16)], sum=75.0
    
    >>> printbins(decreasing_ffk(bkc_ffk(), binsize=76, items=example2, k=2)) # 9 bins
    Bin #0: [(51, 0), (25, 5)], sum=76.0
    Bin #1: [(27.5, 1), (27.5, 2), (12, 6)], sum=67.0
    Bin #2: [(27.5, 3), (27.5, 4), (12, 7)], sum=67.0
    Bin #3: [(10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14)], sum=70.0
    Bin #4: [(10, 15), (10, 16), (51, 0)], sum=71.0
    Bin #5: [(27.5, 1), (27.5, 2), (12, 6)], sum=67.0
    Bin #6: [(27.5, 3), (27.5, 4), (12, 7)], sum=67.0
    Bin #7: [(25, 5), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12)], sum=75.0
    Bin #8: [(10, 13), (10, 14), (10, 15), (10, 16)], sum=40.0
    
    >>> binner = BinnerKeepingContents()
    >>> binner.numbins(decreasing_ffk(bkc_ffk(), binsize=76, items=example2, k=2))
    9
    
    >>> binner.numbins(decreasing_ffk(bkc_ffk(), binsize=75, items=example2, k = 2))
    8
    '''
    # uncomment the below three lines when testing the function
    # logging.basicConfig(filename='ffk.log', level=logging.DEBUG)
    # logger = logging.getLogger(__name__)
    # logger.info(f"FFDk:")
    
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception(f"Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    sorted_new_items = sorted(new_items, key=binner.valueof, reverse=True)
    
    # uncomment the below line if testing.
    # logger.info(f"FFDk: Sorted Items: {sorted_new_items}")
    
    return online_ffk(
        binner,
        binsize, 
        sorted_new_items,
        k
    )



if __name__ == "__main__":
    import doctest

    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))
