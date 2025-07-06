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
import newinstance
 
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
            raise Exception("item is not of the form: (Any, int).")
        elif type(item[1]) != int:
            raise ValueError("Item index is not integer.")
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
            
        Test Cases:
        
        >>> binner = bkc_ffk()
        >>> binner.check(1, [1, 2])
        True
        
        >>> binner.check(1, [2, 3])
        False
        
        >>> binner.check((1, 0), [(1, 0), (1, 1)])
        True
        
        >>> binner.check((1, 0), [(2, 0), (1, 1)])
        False
        
        '''
        return item in bin
    
    def remove_item_from_bin(self, bins:BinsArray, item: Any, bin_index: int) ->BinsArray:
        '''
        Removes an item from the bin.

        Parameters
        ----------
        bins : BinsArray
            DESCRIPTION.
        item : Any
            DESCRIPTION.
        bin_index : int
            DESCRIPTION.

        Returns
        -------
        BinsArray
            DESCRIPTION.
        
        # >>> values = {"a":3, "b":4, "c":5, "d":5, "e":5}
        >>> binner = bkc_ffk()
        >>> bins = binner.new_bins(2)
        >>> printbins(binner.add_item_to_bin(bins, item = (1,0), bin_index = 0))
        Bin #0: [(1, 0)], sum=1.0
        Bin #1: [], sum=0.0
        
        >>> printbins(binner.add_item_to_bin(bins, item = (2,1), bin_index = 1))
        Bin #0: [(1, 0)], sum=1.0
        Bin #1: [(2, 1)], sum=2.0
        
        >>> printbins(binner.remove_item_from_bin(bins, item = (2,1), bin_index = 1))
        Bin #0: [(1, 0)], sum=1.0
        Bin #1: [], sum=0.0
        
        >>> printbins(binner.remove_item_from_bin(bins, item = (1,0), bin_index = 0))
        Bin #0: [], sum=0.0
        Bin #1: [], sum=0.0
        
        # >>> printbins(binner.remove_item_from_bin(bins, item = (1,0), bin_index = 0))
        '''
        sums, lists = bins
        
        if item not in lists[bin_index]:
            raise Exception(f"Item {item} not found in the bin.")
        
        if item in lists[bin_index]:
            lists[bin_index].remove(item)
            sums[bin_index] -= self.valueof(item)
        
        return bins
    
    def replace_item_in_bin(self, bins:BinsArray, item_to_replace: Any, items_to_add:List[Any], bin_index:int) -> BinsArray:
        '''
        Replaces the item_to_replace with items_to_add in the bin with bin index "bin_index".

        Parameters
        ----------
        bins : BinsArray
            DESCRIPTION.
        item : Any
            DESCRIPTION.
        bin_index : int
            DESCRIPTION.

        Returns
        -------
        BinsArray
            DESCRIPTION.
        
        >>> binner = bkc_ffk()
        >>> bins = binner.new_bins(2)
        >>> printbins(binner.add_item_to_bin(bins, item = (1,0), bin_index = 0))
        Bin #0: [(1, 0)], sum=1.0
        Bin #1: [], sum=0.0
        
        >>> printbins(binner.replace_item_in_bin(bins, item_to_replace = (1,0), items_to_add = [(2,0)], bin_index=0))
        Bin #0: [(2, 0)], sum=2.0
        Bin #1: [], sum=0.0
        
        >>> printbins(binner.replace_item_in_bin(bins, item_to_replace = (2,0), items_to_add = [(1,0), (0.5,1)], bin_index=0))
        Bin #0: [(1, 0), (0.5, 1)], sum=1.5
        Bin #1: [], sum=0.0
        '''
        
        bins = self.remove_item_from_bin(bins, item_to_replace, bin_index)
        sums, lists = bins
        
        for item in items_to_add:
            if item in lists[bin_index]:
                raise Exception(f"Item {item} is already present in the bin.")
            else:    
                self.add_item_to_bin(bins, item, bin_index)
        
        return bins
        
        
                
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

def online_ffk_supply(
        binner: Binner,
        binsize: float,
        items: List[Any]
        ):
    '''
    Pack the given list of lists of items into bins using the online first-fit algorithm.
    The online algorithm handles the items in the order they are given.
    An item in the list is in the form of (value, index).
    A given list of items lists can be: [[(1, 0), (2, 1), (3, 2)], [(1, 0), (2, 1)], [(1, 0)]]
    It uses the LCM of the demands to compute the k vector.
    Parameters
    ----------
    binner : Binner
        DESCRIPTION.
    binsize_ float : TYPE
        DESCRIPTION.
    items : List[Any]
        DESCRIPTION.

    Returns
    -------
    None.
    
    Test Cases:
    
    >>> from prtpy.binners import Binner, BinnerKeepingContents, BinnerKeepingSums
    >>> printbins(online_ffk_supply(bkc_ffk(), binsize=5, items=[1,2,3]))
    Bin #0: [(1, 0), (2, 1)], sum=3.0
    Bin #1: [(3, 2), (1, 0)], sum=4.0
    Bin #2: [(2, 1), (3, 2)], sum=5.0
    Bin #3: [(1, 0), (2, 1)], sum=3.0
    Bin #4: [(1, 0)], sum=1.0
    Bin #5: [(1, 0)], sum=1.0
    Bin #6: [(1, 0)], sum=1.0
    
    >>> printbins(online_ffk_supply(bkc_ffk(), binsize=6, items=[1,2,3]))
    Bin #0: [(1, 0), (2, 1), (3, 2)], sum=6.0
    Bin #1: [(1, 0), (2, 1), (3, 2)], sum=6.0
    Bin #2: [(1, 0), (2, 1)], sum=3.0
    Bin #3: [(1, 0)], sum=1.0
    Bin #4: [(1, 0)], sum=1.0
    Bin #5: [(1, 0)], sum=1.0
    
    >>> printbins(online_ffk_supply(bkc_ffk(), binsize=4, items=[1,2,2]))
    Bin #0: [(1, 0), (2, 1)], sum=3.0
    Bin #1: [(2, 2), (1, 0)], sum=3.0
    
    >>> printbins(online_ffk_supply(bkc_ffk(), binsize=10, items=[3,5,5]))
    Bin #0: [(3, 0), (5, 1)], sum=8.0
    Bin #1: [(5, 2), (3, 0)], sum=8.0
    Bin #2: [(5, 1), (5, 2)], sum=10.0
    Bin #3: [(3, 0), (5, 1)], sum=8.0
    Bin #4: [(5, 2), (3, 0)], sum=8.0
    Bin #5: [(3, 0)], sum=3.0
    
    >>> printbins(online_ffk_supply(bkc_ffk(), binsize=2, items=[1,1,1]))
    Bin #0: [(1, 0), (1, 1)], sum=2.0
    Bin #1: [(1, 2)], sum=1.0
    
    # >>> printbins(online_ffk_supply(bkc_ffk(), binsize=25, items=[11,12,13]))
    
    >>> printbins(online_ffk_supply(bkc_ffk(), binsize=55, items=[11,22,33]))
    Bin #0: [(11, 0), (22, 1)], sum=33.0
    Bin #1: [(33, 2), (11, 0)], sum=44.0
    Bin #2: [(22, 1), (33, 2)], sum=55.0
    Bin #3: [(11, 0), (22, 1)], sum=33.0
    Bin #4: [(11, 0)], sum=11.0
    Bin #5: [(11, 0)], sum=11.0
    Bin #6: [(11, 0)], sum=11.0
    
    '''
    # logger = logging.getLogger(__name__)
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception("Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    bins = binner.new_bins(1)
    numbins = 1
    
    # below lines has been used when lcm of the number is used to create instance.
    new_items_instance = newinstance.create_instance_Dk(new_items)
    
    for instance in new_items_instance:
        for item in instance:
            value = binner.valueof(item)
            # logging.info(f"FFk: value: {value}")
            if value > binsize:
                raise ValueError(f"Item {item} has size {value} which is larger than the bin size {binsize}.")
            ibin = 0
            while ibin  < numbins:
                if binner.sums(bins)[ibin] + value <= binsize and not binner.check(item, bins[1][ibin]):
                    binner.add_item_to_bin(bins, item, ibin)
                    #binner.add_item_to_bin(item, ibin, item_index)
                    #print("Bins:{0}, Bin Index:{2}, Item:{1} ".format(bins.bins, item, ibin))
                    break
                ibin += 1
            else:  # if not added to any bin
                bins = binner.add_empty_bins(bins, 1)
                numbins += 1
                binner.add_item_to_bin(bins, item, ibin)
    return bins

def online_ffk_supply_withvectork(
        binner: Binner,
        binsize: float,
        items: List[Any],
        vector_k: List
        ):
    '''
    Pack the given list of lists of items into bins using the online first-fit algorithm.
    The online algorithm handles the items in the order they are given.
    An item in the list is in the form of (value, index).
    A given list of items lists can be: [[(1, 0), (2, 1), (3, 2)], [(1, 0), (2, 1)], [(1, 0)]]
    
    Parameters
    ----------
    binner : Binner
        DESCRIPTION.
    binsize_ float : TYPE
        DESCRIPTION.
    items : List[Any]
        DESCRIPTION.
    vector_k: List
        DESCRIPTION.

    Returns
    -------
    None.
    
    Test Cases:
    
    >>> from prtpy.binners import Binner, BinnerKeepingContents, BinnerKeepingSums
    >>> printbins(online_ffk_supply_withvectork(bkc_ffk(), binsize=5, items=[1,2,3], vector_k = [6, 3, 2]))
    Bin #0: [(1, 0), (2, 1)], sum=3.0
    Bin #1: [(3, 2), (1, 0)], sum=4.0
    Bin #2: [(2, 1), (3, 2)], sum=5.0
    Bin #3: [(1, 0), (2, 1)], sum=3.0
    Bin #4: [(1, 0)], sum=1.0
    Bin #5: [(1, 0)], sum=1.0
    Bin #6: [(1, 0)], sum=1.0
    
    >>> printbins(online_ffk_supply_withvectork(bkc_ffk(), binsize=6, items=[1,2,3], vector_k=[1, 1, 1]))
    Bin #0: [(1, 0), (2, 1), (3, 2)], sum=6.0
    
    >>> printbins(online_ffk_supply_withvectork(bkc_ffk(), binsize=4, items=[1,2,2], vector_k=[2, 1, 1]))
    Bin #0: [(1, 0), (2, 1)], sum=3.0
    Bin #1: [(2, 2), (1, 0)], sum=3.0
    
    >>> printbins(online_ffk_supply_withvectork(bkc_ffk(), binsize=10, items=[3,5,5], vector_k=[5, 3, 3]))
    Bin #0: [(3, 0), (5, 1)], sum=8.0
    Bin #1: [(5, 2), (3, 0)], sum=8.0
    Bin #2: [(5, 1), (5, 2)], sum=10.0
    Bin #3: [(3, 0), (5, 1)], sum=8.0
    Bin #4: [(5, 2), (3, 0)], sum=8.0
    Bin #5: [(3, 0)], sum=3.0
    
    >>> printbins(online_ffk_supply_withvectork(bkc_ffk(), binsize=2, items=[1,1,1], vector_k=[2, 2, 2]))
    Bin #0: [(1, 0), (1, 1)], sum=2.0
    Bin #1: [(1, 2), (1, 0)], sum=2.0
    Bin #2: [(1, 1), (1, 2)], sum=2.0
    
    >>> printbins(online_ffk_supply_withvectork(bkc_ffk(), binsize=25, items=[11,12,13], vector_k=[3, 3, 4]))
    Bin #0: [(11, 0), (12, 1)], sum=23.0
    Bin #1: [(13, 2), (11, 0)], sum=24.0
    Bin #2: [(12, 1), (13, 2)], sum=25.0
    Bin #3: [(11, 0), (12, 1)], sum=23.0
    Bin #4: [(13, 2)], sum=13.0
    Bin #5: [(13, 2)], sum=13.0
    
    >>> printbins(online_ffk_supply_withvectork(bkc_ffk(), binsize=55, items=[11,22,33], vector_k=[6, 3, 2]))
    Bin #0: [(11, 0), (22, 1)], sum=33.0
    Bin #1: [(33, 2), (11, 0)], sum=44.0
    Bin #2: [(22, 1), (33, 2)], sum=55.0
    Bin #3: [(11, 0), (22, 1)], sum=33.0
    Bin #4: [(11, 0)], sum=11.0
    Bin #5: [(11, 0)], sum=11.0
    Bin #6: [(11, 0)], sum=11.0
    
    >>> printbins(online_ffk_supply_withvectork(bkc_ffk(), binsize=2, items=[0.5,1.5,1], vector_k=[6, 2, 3]))
    Bin #0: [(0.5, 0), (1.5, 1)], sum=2.0
    Bin #1: [(1, 2), (0.5, 0)], sum=1.5
    Bin #2: [(1.5, 1), (0.5, 0)], sum=2.0
    Bin #3: [(1, 2), (0.5, 0)], sum=1.5
    Bin #4: [(1, 2), (0.5, 0)], sum=1.5
    Bin #5: [(0.5, 0)], sum=0.5
    '''
    # if __name__=="__main__":
    #     logger = logging.getLogger(__name__)
    # else:
    #     logger = logging.getLogger("online_ffk_supply_withvectork")
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception("Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    bins = binner.new_bins(1)
    numbins = 1
    
    # compute the new_items_instance where each item i occur vector_k[i] times.
    new_items_instance = newinstance.create_instance_Dk_withvectork(new_items, vector_k)
    
    for instance in new_items_instance:
        for item in instance:
            value = binner.valueof(item)
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
    
    # logger.info("Done with the function.")
    return bins


def online_ffk_supply_withvectork_test(
        binner: Binner,
        binsize: float,
        items: List[Any],
        vector_k: List
        ):
    '''
    Pack the given list of lists of items into bins using the online first-fit algorithm.
    The online algorithm handles the items in the order they are given.
    An item in the list is in the form of (value, index).
    A given list of items lists can be: [[(1, 0), (2, 1), (3, 2)], [(1, 0), (2, 1)], [(1, 0)]]
    
    Parameters
    ----------
    binner : Binner
        DESCRIPTION.
    binsize_ float : TYPE
        DESCRIPTION.
    items : List[Any]
        DESCRIPTION.
    vector_k: List
        DESCRIPTION.

    Returns
    -------
    None.
    
    Test Cases:
    
    >>> from prtpy.binners import Binner, BinnerKeepingContents, BinnerKeepingSums
    >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=5, items=[1,2,3], vector_k = [(6,0), (3,1), (2,2)]))
    Bin #0: [(1, 0), (2, 1)], sum=3.0
    Bin #1: [(3, 2), (1, 0)], sum=4.0
    Bin #2: [(2, 1), (3, 2)], sum=5.0
    Bin #3: [(1, 0), (2, 1)], sum=3.0
    Bin #4: [(1, 0)], sum=1.0
    Bin #5: [(1, 0)], sum=1.0
    Bin #6: [(1, 0)], sum=1.0
    
    >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=6, items=[1,2,3], vector_k=[(1,0), (1,1), (1,2)]))
    Bin #0: [(1, 0), (2, 1), (3, 2)], sum=6.0
    
    >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=4, items=[1,2,2], vector_k=[(2,0), (1,1), (1,2)]))
    Bin #0: [(1, 0), (2, 1)], sum=3.0
    Bin #1: [(2, 2), (1, 0)], sum=3.0
    
    >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=10, items=[3,5,5], vector_k=[(5,0), (3,1), (3,2)]))
    Bin #0: [(3, 0), (5, 1)], sum=8.0
    Bin #1: [(5, 2), (3, 0)], sum=8.0
    Bin #2: [(5, 1), (5, 2)], sum=10.0
    Bin #3: [(3, 0), (5, 1)], sum=8.0
    Bin #4: [(5, 2), (3, 0)], sum=8.0
    Bin #5: [(3, 0)], sum=3.0
    
    >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=2, items=[1,1,1], vector_k=[(2,0), (2,1), (2,2)]))
    Bin #0: [(1, 0), (1, 1)], sum=2.0
    Bin #1: [(1, 2), (1, 0)], sum=2.0
    Bin #2: [(1, 1), (1, 2)], sum=2.0
    
    >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=25, items=[11,12,13], vector_k=[(3,0), (3,1), (4,2)]))
    Bin #0: [(11, 0), (12, 1)], sum=23.0
    Bin #1: [(13, 2), (11, 0)], sum=24.0
    Bin #2: [(12, 1), (13, 2)], sum=25.0
    Bin #3: [(11, 0), (12, 1)], sum=23.0
    Bin #4: [(13, 2)], sum=13.0
    Bin #5: [(13, 2)], sum=13.0
    
    >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=55, items=[11,22,33], vector_k=[(6,0), (3,1), (2,2)]))
    Bin #0: [(11, 0), (22, 1)], sum=33.0
    Bin #1: [(33, 2), (11, 0)], sum=44.0
    Bin #2: [(22, 1), (33, 2)], sum=55.0
    Bin #3: [(11, 0), (22, 1)], sum=33.0
    Bin #4: [(11, 0)], sum=11.0
    Bin #5: [(11, 0)], sum=11.0
    Bin #6: [(11, 0)], sum=11.0
    
    >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=2, items=[0.5,1.5,1], vector_k=[(6,0), (2,1), (3,2)]))
    Bin #0: [(0.5, 0), (1.5, 1)], sum=2.0
    Bin #1: [(1, 2), (0.5, 0)], sum=1.5
    Bin #2: [(1.5, 1), (0.5, 0)], sum=2.0
    Bin #3: [(1, 2), (0.5, 0)], sum=1.5
    Bin #4: [(1, 2), (0.5, 0)], sum=1.5
    Bin #5: [(0.5, 0)], sum=0.5
    
    
    # >>> printbins(online_ffk_supply_withvectork_test(bkc_ffk(), binsize=1.5, items=[(1.5,1),(1,2)], vector_k=[(2,1), (3,2)]))
    '''
    # if __name__=="__main__":
    #     logger = logging.getLogger(__name__)
    # else:
    #     logger = logging.getLogger("online_ffk_supply_withvectork")
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception("Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    bins = binner.new_bins(1)
    numbins = 1
    
    
    # compute the new_items_instance where each item i occur vector_k[i] times.
    new_items_instance = newinstance.create_instance_Dk_withvectork_test(new_items, vector_k)
    
    for instance in new_items_instance:
        for item in instance:
            value = binner.valueof(item)
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
    
    # logger.info("Done with the function.")
    return bins
            
            
                
def online_ffk(
    binner: Binner,
    binsize: float,
    items: List[any],
    #valueof: Callable[[Any], float] = lambda x: x,
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
    >>> printbins(online_ffk(bkc_ffk(), binsize=2, items=[0.5,1.5,1], k=2))
    Bin #0: [(0.5, 0), (1.5, 1)], sum=2.0
    Bin #1: [(1, 2), (0.5, 0)], sum=1.5
    Bin #2: [(1.5, 1)], sum=1.5
    Bin #3: [(1, 2)], sum=1.0
    >>> printbins(online_ffk(bkc_ffk(), binsize = 100, items = [26, 31, 17, 6, 12, 35, 22, 24, 13, 14], k=2))     
    Bin #0: [(26, 0), (31, 1), (17, 2), (6, 3), (12, 4)], sum=92.0
    Bin #1: [(35, 5), (22, 6), (24, 7), (13, 8), (6, 3)], sum=100.0
    Bin #2: [(14, 9), (26, 0), (31, 1), (17, 2), (12, 4)], sum=100.0
    Bin #3: [(35, 5), (22, 6), (24, 7), (13, 8)], sum=94.0
    Bin #4: [(14, 9)], sum=14.0
    
    
    >>> binner = bkc_ffk()
    >>> bins = online_ffk(binner, binsize=21, items = [0.2, 0.4, 0.8, 1.7, 3, 6.5, 14], k =20)
    >>> print(binner.numbins(bins))                    
    30
    
    # >>> printbins(online_ffk(bkc_ffk(), binsize=1000, items=[371, 659, 113, 47, 485, 3, 228, 419, 468, 581, 626], k=9))

    
    # >>> import randomitems
    # >>> itemset = randomitems.generateitems(tot_no_of_items=8, binsize=10, groupsize=4)
    # >>> print(f"Items: {itemset}")
    # >>> printbins(online_ffk(bkc_ffk(), binsize=10, items= itemset, k=2))
    
    # >>> binner = BinnerKeepingContents()
    # >>> itemset = [6/101,6/101,6/101,6/101,6/101,6/101,6/101,10/101,10/101,10/101,10/101,10/101,10/101,10/101, 16/101,16/101,16/101, 34/101,34/101,34/101,34/101,34/101,34/101,34/101,34/101,34/101,34/101, 51/101,51/101,51/101,51/101,51/101,51/101,51/101,51/101,51/101,51/101]
    # >>> ffkbins = binner.numbins(online_ffk(bkc_ffk(), binsize=1, items= itemset, k=2))
    # >>> for i in range(1,21):
    # ...     ffkbins = binner.numbins(online_ffk(bkc_ffk(), binsize=1, items= itemset, k=i))
    # ...     print(f"FFk_Bins: {ffkbins}, OPT_Bins: {i*10}, k: {i}")
    """
    
    # logger = logging.getLogger(__name__)
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception("Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    bins = binner.new_bins(1)
    numbins = 1
   
    for times in range(k):
        for item in new_items:
            value = binner.valueof(item)
            # logging.info(f"FFk: value: {value}")
            if value > binsize:
                raise ValueError(f"Item {item} has size {value} which is larger than the bin size {binsize}.")
            ibin = 0
            while ibin  < numbins:
                if binner.sums(bins)[ibin] + value <= binsize and not binner.check(item, bins[1][ibin]):
                    binner.add_item_to_bin(bins, item, ibin)
                    #binner.add_item_to_bin(item, ibin, item_index)
                    #print("Bins:{0}, Bin Index:{2}, Item:{1} ".format(bins.bins, item, ibin))
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
    #valueof: Callable[[Any], float] = lambda x: x,
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
    # logging.basicConfig(filename='ffk.log', level=logging.DEBUG)
    # logger = logging.getLogger(__name__)
    # logger.info(f"FFDk:")
    
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception("Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    #new_items = itemslist_to_tuplelist(items)
    sorted_new_items = sorted(new_items, key=binner.valueof, reverse=True)
    
    # logger.info(f"FFDk: Sorted Items: {sorted_new_items}")
    
    return online_ffk(
        binner,
        binsize, 
        sorted_new_items,
        k
    )

def decreasing_ffk_supply_withvectork(
    binner: Binner,
    binsize: float,
    items: List[any],
    #valueof: Callable[[Any], float] = lambda x: x,
    vector_k: List
) -> BinsArray:
    
    '''
    Pack the given items into bins using the first-fit-decreasing algorithm.
    It sorts the items by descending value, and then runs first-fit.

    >>> from prtpy.binners import Binner, BinnerKeepingContents, BinnerKeepingSums
    >>> printbins(decreasing_ffk_supply_withvectork(bkc_ffk(), binsize=5, items=[1,2,3], vector_k = [6, 3, 2]))
    Bin #0: [(3, 2), (2, 1)], sum=5.0
    Bin #1: [(1, 0), (3, 2)], sum=4.0
    Bin #2: [(2, 1), (1, 0)], sum=3.0
    Bin #3: [(2, 1), (1, 0)], sum=3.0
    Bin #4: [(1, 0)], sum=1.0
    Bin #5: [(1, 0)], sum=1.0
    Bin #6: [(1, 0)], sum=1.0
    
    >>> printbins(decreasing_ffk_supply_withvectork(bkc_ffk(), binsize=6, items=[1,2,3], vector_k=[1, 1, 1]))
    Bin #0: [(3, 2), (2, 1), (1, 0)], sum=6.0
    
    >>> printbins(decreasing_ffk_supply_withvectork(bkc_ffk(), binsize=4, items=[1,2,2], vector_k=[2, 1, 1]))
    Bin #0: [(2, 1), (2, 2)], sum=4.0
    Bin #1: [(1, 0)], sum=1.0
    Bin #2: [(1, 0)], sum=1.0
    
    >>> printbins(decreasing_ffk_supply_withvectork(bkc_ffk(), binsize=10, items=[3,5,5], vector_k=[5, 3, 3]))
    Bin #0: [(5, 1), (5, 2)], sum=10.0
    Bin #1: [(3, 0), (5, 1)], sum=8.0
    Bin #2: [(5, 2), (3, 0)], sum=8.0
    Bin #3: [(5, 1), (5, 2)], sum=10.0
    Bin #4: [(3, 0)], sum=3.0
    Bin #5: [(3, 0)], sum=3.0
    Bin #6: [(3, 0)], sum=3.0
    
    >>> printbins(decreasing_ffk_supply_withvectork(bkc_ffk(), binsize=2, items=[1,1,1], vector_k=[2, 2, 2]))
    Bin #0: [(1, 0), (1, 1)], sum=2.0
    Bin #1: [(1, 2), (1, 0)], sum=2.0
    Bin #2: [(1, 1), (1, 2)], sum=2.0
    
    >>> printbins(decreasing_ffk_supply_withvectork(bkc_ffk(), binsize=25, items=[11,12,13], vector_k=[3, 3, 4]))
    Bin #0: [(13, 2), (12, 1)], sum=25.0
    Bin #1: [(11, 0), (13, 2)], sum=24.0
    Bin #2: [(12, 1), (11, 0)], sum=23.0
    Bin #3: [(13, 2), (12, 1)], sum=25.0
    Bin #4: [(11, 0), (13, 2)], sum=24.0
    
    >>> printbins(decreasing_ffk_supply_withvectork(bkc_ffk(), binsize=55, items=[11,22,33], vector_k=[6, 3, 2]))
    Bin #0: [(33, 2), (22, 1)], sum=55.0
    Bin #1: [(11, 0), (33, 2)], sum=44.0
    Bin #2: [(22, 1), (11, 0)], sum=33.0
    Bin #3: [(22, 1), (11, 0)], sum=33.0
    Bin #4: [(11, 0)], sum=11.0
    Bin #5: [(11, 0)], sum=11.0
    Bin #6: [(11, 0)], sum=11.0
    
    '''
    # logging.basicConfig(filename='ffk.log', level=logging.DEBUG)
    # logger = logging.getLogger(__name__)
    # logger.info(f"FFDk:")
    
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception("Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    #new_items = itemslist_to_tuplelist(items)
    sorted_new_items = sorted(new_items, key=binner.valueof, reverse=True)
    
    # creating a new vector_k
    new_vector_k = [value_k for _, value_k in sorted(list(zip(new_items, vector_k)), key = lambda x: x[0][1] , reverse=True)]
    
    # logger.info(f"FFDk: Sorted Items: {sorted_new_items}")
    
    return online_ffk_supply_withvectork(
        binner,
        binsize, 
        sorted_new_items,
        new_vector_k
    )


def decreasing_ffk_supply_withvectork_test(
    binner: Binner,
    binsize: float,
    items: List[any],
    #valueof: Callable[[Any], float] = lambda x: x,
    vector_k: List
) -> BinsArray:
    
    '''
    Pack the given items into bins using the first-fit-decreasing algorithm.
    It sorts the items by descending value, and then runs first-fit.

    >>> from prtpy.binners import Binner, BinnerKeepingContents, BinnerKeepingSums
    >>> printbins(decreasing_ffk_supply_withvectork_test(bkc_ffk(), binsize=5, items=[1,2,3], vector_k = [(6,0), (3,1), (2,2)]))
    Bin #0: [(3, 2), (2, 1)], sum=5.0
    Bin #1: [(1, 0), (3, 2)], sum=4.0
    Bin #2: [(2, 1), (1, 0)], sum=3.0
    Bin #3: [(2, 1), (1, 0)], sum=3.0
    Bin #4: [(1, 0)], sum=1.0
    Bin #5: [(1, 0)], sum=1.0
    Bin #6: [(1, 0)], sum=1.0
    
    >>> printbins(decreasing_ffk_supply_withvectork_test(bkc_ffk(), binsize=6, items=[1,2,3], vector_k=[(1,0), (1,1), (1,2)]))
    Bin #0: [(3, 2), (2, 1), (1, 0)], sum=6.0
    
    >>> printbins(decreasing_ffk_supply_withvectork_test(bkc_ffk(), binsize=4, items=[1,2,2], vector_k=[(2,0), (1,1), (1,2)]))
    Bin #0: [(2, 1), (2, 2)], sum=4.0
    Bin #1: [(1, 0)], sum=1.0
    Bin #2: [(1, 0)], sum=1.0
    
    >>> printbins(decreasing_ffk_supply_withvectork_test(bkc_ffk(), binsize=10, items=[3,5,5], vector_k=[(5,0), (3,2), (3,1)]))
    Bin #0: [(5, 1), (5, 2)], sum=10.0
    Bin #1: [(3, 0), (5, 1)], sum=8.0
    Bin #2: [(5, 2), (3, 0)], sum=8.0
    Bin #3: [(5, 1), (5, 2)], sum=10.0
    Bin #4: [(3, 0)], sum=3.0
    Bin #5: [(3, 0)], sum=3.0
    Bin #6: [(3, 0)], sum=3.0
    
    >>> printbins(decreasing_ffk_supply_withvectork_test(bkc_ffk(), binsize=2, items=[1,1,1], vector_k=[(2,0), (2,1), (2,2)]))
    Bin #0: [(1, 0), (1, 1)], sum=2.0
    Bin #1: [(1, 2), (1, 0)], sum=2.0
    Bin #2: [(1, 1), (1, 2)], sum=2.0
    
    >>> printbins(decreasing_ffk_supply_withvectork_test(bkc_ffk(), binsize=25, items=[11,12,13], vector_k=[(3,0), (3,1), (4,2)]))
    Bin #0: [(13, 2), (12, 1)], sum=25.0
    Bin #1: [(11, 0), (13, 2)], sum=24.0
    Bin #2: [(12, 1), (11, 0)], sum=23.0
    Bin #3: [(13, 2), (12, 1)], sum=25.0
    Bin #4: [(11, 0), (13, 2)], sum=24.0
    
    >>> printbins(decreasing_ffk_supply_withvectork_test(bkc_ffk(), binsize=55, items=[11,22,33], vector_k=[(6,0), (3,1), (2,2)]))
    Bin #0: [(33, 2), (22, 1)], sum=55.0
    Bin #1: [(11, 0), (33, 2)], sum=44.0
    Bin #2: [(22, 1), (11, 0)], sum=33.0
    Bin #3: [(22, 1), (11, 0)], sum=33.0
    Bin #4: [(11, 0)], sum=11.0
    Bin #5: [(11, 0)], sum=11.0
    Bin #6: [(11, 0)], sum=11.0
    
    # >>> printbins(decreasing_ffk_supply_withvectork_test(bkc_ffk(), binsize=10, items=[3,5,5], vector_k=[(5,0), (3,1), (3,2)]))
    
    '''
    # logging.basicConfig(filename='ffk.log', level=logging.DEBUG)
    # logger = logging.getLogger(__name__)
    # logger.info(f"FFDk:")
    
    if all(isinstance(item, tuple) and len(item)==2 for item in items):
        if not all(len(item)==2 for item in items):
            raise Exception("Not all items are of type (Any, int).")
        else:
            new_items = items
    else:
        new_items = itemslist_to_tuplelist(items)
    
    #new_items = itemslist_to_tuplelist(items)
    sorted_new_items = sorted(new_items, key=binner.valueof, reverse=True)
    
    # creating a new vector_k
    new_vector_k = [value_k for _, value_k in sorted(list(zip(new_items, vector_k)), key = lambda x: x[0][1] , reverse=True)]
    
    # logger.info(f"FFDk: Sorted Items: {sorted_new_items}")
    
    return online_ffk_supply_withvectork_test(
        binner,
        binsize, 
        sorted_new_items,
        new_vector_k
    )



if __name__ == "__main__":
    import doctest

    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))
