from unittest import skip
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
from numba.typed import Dict, List

def disparity1(facet_dict):
    """Measure disparity of a given facet distribution.

    Parameters
    ----------
    facet_dict : dict
        (node : facets) pair distribution

    Returns
    -------
    dict
        (node : disparity) dictionary

    some detail explanation. 
    """    
    disp = {}
    for node in tqdm(facet_dict):
        s = 0 #sum of s
        ss = 0 #squared sum of s
        for facet in facet_dict[node]:
            k= len(facet) - 1
            s += k
            ss += k*k
        if s == 0:
            disp[node] = 0
            continue
        y = ss/s/s # disparity $Y = \sum (\frac{s}{\sum s})^2 = (\sum s^2)/(\sum s)^2$
        disp[node]=y
    return disp

def disparity2(facet_dict):
    """Measure disparity of a given facet distribution.

    Parameters
    ----------
    facet_dict : dict
        (node : facets) pair distribution

    Returns
    -------
    dict
        (node : disparity) dictionary
    """    
    disp = {}
    for node in tqdm(facet_dict):
        s = 0  #sum of s
        ss = 0 #squared sum of s
        nn_set = set()
        for facet in facet_dict[node]:
            k= len(facet) - 1
            ss += k*k
            for nn in facet:
                nn_set.add(nn)
        nn = len(nn_set) -1 # remove self node.
        if nn == 0:
            disp[node] = 0
            continue
        y = ss/nn/nn # disparity $Y = \sum (\frac{s}{ |\{nn\}| })^2 = (\sum s^2)/(|\{nn\}|)^2$
        disp[node] = y
    return disp

@njit
def _FFcor(flist, fdict = None):
    ret = 0
    if fdict is None:
        for i, f1 in enumerate(flist):
            for j, f2 in enumerate(flist):
                if i>j:
                    ret += 2*(f1==f2) -1
    else:
        for i, f1 in enumerate(flist):
            for j, f2 in enumerate(flist):
                if i>j:
                    ret += 2*(f1==f2) -1
                    fdict[f1,f2] = fdict.get((f1,f2),0)+1
    return ret


def FF_corr(facet_dict, skip_scatter = False):
    """Facet-Facet correlation function.

    Parameters
    ----------
    facet_dict : dict
        node-facet(list) pair dictionary.
    scatter : bool, optional
        If True, the scatter plot for facet-facet size correlation scatter, by default True.
    
    Returns
    ----------
    corr : float
        The facet-facet correlation number
    ff_pair : dict
        the dictionary whose key is facet-facet pair and value is its degeneracy.

    """    
    ff_corr = {}
    ff_pair = Dict()
    ff_pair[1,1] = 1 # type set of numba typed Dict
    del ff_pair[1,1] # remove dummy data
    tot_corr = 0
    tot_N = 0
    for node in tqdm(facet_dict):
        fs = List()
        for facet in facet_dict[node]:
            fs.append(len(facet)-1) # gathering all of facet sizes of target node.
        
        if skip_scatter:
            tot_corr += _FFcor(fs)
            tot_N += (len(fs)-1)*(len(fs))/2
            continue
        else:
            corr = _FFcor(fs, ff_pair)
            ff_corr[node] = corr

    if skip_scatter:
        return tot_corr/tot_N

    pairs=  []
    weight = []
    for pair in ff_pair:
        pairs.append(pair)
        weight.append(ff_pair[pair])
    pairs = np.array(pairs)
    corr = 0
    for node in ff_corr:
        corr+= ff_corr[node]
    corr /= sum(weight)
    plt.scatter(*pairs.T, s = np.log10(weight)+1, label = f'corr. = {corr}')
    plt.legend()
    return corr, ff_pair

    
def simplicial_degree(facet_dict, loglog = False):
    facets  = facet_dict
    degrees = np.array([len(facets[f]) for f in facets])
    plt.hist(degrees, bins = np.arange(degrees.min(), degrees.max()+1), density = True)
    plt.title('simplicial degree')
    if loglog:
        plt.xscale('log')
        plt.yscale('log')

def facet_size_dist(facet_dict):
    facets = facet_dict
    sizes =[]
    unique_facet = set()
    for f in facets:
        for sets in facets[f]:
            unique_facet.add(tuple(sets))
    for f in unique_facet:
        sizes.append(len(f))
    sizes = np.array(sizes)
    plt.hist(sizes, bins = np.arange(sizes.min(), sizes.max()+1), density=True)
    plt.title('facet size dist')



