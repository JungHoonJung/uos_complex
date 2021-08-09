import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def FF_corr(facet_dict, scatter = True):
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
    ff_pair = {}
    for node in tqdm(facet_dict):
        fs = []
        for facet in facet_dict[node]:
            fs.append(len(facet)-1) # gathering all of facet sizes of target node.
        
        corr=0
        for i in fs:
            for j in fs:
                ff_pair[i,j] = ff_pair.get((i,j), 0) + 1
                corr += 2*(i==j) -1 # 1 if i == j else -1
            corr -= 1
            ff_pair[i,i] -= 1 # remove self correlation
        ff_corr[node] = corr
    
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
    plt.scatter(*pairs.T, s = weight, label = f'corr. = {corr}')
    plt.legend()
    return corr, ff_pair

    

    

