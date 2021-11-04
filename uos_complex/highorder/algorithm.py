import networkx as nx
import numpy as np
from numba import njit, prange
from numba.typed import Dict, List
from numba.types import int64, int32, ListType, DictType
from uos_complex.highorder.dataset import HGData
from tqdm import tqdm


Lcomplex_type = ListType(ListType(int64))
Scomplex_type = (int32, ListType(int32))

def numbized_dict(dic):
    res = Dict()
    for i in dic:
        res[i] = List(dic[i])
    return res

def to_numba_dict(facets, simps):
    return numbized_dict(facets), numbized_dict(simps)

def np_nerve_complex_from(data):    
    """Get nerve complex

    Returns:
        [type]: Dictionary which has (node - List of simplex(List)) as a key - value pair.
    """        
    k = [int(ks) for ks in data]
    k.sort(reverse= True)
    facets = Dict.empty(*Scomplex_type)
    simps = set()
    simplices = Dict.empty(*Scomplex_type)
    nsimplex = 0
    for i in tqdm(k):
        for j in data[i]:
            simp = tuple(sorted(j))
            simpset = List(simp)
            if simp in simps:  #overlap check
                continue
            simps.add(simp)
            simplices[nsimplex+1] = List(simp)
            nsimplex += 1
            facet  = True
            #nfacet  = False
            for node in simp: 
                faces = facets.get(node, List.empty_list(int32))
                if not faces or facet:
                    break # this hyperedge is facet
                else:
                    for face in faces:
                        if len(set(simpset) - set(simplices[face]))== 0: #if there is larger than this one 
                            facet = False            #(i.e. this is a one of the face of an existing facet.)
                        break
                if not facet:
                    break
            if facet:
                for node in simp: 
                    faces = facets.get(node, List.empty_list(int32))
                    faces.append(nsimplex)
                    facets[node] = faces
    return facets, simplices

def nerve_complex_from(data):    
    """Get nerve complex

    Returns:
        [type]: Dictionary which has (node - List of simplex(List)) as a key - value pair.
    """        
    k = [int(ks) for ks in data]
    k.sort(reverse= True)
    facets = {}
    simps = set()
    simplices = {}
    nsimplex = 0
    for i in tqdm(k):
        for j in data[i]:
            simp = tuple(sorted(j))
            simpset = set(simp)
            if simp in simps:  #overlap check
                continue
            simps.add(simp)
            
            facet  = True
            #nfacet  = False
            for node in simp: 
                faces = facets.get(node, [])
                for face in faces:
                    if len(simpset - simplices[face])== 0: #if there is larger than this one 
                        facet = False                      #(i.e. this is a one of the face of an existing facet.)
                    break
                if not facet:
                    break
            if facet:
                simplices[nsimplex+1] = simpset
                nsimplex += 1
                for node in simp: 
                    faces = facets.get(node, [])
                    faces.append(nsimplex)
                    facets[node] = faces
    return facets, simplices

def clique_complex_from(data):    
    facets =  Dict.empty(Scomplex_type)
    simps = set()
    simplices = List.empty(Lcomplex_type)
    for j in tqdm(nx.find_cliques(data.get_clique_network())):
        simp = tuple(sorted(j))
        simpset = set(simp)
        if simp in simps:  #overlap check
            continue
        simps.add(simp)

        facet  = True
        #nfacet  = False
        for node in simp: 
            faces = facets.get(node, [])
            if not faces or facet:
                break # this hyperedge is facet
            else:
                for face in faces:
                    if len(simpset - face) == 0: #if there is larger than this one 
                        facet = False            #(i.e. this is a one of the face of an existing facet.)
                    break
            if not facet:
                break
        if facet:
            for node in simp: 
                faces = facets.get(node, [])
                faces.append(simpset)
                facets[node] = faces
    return facets



############# MEASUREMENT ##########################


def nodeid(facets):
    nid = Dict.empty(int32, int32) 
    for i, ni in enumerate(facets):
        nid[i] = ni
    return nid



@njit
def simplicial_degrees(facets:Dict):
    """get micro(node) level simplicial degree.

    Parameters
    ----------
    facet : Dict
        Simplicial complex which is made with node-Lsimplex pair

    Returns
    ----------
    np.ndarray
        node - degree pair
    """
    degrees = -np.ones((len(facets),), dtype=np.int32)
    for i in facets:
        degrees[i-1] = len(facets[i]) # np : C flags (0 start), facets: fortran flags (1 start)
    return degrees


@njit
def local_simplicial_degrees(facets:Dict, simps: Dict):
    """get simplex level simplicial degree.

    Parameters
    ----------
    facets : Dict
        Simplicial complex which is made with node-Lsimplex pair
    
    simps : Dict
        Simplicial complex which is made with simplex-node pair

    Returns
    ----------
    np.ndarray
        degrees of each simplex
    """
    degrees = simplicial_degrees(facets)
    ldegrees = -np.ones((len(simps),), dtype=np.int32)
    for i in simps:
        ldegrees[i-1] = 0
        for node in simps[i]:
            ldegrees[i-1] += degrees[node-1] - 1
    return ldegrees

@njit
def k(simps):
    res = -np.ones((len(simps),), dtype=np.int32)
    for i in simps:
        res[i-1] = len(simps[i])-1
    return res

dims = k

@njit
def Nnn(facets, simps):
    """Number of nearest neighbors.

    Parameters
    ----------
    facets : Dict
        Simplicial complex which is made with node-Lsimplex pair
    
    simps : Dict
        Simplicial complex which is made with simplex-node pair

    Returns
    ----------
    np.ndarray
        Number of nearest neighbors.
    """
    res = -np.ones((len(facets),), dtype=np.int32)
    for i in facets:
        nn = Dict.empty(int32, int32)
        simplices = facets[i]
        for simplex in simplices:
            for node in simps[simplex]:
                nn[node] = 1
        res[i-1] = len(nn) - 1 # remove self node 
    return res

@njit(parallel=True)
def local_Nnn(facets, simps):
    """Number of nearest neighbors of simplices.

    Parameters
    ----------
    facets : Dict
        Simplicial complex which is made with node-Lsimplex pair
    
    simps : Dict
        Simplicial complex which is made with simplex-node pair

    Returns
    ----------
    np.ndarray
        Number of nearest neighbors of simplices.
    """
    res = -np.ones((len(simps),), dtype=np.int32)
    tot = len(simps)
    for s in prange(tot):
        nn = Dict.empty(int32, int32)
        for snode in simps[s+1]:
            for i in facets[snode]:
                simplices = simps[i]
                for node in simplices:
                    nn[node] = 1
        res[s] = len(nn) - len(simps[s+1]) # remove nodes of self
    return res
    
@njit
def strength(facets, simps):
    res = -np.ones((len(facets),), dtype=np.int32)
    for i in facets:
        res[i-1] = 0
        for sid in facets[i]:
            res[i-1] += len(simps[sid])
    return res
    
@njit
def local_strength(facets, simps):
    res = -np.ones((len(simps),), dtype=np.int32)
    ms = strength(facets, simps)
    for s in simps:
        res[s-1] = 0
        for node in simps[s]:
            res[s-1] += ms[node-1]
        k = len(simps[s])
        res[s-1] -= k*k - k
    return res



@njit(parallel = True)
def local_connectivity(facets, simps):
    res = -np.ones((len(simps),), dtype=np.int32)
    tot = len(simps)
    for s in prange(tot):
        res[s] = 0
        sset = Dict.empty(int64, int64)
        for node in simps[s+1]:
            ch =False
            for simplex in facets[node]:
                if simplex ==s+1: continue
                if sset.get(simplex , 0) == 0:
                    ch = True
                    sset[simplex] = 1
            if ch:
                res[s] +=1

    return res
    


@njit
def disparitys(facets, simps, NearestNeighbors = None):
    """Measure disparity of a given facet distribution.
    it is squared sum of k/s(strength) of each node.

    Parameters
    ----------
    facet_dict : dict
        (node : facets) pair distribution

    Returns
    -------
    np.ndarray
        (node : disparity1, disparity2) array

    some detail explanation. 
    """    
    ksimps = dims(simps)
    if NearestNeighbors is None:
        NearestNeighbors = Nnn(facets, simps)
    disp = -np.ones((len(facets),2), dtype=np.float64)
    for node in facets:
        s = 0 #sum of s
        ss = 0 #squared sum of s
        for sim in facets[node]:
            k= ksimps[sim-1]
            s += k
            ss += k*k
        if s == 0:
            disp[node-1] = 0
            continue
        
        disp[node-1,0]=ss/s/s                        # disparity1 $Y_1 = \sum (\frac{s}{\sum s})^2 = (\sum s^2)/(\sum s)^2$
        disp[node-1,1]=ss/NearestNeighbors[node-1]**2  # disparity2 $Y_2 = \sum (\frac{s}{ |\{nn\}| })^2 = (\sum s^2)/(|\{nn\}|)^2$
    return disp

@njit(parallel=True)
def local_disparitys(facets, simps, NearestNeighbors):
    """Measure disparity of a given facet distribution.
    it is squared sum of k/s(strength) of each node.

    Parameters
    ----------
    facet_dict : dict
        (node : facets) pair distribution

    Returns
    -------
    np.ndarray
        (node : disparity1, disparity2) array

    some detail explanation. 
    """    
    s : float
    ss : float
    ksimps = dims(simps)
    disp = -np.ones((len(simps),2), dtype=np.float64)
    for i in prange(len(simps)):
        s = 0. #sum of s
        ss = 0. #squared sum of s
        for node in simps[i+1]:
            for sim in facets[node]:
                k = ksimps[sim-1]
                s += k
                ss += k*k
        s -= ksimps[i]*ksimps[i] + ksimps[i]
        ss -= (ksimps[i]*ksimps[i] + ksimps[i])*ksimps[i]
        if s == 0:
            disp[i] = 0.
            continue
        
        disp[i,0] = ss/s/s                        # disparity1 $Y_1 = \sum (\frac{s}{\sum s})^2 = (\sum s^2)/(\sum s)^2$
        disp[i,1] = ss/(NearestNeighbors[i]**2)   # disparity2 $Y_2 = \sum (\frac{s}{ |\{nn\}| })^2 = (\sum s^2)/(|\{nn\}|)^2$
    return disp



def save_Scomplex(path, facets = None, simps = None):
    """Saving algorithm of simplicial complex. because of its sparsity,
    we only save pairs of (node id , simplex id).

    Parameters
    ----------
    path : 'Path like str'
        the path that the complex will be saved.
    facets : Dict, optional
        Dictionary type container which is consist of node id as a key and simplex id list as a value, by default None
    simps : Dict, optional
        Dictionary type container which is consist of simplex id as a key and node id list as a value, by default None

    Raises
    ------
    ValueError
        If facets and simps are None, raise Value Error
    """
    if facets is None and simps is None:
        raise ValueError("There is no data.")

    pair = []
    if facets is not None:
        for i in facets:
            for sid in facets[i]:
                pair.append((i, sid))
    else:
        for sid in simps:
            for i in simps[sid]:
                pair.append((i, sid))
    a = np.array(pair)
    np.sort(a, axis = 0)
    np.save(path, a)


@njit
def _load_alg(pairs):
    facets =  Dict.empty(int32, ListType(int32))
    simps =  Dict.empty(int32, ListType(int32))
    for pair in pairs:
        i = pair[0]
        sid = pair[1]
        face = facets.get(i, List.empty_list(int32))
        face.append(sid)
        facets[i] = face

        simp = simps.get(sid, List.empty_list(int32))
        simp.append(i)
        simps[sid] = simp
    return facets, simps

def load_Scomplex(path):
    return _load_alg(np.load(path))