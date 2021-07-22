import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



def edge_weight_percolation(network : nx.Graph, weight_name : str, connecting = 'strong') -> np.ndarray:
    '''Percolation with given weight on given network.
    Parameters
    ---------------
    network : `nx.Graph` (or `nx.DiGraph`)
        The network which will be percolated.
    weight_name : `str`
        The string name of the weight to be used.

    Return
    ---------------
    `np.ndarray` or `list`
        From each node as each cluster, the return is a merging ordered list of edges and its cluster.
        The form of each element is ('weight', startnode, endnode, cluster1, cluster2).
        If cluster is negative, it means there is no merging of clusters between such a edge.
    
    Examples
    -----------
    ```!python
    >>> network = nx.Graph()
    >>> network.add_edge(1,2, 'value' = 0.2 ) 
    >>> network.add_edge(2,3, 'value' = 0.5 ) 
    >>> network.add_edge(3,4, 'value' = 0.3 ) 
    >>> 
    >>> edge_weight_percolation(network, 'value')   
    ((0.2, 1, 2, 1, 2)                              
    (0.3, 3, 4, 3, 4)                               
    (0.5, 2, 3, 1, 3))                              
    ```
    First state : 1, 2, 3, 4 (each node is a single cluster).
    Edge (1,2) was merged. Current state : 1, 1, 3, 4 (cluster 1 contains node 1 and 2.).
    Edge (3,4) was merged. Current state : 1, 1, 3, 3 (cluster 3 contains node 3 and 4.).
    Finally, edge (2,3) was merged. Current state : 1, 1, 1, 1 (cluster 1 contains all nodes.).
    '''

    strong = True if connecting == 'strong' else False

    if not isinstance(network, nx.DiGraph) and not isinstance(network, nx.Graph):
        raise ValueError("'network' must be a instance of networkx `DiGraph` or `Graph`.")
    
    if strong and not isinstance(network, nx.DiGraph):
        strong = False
        raise Warning("Undirected graph cannot be operated on strongly connected component. Weakly connected component will be calculated.")

    edges = np.array([[wt,u,v] for (u,v, wt) in network.edges.data(weight_name)])   # container for sorting
    order = np.argsort(edges[:,0])                                                  # sort by weight

    class cluster:
        def __init__(self, data):
            self.head = None    # the representative element of cluster
            self.data = data    # the representative data of cluster

        def leader(self):
            current = self
            while current.head is not None:
                #print(current.data)
                current = current.head
            return current
        
        def merge(self, other):
            target, cl = (self, other) if self>other else (other, self)
            target.leader().head = cl.leader()
            
        def __gt__(self, other):
            return self.leader().data > other.leader().data # for deciding which is more representative

    #make empty Graph
    percolation = []
    percolnet = network.__class__()
    clusters = {i:cluster(i) for i in range(len(network.nodes))} # initial cluster state is that each node is a cluster itself.
    for o in order:
        weight, st_node, end_node = edges[o]                # weight, start node, end node, respectively.
        cl1 , cl2 = clusters[st_node], clusters[end_node]   # cl1 and cl2 are cluster1 and cluster2, respectively.

        percolnet.add_edge(st_node, end_node)               # Make percolation by adding new edge by edge.

        merged = False
        if strong:
            ## checking strongly connencted  #################################### YG code here, if merge occurs, then merged = True
            print("Hello, world")
            pass
        
        else:
            ## checking weakly connencted    #################################### YG code here, if merge occurs, then merged = True
            pass
        
        if merged and cl1 != cl2:
            percolation.append([weight, st_node, end_node, cl1.head.data, cl2.head.data])
            cl1.merge(cl2)
        else:
            percolation.append([weight, st_node, end_node, -1, -1])
    
    return percolation
    


