import os
import networkx as nx
import numpy as np
from utils.visualize_pyvis import visualize_dpgc_pyvis
from utils.visualize import visualize_graph
import time

"""
This module contains functions to find the minimum spanning tree (MST) connecting two nodes in a graph using two edge-disjoint paths and additional connections.
to use this module, call the mst_k2_generator function with a networkx graph and the start and end nodes. example: using file dfn-gwin.pkl, start node 1, end node 5

file = "dfn-gwin.pkl"
start_node = 1 
end_node = 5
with open(file, 'rb') as f:
    graph = pickle.load(f)
    mst, total_weight = mst_k2_generator(graph, start_node, end_node)

will return the mst connections, the total weight of the mst and the calculation time.
"""

def are_paths_independant(path1, path2, start_node, end_node):
    included_nodes = [start_node, end_node]
    for step1 in range(1, len(path1)):
        included_nodes.append(path1[step1][0])

    #checking if the nodes from path2 are already in the included list          
    for step2 in range(1, len(path2)):
        if included_nodes.__contains__(path2[step2][0]):
            return False
    return True

def find_paths_aux(graph, current, end, visited, steps, all_paths):
    if current == end:
        all_paths.append(steps.copy())
        return
    
    n = np.size(graph, 0)
    for nxt in range(n):
        if graph[current][nxt] != 0 and nxt not in visited:
            # take step
            visited.add(nxt)
            steps.append([current, nxt, float(graph[current][nxt])])
            
            find_paths_aux(graph, nxt, end, visited, steps, all_paths)
            
            # backtrack
            steps.pop()
            visited.remove(nxt)

def mst_k2_generator(graph, start_node, end_node):
    filepath = None
    time_start = time.time()
    arr = nx.to_numpy_array(graph, weight = 'weight')
    # make sure index order does not matter
    if start_node > end_node:
        start_node, end_node = end_node, start_node

    # converting start_node and end_node to the coresponding values in the nodes index array in the graph
    for i in range(len(list(graph.nodes()))):
        if list(graph.nodes())[i] == start_node:
            start_node = i
        if list(graph.nodes())[i] == end_node:
            end_node = i
    #print(f"start: {start_node}, end: {end_node}")
    #the values wil be converted back at the end (right before returning the mst and total weight)

    all_paths = []
    visited = {start_node}
    steps = []
    
    find_paths_aux(arr, start_node, end_node, visited, steps, all_paths)
    
    # finding all the independant paths between start_node and end_node and connecting unincluded nodes in the "cheapest way possbile"
    mst = []
    final_weights = 0.0
    for i in range(len(all_paths)):
        for j in range(len(all_paths)):
            if(are_paths_independant(all_paths[i], all_paths[j], start_node, end_node)):
                # print(f"first path: {all_paths[i]}\nsecond path: {all_paths[j]}")
                added_connections = []
                included_nodes = [end_node]
                total_weights = 0.0

                # adding weights and nodes from the two independant paths
                for step in range(len(all_paths[i])):
                    total_weights += all_paths[i][step][2]
                    included_nodes.append(all_paths[i][step][0])
                for step in range(len(all_paths[j])):
                    total_weights += all_paths[j][step][2]
                    if not included_nodes.__contains__(all_paths[j][step][0]):
                        included_nodes.append(all_paths[j][step][0])
                    if not included_nodes.__contains__(all_paths[j][step][1]):
                        included_nodes.append(all_paths[j][step][1])
                #print(f"nodes in paths: {included_nodes}, total weight is {total_weights}")

                # checking which nodes arent included. and finding the "cheapest" node to connect them to based on the graph
                unincluded_nodes = []
                for node in range(len(arr)):
                    if not included_nodes.__contains__(node):
                        unincluded_nodes.append(node)
                #print(f"unincluded: {unincluded_nodes}")

                # connecting the unconnected nodes
                while unincluded_nodes != []:
                    cheapest_connection_weight = 0.0
                    cheapest_connection = []
                    for node in unincluded_nodes:  
                        for connection_node in included_nodes:
                            if arr[node][connection_node] != 0:
                                if cheapest_connection == [] or cheapest_connection_weight > arr[node][connection_node]:
                                    cheapest_connection = [node, connection_node]
                                    cheapest_connection_weight = arr[node][connection_node]
                                    #print(f"current best option is to connect {cheapest_connection} and the weight is {cheapest_connection_weight}")

                    if not cheapest_connection_weight == 0.0:
                        new_connection = [cheapest_connection[0], cheapest_connection[1], float(cheapest_connection_weight)]
                        #print(f"new connection is : {new_connection}")
                        included_nodes.append(cheapest_connection[0])
                        unincluded_nodes.remove(cheapest_connection[0])
                        total_weights += cheapest_connection_weight
                        added_connections.append(new_connection)
                
                #print(f"final weight is {total_weights}")
                if final_weights == 0.0 or final_weights > total_weights:
                    final_weights = total_weights

                    #converting the values back to the cooresponding ones from graph based on nodes list in graph
                    #converting the mst indexes back to there coresponding values from the graph
                    dual_paths = []
                    recovered_edges = []
                    for k in range(len(all_paths[i])):
                        dual_paths.append([list(graph.nodes())[all_paths[i][k][0]], list(graph.nodes())[all_paths[i][k][1]]])
                    for k in range(len(all_paths[j])):
                        dual_paths.append([list(graph.nodes())[all_paths[j][k][0]], list(graph.nodes())[all_paths[j][k][1]]])
                    
                    for k in range(len(added_connections)):
                        recovered_edges.append([list(graph.nodes())[added_connections[k][0]], list(graph.nodes())[added_connections[k][1]]])
                   

                    dual_paths = {tuple(e) for e in dual_paths}
                    mst_edges = {tuple(e) for e in recovered_edges}
                    #print(f"dual paths are: {dual_paths}")
                    #print(f"recovered edges are: {recovered_edges}")
                    
                    filepath = visualize_dpgc_pyvis(
                        graph,
                        dual_path_edges=dual_paths,
                        mst_edges=mst_edges,
                        filename_prefix="testing_mst_k2",
                        results_dir="results"
                    )
                         
                    mst = []
                    mst.extend(all_paths[i])
                    mst.extend(all_paths[j])
                    mst.extend(added_connections)   
                    #print(f"current best total weight is {final_weights}\nmst connections:\n {mst}\n")
                    
    #converting the mst indexes back to there coresponding values from the graph
    for k in range(len(mst)):
        mst[k][0] = list(graph.nodes())[mst[k][0]]
        mst[k][1] = list(graph.nodes())[mst[k][1]]

    if filepath is not None:
        print("graph saved as:", filepath)
        print(f"minimum spanning tree connections are [node1, node2 , weight of edge]:\n {mst} \nwith total weight of: {final_weights}")
        time_end = time.time()
        print(f"calculation time: {time_end - time_start} seconds")
        return mst, final_weights
    else:
        print("no mst found")
        return [], 0.0


filename = "dfn-gwin.pkl"
base_dir = os.path.dirname(__file__) 
graph_dir = os.path.join(base_dir, "example_graphs")
test_graphs_dir = os.path.abspath(graph_dir)
file = os.path.join(test_graphs_dir, filename)
graph = nx.Graph()


"""
the minimum spanning tree connections are: ( for dfn-gwin.pkl, start node 1, end node 5)
 [[1, 2, 1.211615411300131], [2, 9, 2.4721839546441577], [9, 5, 1.671324127750211], [1, 8, 3.6426894736718913], [8, 5, 0.7185236689907966], [3, 9, 1.423060747684364], [6, 3, 2.1019236409774735], [4, 6, 1.459189268052645], [0, 6, 2.223989285046129], [7, 0, 1.4440425374621095], [10, 2, 2.7947887151804505]]

"""

if __name__ == "__main__":

    # Example graph construction for testing
    graph = nx.Graph()
    edges = [
            # Core block around s=1 and t=14
            (1, 2, 1),
            (2, 3, 2),
            (3, 14, 3),
            (1, 4, 2),
            (4, 5, 2),
            (5, 14, 2),

            # Extra alternate cheap path
            (1, 6, 1),
            (6, 7, 1),
            (7, 14, 4),

            # A diamond cycle in the middle
            (3, 8, 2),
            (8, 9, 2),
            (9, 14, 2),
            (8, 5, 3),

            # A long misleading "tempting" but expensive path
            (1, 10, 5),
            (10, 11, 8),
            (11, 12, 8),
            (12, 13, 3),
            (13, 14, 6),

            # Short cross-links to create competing MST choices
            (2, 8, 3),
            (6, 9, 2),
            (7, 5, 5),
    ]
    for u, v, w in edges:
            graph.add_edge(u, v, weight=w)

    # finding the k=2 shortest path by checking every possible combination
    mst, total_weight = mst_k2_generator(graph, 1, 14)
    
    
        