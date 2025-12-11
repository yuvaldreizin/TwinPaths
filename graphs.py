import random
import os
import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle



"""
to open graphs in python from example_graphs folder :

import pickle
with open("example_graphs\<name_of_graph>.pkl", "rb") as f:
    graph = pickle.load(f)
"""


base_dir = os.path.dirname(__file__) 
graph_dir = os.path.join(base_dir, "test_graphs_txt_files")
test_graphs_dir = os.path.abspath(graph_dir)

def graph_init_from_txt(filename, plot = False):
    file = os.path.join(test_graphs_dir, filename)
    nodes = []
    node_flag = False
    coordinates = []
    links = []
    links_flag = False
    with open(file, 'r') as f:
        for line in f:
            #parsing nodes part of text doc
            if line.strip() == ")" and node_flag:
                node_flag = False  
            if node_flag:
                nodes.append(line.strip().split(" ")[0])
                coordinate = [float(line.strip().split(" ")[2]), float(line.strip().split(" ")[3])]
                coordinates.append(coordinate)
            if "NODES ("  in line: 
                node_flag = True

            #parsing links part of text doc
            if line.strip() == ")" and links_flag:
                links_flag = False  
            if links_flag: 
                start = line.find("(")
                end = line.find(")")
                if start != -1 and end != -1 and end > start:
                    content = line[start+1:end].strip()
                    links.append(content)    
            if "LINKS ("  in line: 
                links_flag = True

        #creating the graph from the parsed information
        graph = nx.Graph()
        for i in range(len(nodes)):
            graph.add_node(i + 1)
        for connection_num in range(len(links)):
            connection = links[connection_num].split(" ")
            #print(connection)
            for i in range(len(nodes)):
                if nodes[i] == connection[0]:
                    start = i + 1
                if nodes[i] == connection[1]:
                    end = i + 1
            start_point = coordinates[start - 1]
            end_point = coordinates[end - 1]
            distance = math.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)
            #print(f"connecting from {start} to {end}, distance ( acting as the weight) is: {distance:.2f}")
            graph.add_edge(start, end, weight = distance)
    
    # plotting graph
    if plot:
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray")
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)
        plt.show()

    with open(f"example_graphs/{filename.split(".")[0]}.pkl", "wb") as f:
        pickle.dump(graph, f)
        print(f"{filename.split(".")[0]}.pkl saved!")



for filename in os.listdir(test_graphs_dir):
    if filename.endswith(".txt"):
        graph_init_from_txt(filename)
