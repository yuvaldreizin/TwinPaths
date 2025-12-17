from ground_truth_calculation import mst_k2_generator as mst_calculator
import pickle 
import os  

# Example usagefile = "abilene.pkl"
filename = "abilene.pkl"
start_node = 5
end_node = 7

# getting the directory of the current file
base_dir = os.path.dirname(__file__) 
graph_dir = os.path.join(base_dir, "example_graphs")
test_graphs_dir = os.path.abspath(graph_dir)
file = os.path.join(test_graphs_dir, filename)

# Load the graph from the pickle file and calculate the MST using the mst_k2_generator function
with open(file, 'rb') as f:
    graph = pickle.load(f)
    mst, total_weight = mst_calculator(graph, start_node, end_node)