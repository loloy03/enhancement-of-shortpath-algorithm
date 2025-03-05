import networkx as nx
import pandas as pd
import heapq
from itertools import islice

import tracemalloc
import time
import random

import pandas as pd

# Initialize K shortest path function
def k_shortest_paths(G, source, target, k, weight):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )

def get_shortest_paths(selected_csv, selected_source, selected_destination, selected_algo):
    df = pd.read_csv(selected_csv)

    # Line 2 
    G = nx.from_pandas_edgelist(df, source='Source Node', target='Destination Node', edge_attr='Actual Length', create_using=nx.DiGraph())

    # Line 3 to 6
    avgEdgeLength = 0
    edgeCounter = 0
    source = selected_source
    destination = selected_destination 

    # Line 7 to 12
    for node in G.nodes:
        for edges in G.edges(node, data='Actual Length'):
            avgEdgeLength += edges[2]
            edgeCounter += 1
    # Line 13
    avgEdgeLength = avgEdgeLength/edgeCounter

    # Line 14
    # p = list(nx.all_simple_paths(G, source, destination, cutoff=9))

    # Yen's k shortest path
    p = list(k_shortest_paths(G, source, destination, 10,'Actual Length'))


    # Line 15
    avgPathLength = 0

    # Line 16 to 18
    for i in range(len(p)-1):
        avgPathLength += nx.path_weight(G, p[i], weight='Actual Length')
    # Line 19
    avgPathLength = avgPathLength/10  

    # Line 20
    apLengthThreshold = avgEdgeLength + avgPathLength

    edge_data_dict = df.set_index(['Source Node', 'Destination Node']).to_dict(orient="index")

    if selected_algo == "existing":
        calculate_existing_algo(df, G, p, apLengthThreshold, avgEdgeLength, edge_data_dict)
    elif selected_algo == "propose":
        calculate_propose_algo(df, G, p, apLengthThreshold, avgEdgeLength, edge_data_dict)

def calculate_existing_algo(df, G, p, apLengthThreshold, avgEdgeLength, edge_data_dict):
    # Line 21 to 25
    path_distance_list = []
    for i in range(len(p)-1, -1, -1):  # Iterate in reverse without extra copy
        path_distance = nx.path_weight(G, p[i], weight='Actual Length')
        if (path_distance > apLengthThreshold):
            p.pop(i)
            continue
        path_distance_list.append(path_distance) 


    # Line 26 to 29
    # Compute penalty
    rp_list = []
    for path in p:
        access_level = 0
        rp = 0
        for i in range(len(path)-1):
            access_level = df[(df['Source Node'] == path[i]) & (df['Destination Node'] == path[i+1])]['Accessibility Level'].values[0]
            edge_distance = G[path[i]][path[i+1]].get('Actual Length')
            ramp_not_present = df[(df['Source Node'] == path[i]) & (df['Destination Node'] == path[i+1])]['Is Ramp?'].values[0]
            rp += edge_distance * access_level + ramp_not_present * avgEdgeLength
        rp_list.append(rp)

    zipped_list = list(zip(p, path_distance_list, rp_list))

    # Line 30
    sorted_list = sorted(zipped_list, key=lambda x: x[2])

    path_df = pd.DataFrame(sorted_list, columns = ['Alternative Paths', 'Actual Distance', 'Calculated Weight'])
    print(path_df)
    return path_df

def calculate_propose_algo(df, G, p, apLengthThreshold, avgEdgeLength, edge_data_dict):
    # Initialize lists
    filtered_paths = []
    heap = []  # Min-heap for sorting based on 'Calculated Weight'
    for path in p:
        path_distance = nx.path_weight(G, path, weight='Actual Length')
        if path_distance > apLengthThreshold:
            continue  # Skip paths that exceed the threshold

        # Compute penalty
        rp = 0
        for u, v in zip(path, path[1:]):
            edge_data = edge_data_dict[(u, v)]
            access_level = 5
            if edge_data['road_width'] < 3:
                access_level -= 1
            if edge_data['smoothness'] not in ["excellent", "good"]:
                access_level -= 1
            if edge_data['surface'] not in ["asphalt", "concrete"]:
                access_level -= 1
            if edge_data['slope'] not in ["flat", "gentle"]:
                access_level -= 1

            rp += G[u][v]['Actual Length'] * access_level + edge_data['bike_lane'] * avgEdgeLength

        # Push to min-heap
        heapq.heappush(heap, (rp, path_distance, path))  

    # Convert heap to DataFrame 
    
    # Create DataFrame
    path_df = pd.DataFrame(heap, columns=['Calculated Weight', 'Actual Distance', 'Alternative Paths'])
    print(path_df[['Alternative Paths', 'Actual Distance', 'Calculated Weight']])
    return path_df

selected_csv = 'test_data1.csv'
num_executions = 1

# Set random seed for reproducibility
random.seed(42)
df = pd.read_csv(selected_csv)
nodes = set(df['Source Node']).union(set(df['Destination Node']))  # Collect all unique nodes
nodes = list(nodes)  # Convert to list for random selection
# Generate all source-destination pairs before the loop
source_dest_pairs = [random.sample(nodes, 2) for _ in range(num_executions)]


# Run for "existing" mode
start_time = time.perf_counter()
tracemalloc.start()
current, peak = tracemalloc.get_traced_memory()
print(f"Initial memory usage: {current / 10**3:.2f} KB")

print("Starting executions for 'existing' mode...\n")
for i, (selected_source, selected_destination) in enumerate(source_dest_pairs, start=1):
    print(f"Execution {i}: Mode=existing, Source={selected_source}, Destination={selected_destination}")
    get_shortest_paths(selected_csv, selected_source, selected_destination, "existing")

end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
print(f"Memory usage after execution: {current / 10**3:.2f} KB")
print(f"Peak memory usage: {peak / 10**3:.2f} KB")
print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
tracemalloc.stop()

# Run for "propose" mode
start_time = time.perf_counter()
tracemalloc.start()
current, peak = tracemalloc.get_traced_memory()
print(f"Initial memory usage: {current / 10**3:.2f} KB")

print("Starting executions for 'propose' mode...\n")
for i, (selected_source, selected_destination) in enumerate(source_dest_pairs, start=1):
    print(f"Execution {i}: Mode=propose, Source={selected_source}, Destination={selected_destination}")
    get_shortest_paths(selected_csv, selected_source, selected_destination, "propose")

end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
print(f"Memory usage after execution: {current / 10**3:.2f} KB")
print(f"Peak memory usage: {peak / 10**3:.2f} KB")
print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
tracemalloc.stop()

print("-------------------------------------------\n")