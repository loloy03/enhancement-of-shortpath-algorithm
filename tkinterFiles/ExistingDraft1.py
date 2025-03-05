import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import networkx as nx
import random
import time
import tracemalloc
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize K shortest path function
def k_shortest_paths(G, source, target, k, weight):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )

class ShortestPathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shortest Path Finder")
        self.root.geometry("1200x700")
        
        # # Add this line
        # self.num_executions = 100
        
        # Load data
        self.csv_file = 'test_data1.csv'
        self.df = pd.read_csv(self.csv_file)
        self.nodes = sorted(list(set(self.df['Source Node']).union(set(self.df['Destination Node']))))
        
        # Create NetworkX graph
        self.G = nx.from_pandas_edgelist(
            self.df, 
            source='Source Node', 
            target='Destination Node', 
            edge_attr=['Actual Length', 'Accessibility Level', 'Is Ramp?'],
            create_using=nx.DiGraph()
        )
        
        # Cache node positions for consistent graph drawing
        self.pos = nx.spring_layout(self.G, seed=42)
        
        # Calculate average edge length (cached)
        self.avgEdgeLength = self.calculate_avg_edge_length()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for input controls
        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.X, pady=10)
        
        # Source node selection
        ttk.Label(self.top_frame, text="Source Node:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(self.top_frame, textvariable=self.source_var, values=self.nodes, width=10)
        self.source_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Destination node selection
        ttk.Label(self.top_frame, text="Destination Node:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.dest_var = tk.StringVar()
        self.dest_combo = ttk.Combobox(self.top_frame, textvariable=self.dest_var, values=self.nodes, width=10)
        self.dest_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Calculate button
        self.calculate_btn = ttk.Button(self.top_frame, text="Calculate Shortest Paths", command=self.calculate_paths)
        self.calculate_btn.grid(row=0, column=4, padx=20, pady=5)
        
        # Random button
        self.random_btn = ttk.Button(self.top_frame, text="Random Source/Dest", command=self.select_random_nodes)
        self.random_btn.grid(row=0, column=5, padx=5, pady=5)
        
        # Create middle frame for results table
        self.middle_frame = ttk.LabelFrame(self.main_frame, text="Results")
        self.middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create treeview for results
        self.create_results_treeview()
        
        # Create bottom frame for graph visualization and info
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel for graph visualization
        self.graph_frame = ttk.LabelFrame(self.bottom_frame, text="Path Visualization")
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create right side frame to contain both statistics and details
        self.right_frame = ttk.Frame(self.bottom_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Statistics panel (top right)
        self.info_frame = ttk.LabelFrame(self.right_frame, text="Statistics")
        self.info_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Add stats text widget
        self.stats_text = scrolledtext.ScrolledText(self.info_frame, width=40, height=10, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Path details panel (bottom right)
        self.details_frame = ttk.LabelFrame(self.right_frame, text="Path Details")
        self.details_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Add path details text widget
        self.details_text = scrolledtext.ScrolledText(self.details_frame, width=40, height=10, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready. Please select source and destination nodes.")
        
        # Initialize figure for path visualization
        self.initialize_figure()
        
        # Display network statistics
        self.display_network_stats()
    
    def create_results_treeview(self):
        # Create a frame for the treeview and scrollbar
        tree_frame = ttk.Frame(self.middle_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ("path_id", "path", "actual_distance", "calculated_weight")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=5, yscrollcommand=tree_scroll.set)
        
        # Configure columns
        self.tree.heading("path_id", text="Path #")
        self.tree.heading("path", text="Path")
        self.tree.heading("actual_distance", text="Actual Distance")
        self.tree.heading("calculated_weight", text="Calculated Weight (Lower is Better)")
        
        self.tree.column("path_id", width=50, anchor=tk.CENTER)
        self.tree.column("path", width=400)
        self.tree.column("actual_distance", width=100, anchor=tk.CENTER)
        self.tree.column("calculated_weight", width=200, anchor=tk.CENTER)
        
        # Pack treeview
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        tree_scroll.config(command=self.tree.yview)
        
        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_path_select)
    
    def initialize_figure(self):
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Display the complete network graph initially
        self.display_full_graph()
    
    def display_full_graph(self):
        # Clear the current figure
        self.ax.clear()
        
        # Draw all nodes
        nx.draw_networkx_nodes(self.G, self.pos, node_color='skyblue', 
                               node_size=300, ax=self.ax)
        
        # Draw all edges
        nx.draw_networkx_edges(self.G, self.pos, edge_color='gray', width=1, 
                               arrows=True, arrowsize=10, ax=self.ax)
        
        # Add node labels for some of the nodes (to avoid cluttering)
        # Only label nodes with degree > 3 to reduce clutter
        important_nodes = [node for node in self.G.nodes() if self.G.degree(node) > 3]
        node_labels = {node: str(node) for node in important_nodes}
        nx.draw_networkx_labels(self.G, self.pos, labels=node_labels, font_size=8, ax=self.ax)
        
        # Add title
        self.ax.set_title("Complete Network Graph")
        self.ax.set_axis_off()
        
        # Redraw the canvas
        self.canvas.draw()
    
    def calculate_avg_edge_length(self):
        # Calculate average edge length
        avg_edge_length = 0
        edge_counter = 0
        
        for node in self.G.nodes:
            for edges in self.G.edges(node, data='Actual Length'):
                avg_edge_length += edges[2]
                edge_counter += 1
        
        return avg_edge_length / edge_counter if edge_counter > 0 else 0
    
    def select_random_nodes(self):
        # Select random source and destination
        source, dest = random.sample(self.nodes, 2)
        self.source_var.set(source)
        self.dest_var.set(dest)
        self.status_var.set(f"Randomly selected: Source={source}, Destination={dest}")
    
    def display_network_stats(self):
        # Display basic network statistics
        self.stats_text.delete(1.0, tk.END)
        
        stats = [
            "Network Statistics:",
            f"Number of nodes: {self.G.number_of_nodes()}",
            f"Number of edges: {self.G.number_of_edges()}",
            f"Average edge length: {self.avgEdgeLength:.2f}",
            f"Network density: {nx.density(self.G):.4f}",
            f"Is network connected: {nx.is_strongly_connected(self.G)}",
            f"Number of connected components: {nx.number_strongly_connected_components(self.G)}",
            "\nSelect source and destination nodes",
            "to calculate shortest paths."
        ]
        
        self.stats_text.insert(tk.END, "\n".join(stats))
    
    def calculate_paths(self):
        source = self.source_var.get()
        destination = self.dest_var.get()
        
        # Validate input
        if not source or not destination:
            messagebox.showwarning("Input Error", "Please select both source and destination nodes.")
            return
        
        try:
            source = int(source)
            destination = int(destination)
        except ValueError:
            messagebox.showwarning("Input Error", "Source and destination must be valid node numbers.")
            return
        
        # Check if source and destination exist in graph
        if source not in self.G.nodes or destination not in self.G.nodes:
            messagebox.showwarning("Input Error", "Selected nodes do not exist in the graph.")
            return
        
        # Check if path exists
        if not nx.has_path(self.G, source, destination):
            messagebox.showinfo("No Path", "No path exists between the selected nodes.")
            return
        
        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Start timing and memory tracking
        start_time = time.perf_counter()
        tracemalloc.start()
        
        self.status_var.set("Calculating paths...")
        self.root.update()
        
        try:
            # Find K shortest paths
            p = list(k_shortest_paths(self.G, source, destination, 10, 'Actual Length'))
            
            # Calculate average path length
            avgPathLength = 0
            for i in range(len(p)-1):
                avgPathLength += nx.path_weight(self.G, p[i], weight='Actual Length')
            
            avgPathLength = avgPathLength / 10 if len(p) > 0 else 0
            
            # Calculate threshold
            apLengthThreshold = self.avgEdgeLength + avgPathLength
            
            # Filter paths based on threshold
            path_distance_list = []
            for i in range(len(p)-1, -1, -1):
                path_distance = nx.path_weight(self.G, p[i], weight='Actual Length')
                if path_distance > apLengthThreshold:
                    p.pop(i)
                    continue
                path_distance_list.append(path_distance)
            
            # Compute penalties
            rp_list = []
            for path in p:
                rp = 0
                for i in range(len(path)-1):
                    edge_data = self.df[(self.df['Source Node'] == path[i]) & 
                                      (self.df['Destination Node'] == path[i+1])]
                    
                    access_level = edge_data['Accessibility Level'].values[0]
                    edge_distance = self.G[path[i]][path[i+1]].get('Actual Length')
                    ramp_not_present = edge_data['Is Ramp?'].values[0]
                    
                    rp += edge_distance * access_level + ramp_not_present * self.avgEdgeLength
                
                rp_list.append(rp)
            
            # Create zipped list and sort by penalty
            zipped_list = list(zip(p, path_distance_list, rp_list))
            sorted_list = sorted(zipped_list, key=lambda x: x[2])
            
            # Populate treeview with results
            for i, (path, distance, weight) in enumerate(sorted_list, start=1):
                path_str = " → ".join(map(str, path))
                self.tree.insert("", "end", values=(i, path_str, f"{distance:.2f}", f"{weight:.2f}"))
            
            # Update statistics
            self.update_statistics(source, destination, sorted_list, start_time)
            
            # If paths were found, visualize the best one
            if sorted_list:
                best_path = sorted_list[0][0]
                self.visualize_path(best_path)
            
            self.status_var.set(f"Found {len(sorted_list)} paths from {source} to {destination}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred during calculation.")
        finally:
            # Stop memory tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    
    def update_statistics(self, source, destination, sorted_list, start_time):
        # Calculate statistics
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        
        # Clear previous stats
        self.stats_text.delete(1.0, tk.END)
        
        # Add new stats
        stats = [
            f"Source: {source}, Destination: {destination}",
            f"Number of paths found: {len(sorted_list)}",
            f"Execution time: {(end_time - start_time) * 1000:.2f} ms",
            f"Memory usage: {current / 10**3:.2f} KB",
            f"Peak memory usage: {peak / 10**3:.2f} KB",
            f"Average edge length: {self.avgEdgeLength:.2f}",
            "\nPath statistics:",
        ]
        
        if sorted_list:
            best_path = sorted_list[0]
            stats.extend([
                f"Best path: {' → '.join(map(str, best_path[0]))}",
                f"Best path distance: {best_path[1]:.2f}",
                f"Best path weight: {best_path[2]:.2f}",
                "\nAccessibility info:",
                "Lower weight = More accessible path",
                "Factors considered: Path length, accessibility level, ramp availability",
                "\nClick on a path in the table above to see detailed edge information."
            ])
        
        self.stats_text.insert(tk.END, "\n".join(stats))
    
    def on_path_select(self, event):
        selected_items = self.tree.selection()
        if not selected_items:
            return
        
        item = self.tree.item(selected_items[0])
        path_str = item['values'][1]
        path = [int(node) for node in path_str.split(" → ")]
        
        self.visualize_path(path)
        self.display_path_details(path)

    def display_path_details(self, path):
        # Clear previous details
        self.details_text.delete(1.0, tk.END)
        
        # Add path overview
        self.details_text.insert(tk.END, f"Selected Path: {' → '.join(map(str, path))}\n\n")
        
        # Display information for each edge in the path
        for i in range(len(path)-1):
            source = path[i]
            dest = path[i+1]
            
            # Get edge data from dataframe
            edge_data = self.df[(self.df['Source Node'] == source) & 
                            (self.df['Destination Node'] == dest)]
            
            if not edge_data.empty:
                edge_info = [
                    f"Edge: {source} → {dest}",
                    f"Actual Length: {edge_data['Actual Length'].values[0]}",
                    f"Road Width: {edge_data['road_width'].values[0]}",
                    f"Smoothness: {edge_data['smoothness'].values[0]}",
                    f"Surface: {edge_data['surface'].values[0]}",
                    f"Slope: {edge_data['slope'].values[0]}",
                    f"Bike Lane: {'Yes' if edge_data['bike_lane'].values[0] == 1 else 'No'}",
                    f"Accessibility Level: {edge_data['Accessibility Level'].values[0]}",
                    f"Has Ramp: {'Yes' if edge_data['Is Ramp?'].values[0] == 1 else 'No'}",
                    "----------------------------"
                ]
                self.details_text.insert(tk.END, "\n".join(edge_info) + "\n\n")

    
    def visualize_path(self, path):
        # Clear the current figure
        self.ax.clear()
        
        # Create a subgraph with only the nodes in the path
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Draw all nodes with light color
        nx.draw_networkx_nodes(self.G, self.pos, node_color='lightgray', 
                              node_size=300, ax=self.ax)
        
        # Draw path nodes with different color
        path_nodes = set(path)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=path_nodes, 
                              node_color='lightblue', node_size=400, ax=self.ax)
        
        # Highlight source and destination
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[path[0]], 
                              node_color='green', node_size=500, ax=self.ax)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[path[-1]], 
                              node_color='red', node_size=500, ax=self.ax)
        
        # Draw all edges with light color
        nx.draw_networkx_edges(self.G, self.pos, edge_color='lightgray', width=1, 
                              arrows=True, arrowsize=15, ax=self.ax)
        
        # Draw path edges with different color
        nx.draw_networkx_edges(self.G, self.pos, edgelist=path_edges, 
                              edge_color='blue', width=2, arrows=True, 
                              arrowsize=20, ax=self.ax)
        
        # Add node labels
        nx.draw_networkx_labels(self.G, self.pos, font_size=10, ax=self.ax)
        
        # Add title
        self.ax.set_title(f"Path: {' → '.join(map(str, path))}")
        self.ax.set_axis_off()
        
        # Redraw the canvas
        self.canvas.draw()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create and run the application
    root = tk.Tk()
    app = ShortestPathApp(root)
    root.mainloop()