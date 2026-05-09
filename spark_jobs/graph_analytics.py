"""
Task 12: Build Road Network Analytics (NetworkX Version)
======================================
Why: Convert the road network topology (207 sensors) into a graph structure
to identify critical nodes (PageRank) and compute efficient routing paths.
"""

import os
import sys
import pickle
import numpy as np
import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, MapType, IntegerType, FloatType
from dotenv import load_dotenv

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

load_dotenv()

print("[GraphAnalytics] Initializing Spark Session...")

spark = SparkSession.builder \
    .appName("TrafficGraphAnalytics") \
    .master("local[4]") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g")) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("[GraphAnalytics] Loading adjacency matrix...")
try:
    # Try loading the standard NumPy array first
    adj_matrix = np.load('data/metr-la/adj_mat.npy')
except FileNotFoundError:
    # Fallback to unpacking the pickle file tuple
    with open('data/metr-la/adj_METR-LA.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        adj_matrix = data[2]  # The 3rd item is the actual matrix

print("[GraphAnalytics] Building graph with NetworkX...")
G = nx.DiGraph()

n_sensors = adj_matrix.shape[0]
for i in range(n_sensors):
    G.add_node(i, sensor_id=i)

print(f"[GraphAnalytics] Added {n_sensors} vertices")

print("[GraphAnalytics] Adding edges...")
edge_count = 0
for i in range(n_sensors):
    for j in range(n_sensors):
        weight = float(adj_matrix[i][j])
        if i != j and weight > 0:
            G.add_edge(i, j, weight=weight)
            edge_count += 1

print(f"[GraphAnalytics] Added {edge_count} edges")

print("[GraphAnalytics] Computing PageRank...")
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
print("[GraphAnalytics] Top 10 most critical nodes (by PageRank):")
for node_id, score in top_nodes:
    print(f"  Sensor {node_id}: {score:.6f}")

print("[GraphAnalytics] Computing shortest paths to landmark sensors...")
landmarks = [0, 50, 100]

sp_data = []
for node in G.nodes():
    distances_map = {}
    for target in landmarks:
        try:
            dist = nx.shortest_path_length(G, source=node, target=target)
            distances_map[str(target)] = int(dist)
        except nx.NetworkXNoPath:
            pass 
    
    sp_data.append((str(node), distances_map))

print("[GraphAnalytics] Saving PageRank to Delta Lake...")
pagerank_data = [(str(node_id), float(score)) for node_id, score in pagerank.items()]
pr_schema = StructType([
    StructField("id", StringType(), True),
    StructField("pagerank", FloatType(), True)
])
pagerank_df = spark.createDataFrame(pagerank_data, pr_schema)
pagerank_df.write.format("delta").mode("overwrite").save("delta_tables/graph_pagerank")
print(f"[GraphAnalytics] ✅ Saved {len(pagerank_data)} records to delta_tables/graph_pagerank")

print("[GraphAnalytics] Saving shortest paths to Delta Lake...")
sp_schema = StructType([
    StructField("id", StringType(), True),
    StructField("distances", MapType(StringType(), IntegerType()), True)
])
sp_df = spark.createDataFrame(sp_data, sp_schema)
sp_df.write.format("delta").mode("overwrite").save("delta_tables/graph_shortest_paths")
print(f"[GraphAnalytics] ✅ Saved {len(sp_data)} records to delta_tables/graph_shortest_paths")

print("\n[GraphAnalytics] Summary Statistics:")
print(f"  • Nodes: {G.number_of_nodes()}")
print(f"  • Edges: {G.number_of_edges()}")
print(f"  • Density: {nx.density(G):.4f}")
avg_degree = sum(dict(G.out_degree()).values()) / max(1, G.number_of_nodes())
print(f"  • Avg out-degree: {avg_degree:.2f}")

spark.stop()
print("\n[GraphAnalytics] ✅ Task 12 Complete!")