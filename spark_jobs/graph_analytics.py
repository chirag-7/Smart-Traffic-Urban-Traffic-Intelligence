"""
Road-network analytics on the METR-LA sensor graph.

Loads ``data/metr-la/adj_mat.npy`` (or pickle fallback), builds a directed graph in NetworkX,
computes PageRank and multi-target shortest-path distances to landmarks (0, 50, 100),
and writes results to Delta Lake for use by the streaming rerouting logic.

Vertex ids are string indices ``0 .. n-1`` and must match ``sensor_index`` emitted by
``producers/sensor_producer.py``.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, IntegerType, MapType, StringType, StructField, StructType

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

logger.info("Starting Spark session...")
spark = (
    SparkSession.builder.appName("TrafficGraphAnalytics")
    .master("local[4]")
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    adj_matrix = np.load("data/metr-la/adj_mat.npy")
except FileNotFoundError:
    with open("data/metr-la/adj_METR-LA.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")
        adj_matrix = data[2]

n_sensors = adj_matrix.shape[0]
G = nx.DiGraph()
for i in range(n_sensors):
    G.add_node(i, sensor_id=i)

edge_count = 0
for i in range(n_sensors):
    for j in range(n_sensors):
        w = float(adj_matrix[i][j])
        if i != j and w > 0:
            G.add_edge(i, j, weight=w)
            edge_count += 1

logger.info("Graph: %s nodes, %s edges", G.number_of_nodes(), edge_count)

pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
logger.info("Top PageRank nodes: %s", [(nid, round(sc, 6)) for nid, sc in top_nodes])

landmarks = [0, 50, 100]
sp_data = []
for node in G.nodes():
    distances_map = {}
    for target in landmarks:
        try:
            distances_map[str(target)] = int(nx.shortest_path_length(G, source=node, target=target))
        except nx.NetworkXNoPath:
            pass
    sp_data.append((str(node), distances_map))

pagerank_data = [(str(node_id), float(score)) for node_id, score in pagerank.items()]
pr_schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("pagerank", FloatType(), True),
    ]
)
pagerank_df = spark.createDataFrame(pagerank_data, pr_schema)
pagerank_df.write.format("delta").mode("overwrite").save("delta_tables/graph_pagerank")

sp_schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("distances", MapType(StringType(), IntegerType()), True),
    ]
)
sp_df = spark.createDataFrame(sp_data, sp_schema)
sp_df.write.format("delta").mode("overwrite").save("delta_tables/graph_shortest_paths")

avg_degree = sum(dict(G.out_degree()).values()) / max(1, G.number_of_nodes())
logger.info(
    "Wrote Delta: graph_pagerank (%s rows), graph_shortest_paths (%s rows); density=%.4f avg_out_deg=%.2f",
    len(pagerank_data),
    len(sp_data),
    nx.density(G),
    avg_degree,
)

spark.stop()
