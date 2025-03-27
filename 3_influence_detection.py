import json
import networkx as nx
import pandas as pd
from collections import defaultdict
import os
import argparse # Import the argparse library

def calculate_participation_coefficient(G, nodes_data):
    """
    Calculates the participation coefficient for each node.
    Pi = 1 - sum((k_is / k_i)^2)
    where k_is is the number of links of node i to nodes in module s,
    and k_i is the total degree of node i.
    """
    participation = {}
    node_cluster = {node['id']: node['cluster'] for node in nodes_data}
    clusters = set(node_cluster.values())
    num_clusters = len(clusters)

    for node in G.nodes():
        k_i = G.degree(node)
        if k_i == 0:
            participation[node] = 0.0
            continue

        links_to_clusters = defaultdict(int)
        for neighbor in G.neighbors(node):
            if neighbor in node_cluster: # Ensure neighbor is in our dataset
                neighbor_cluster = node_cluster[neighbor]
                links_to_clusters[neighbor_cluster] += 1

        sum_sq_ratio = 0
        for cluster_id in clusters:
            k_is = links_to_clusters[cluster_id]
            sum_sq_ratio += (k_is / k_i) ** 2

        participation[node] = 1.0 - sum_sq_ratio

    return participation

# Update main to accept arguments
def main(json_file_path, output_csv_path):
    """
    Loads network data, calculates centrality measures, and exports to CSV.
    """
    # --- 1. Load Data ---
    try:
        with open(json_file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {json_file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_file_path}: {e}")
        return

    nodes_data = data['network']['items']
    links_data = data['network']['links']

    # --- 2. Build Graph ---
    G = nx.Graph()

    # Add nodes with attributes
    node_labels = {}
    node_clusters = {}
    for node in nodes_data:
        G.add_node(node['id'], label=node['label'], cluster=node['cluster'])
        node_labels[node['id']] = node['label']
        node_clusters[node['id']] = node['cluster']

    # Add edges (assuming unweighted for most centrality measures unless specified)
    for link in links_data:
        # Ensure both source and target nodes exist in the graph before adding edge
        if G.has_node(link['source_id']) and G.has_node(link['target_id']):
             G.add_edge(link['source_id'], link['target_id'], strength=link['strength'])
        # else:
        #     print(f"Warning: Skipping link {link} as one or both nodes not found in items list.")


    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- 3. Calculate Overall Graph Measures ---
    print("Calculating overall graph measures...")
    degree_centrality = nx.degree_centrality(G)
    # Note: networkx betweenness/closeness can be slow on large graphs
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True, weight=None)
    closeness_centrality = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85, weight=None) # Using default alpha
    clustering_coefficient = nx.clustering(G)

    # --- 4. Calculate Within-Cluster Measures ---
    print("Calculating within-cluster measures...")
    within_cluster_degree = defaultdict(float)
    within_cluster_closeness = defaultdict(float)
    within_cluster_pagerank = defaultdict(float)

    clusters = set(node_clusters.values())
    for cluster_id in clusters:
        # Create subgraph for the cluster
        cluster_nodes = [n for n, data in G.nodes(data=True) if data['cluster'] == cluster_id]
        subgraph = G.subgraph(cluster_nodes)

        if subgraph.number_of_nodes() > 0:
            # Calculate measures within the subgraph
            sub_degree = nx.degree_centrality(subgraph)
            sub_closeness = nx.closeness_centrality(subgraph)
            # Handle potential errors in PageRank for small/disconnected subgraphs
            try:
                sub_pagerank = nx.pagerank(subgraph, alpha=0.85, weight=None)
            except nx.NetworkXError:
                print(f"Warning: PageRank failed for cluster {cluster_id} (possibly disconnected or too small). Setting to 0.")
                sub_pagerank = {n: 0.0 for n in subgraph.nodes()}


            # Store results, mapping back to original node IDs
            for node_id in subgraph.nodes():
                within_cluster_degree[node_id] = sub_degree.get(node_id, 0.0)
                within_cluster_closeness[node_id] = sub_closeness.get(node_id, 0.0)
                within_cluster_pagerank[node_id] = sub_pagerank.get(node_id, 0.0)
        else:
             print(f"Warning: Cluster {cluster_id} has no nodes in the graph.")


    # --- 5. Calculate Participation Coefficient ---
    print("Calculating participation coefficient...")
    participation_coefficient = calculate_participation_coefficient(G, nodes_data)

    # --- 6. Combine Results ---
    print("Combining results...")
    results = []
    for node_id in G.nodes():
        results.append({
            'node_id': node_id,
            'label': node_labels.get(node_id, 'N/A'),
            'cluster': node_clusters.get(node_id, 'N/A'),
            'degree_centrality': degree_centrality.get(node_id, 0.0),
            'betweenness_centrality': betweenness_centrality.get(node_id, 0.0),
            'closeness_centrality': closeness_centrality.get(node_id, 0.0),
            'pagerank': pagerank.get(node_id, 0.0),
            'clustering_coefficient': clustering_coefficient.get(node_id, 0.0),
            'within_cluster_degree': within_cluster_degree.get(node_id, 0.0),
            'within_cluster_closeness': within_cluster_closeness.get(node_id, 0.0),
            'within_cluster_pagerank': within_cluster_pagerank.get(node_id, 0.0),
            'participation_coefficient': participation_coefficient.get(node_id, 0.0)
        })

    df = pd.DataFrame(results)

    # Reorder columns for clarity
    column_order = [
        'node_id', 'label', 'cluster',
        'degree_centrality', 'pagerank', 'closeness_centrality', 'betweenness_centrality', 'clustering_coefficient', # Overall
        'within_cluster_degree', 'within_cluster_pagerank', 'within_cluster_closeness', # Within-cluster
        'participation_coefficient' # Inter-cluster
    ]
    df = df[column_order]

    # --- 7. Export to CSV ---
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Results successfully exported to {output_csv_path}")
    except Exception as e:
        print(f"Error exporting results to CSV: {e}")


if __name__ == "__main__":
    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(description='Calculate network centrality measures from a VOSviewer JSON file.')
    parser.add_argument('input_json',
                        type=str,
                        help='Path to the input VOSviewer JSON file.')
    parser.add_argument('-o', '--output_csv',
                        type=str,
                        default='output/network_measures.csv',
                        help='Path to the output CSV file (default: output/network_measures.csv)')

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Run Main Function ---
    main(args.input_json, args.output_csv)