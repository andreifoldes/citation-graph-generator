#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import community as community_louvain  # python-louvain
import leidenalg as la
import igraph as ig
from collections import defaultdict, Counter
import matplotlib.cm as cm
import os
import pandas as pd

def load_citation_graph(file_path):
    """Load a citation graph from a GraphML file."""
    print(f"Loading citation graph from {file_path}...")
    G = nx.read_graphml(file_path)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def detect_communities_louvain(G):
    """Detect communities using the Louvain algorithm."""
    print("Detecting communities using Louvain algorithm...")
    
    # Convert to undirected graph for Louvain algorithm
    print("Converting directed graph to undirected for Louvain algorithm...")
    G_undirected = G.to_undirected()
    
    partition = community_louvain.best_partition(G_undirected)
    
    # Group nodes by community
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)
    
    print(f"Detected {len(communities)} communities using Louvain algorithm.")
    # Print community sizes
    for community_id, nodes in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"Community {community_id}: {len(nodes)} nodes")
    
    return partition, communities

def detect_communities_leiden(G):
    """Detect communities using the Leiden algorithm."""
    print("Detecting communities using Leiden algorithm...")
    
    # Convert NetworkX graph to igraph
    # Create a mapping of node names to indices
    node_map = {node: i for i, node in enumerate(G.nodes())}
    reverse_map = {i: node for node, i in node_map.items()}
    
    # Create igraph graph with the same number of vertices
    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(len(node_map))
    
    # Add edges using the indices
    edges = [(node_map[edge[0]], node_map[edge[1]]) for edge in G.edges()]
    ig_graph.add_edges(edges)
    
    # For Leiden, we can also convert to undirected if needed
    print("Converting to undirected graph for Leiden algorithm...")
    ig_graph_undirected = ig_graph.as_undirected(mode="collapse")
    
    # Run Leiden algorithm
    partition = la.find_partition(ig_graph_undirected, la.ModularityVertexPartition)
    
    # Map back to original node names
    communities = defaultdict(list)
    for i, community in enumerate(partition):
        for node_idx in community:
            communities[i].append(reverse_map[node_idx])
    
    print(f"Detected {len(communities)} communities using Leiden algorithm.")
    # Print community sizes, sorted by size
    for community_id, nodes in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"Community {community_id}: {len(nodes)} nodes")
    
    # Convert to same format as Louvain for consistency
    node_community = {}
    for community_id, nodes in communities.items():
        for node in nodes:
            node_community[node] = community_id
    
    return node_community, communities

def plot_communities(G, partition, algorithm_name, output_file=None, label_all_nodes=True):
    """Plot the graph with communities colored differently."""
    plt.figure(figsize=(14, 14))
    
    # Get unique communities
    community_ids = set(partition.values())
    
    # Sort communities by size (largest first)
    community_sizes = [(comm_id, sum(1 for node, comm in partition.items() if comm == comm_id)) 
                       for comm_id in community_ids]
    sorted_communities = [comm_id for comm_id, size in sorted(community_sizes, key=lambda x: x[1], reverse=True)]
    
    # Use a colormap suitable for categorical data
    colors = cm.tab20(np.linspace(0, 1, len(sorted_communities))) if len(sorted_communities) <= 20 else \
             cm.viridis(np.linspace(0, 1, len(sorted_communities)))
    color_map = {comm_id: colors[i] for i, comm_id in enumerate(sorted_communities)}
    
    # Position nodes using a layout algorithm with stronger gravity
    pos = nx.spring_layout(G, k=0.15, iterations=100, seed=42)
    
    # Draw the graph
    plt.title(f"Citation Network Communities ({algorithm_name} Algorithm)", fontsize=18)
    
    # Draw edges with transparency
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True, arrowsize=8)
    
    # Draw nodes by community
    for community_id in sorted_communities:
        node_list = [node for node, comm in partition.items() if comm == community_id]
        
        # If it's a single-node community, make it smaller
        node_size = 100 if len(node_list) > 1 else 50
        
        # Only label the top 5 communities to avoid cluttering
        if sorted_communities.index(community_id) < 5:
            label = f"Community {community_id} ({len(node_list)} papers)"
        else:
            label = None
            
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=node_list, 
                              node_color=[color_map[community_id]],
                              node_size=node_size, 
                              alpha=0.8,
                              label=label)
    
    # Create node labels
    labels = {}
    
    if label_all_nodes:
        # Create labels for all nodes using first author and year
        for node in G.nodes():
            first_author = G.nodes[node].get('first_author', '')
            year = G.nodes[node].get('year', '')
            if first_author and year:
                labels[node] = f"{first_author}, {year}"
            else:
                labels[node] = node
                
        # Draw labels for all nodes with smaller font size
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_family="sans-serif")
        
    else:
        # Only label important nodes (as before)
        # Calculate node importance (using degree centrality)
        centrality = nx.degree_centrality(G)
        
        # Get the top 15 nodes by centrality
        important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Create a dictionary with labels for important nodes
        for node, _ in important_nodes:
            # Use the first author and year from node attributes
            first_author = G.nodes[node].get('first_author', '')
            year = G.nodes[node].get('year', '')
            if first_author and year:
                labels[node] = f"{first_author}, {year}"
            else:
                labels[node] = node
        
        # Draw labels for important nodes
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_family="sans-serif")
    
    plt.legend(loc='upper right', fontsize=10)
    plt.axis('off')
    
    if output_file:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, format="PNG", dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    
    plt.show()

def analyze_communities(G, communities, first_author_key='first_author', year_key='year'):
    """Analyze the content of each community to identify themes."""
    print("\nCommunity Analysis:")
    
    # Sort communities by size (largest first)
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    
    metrics = []
    
    for community_id, nodes in sorted_communities:
        print(f"\nCommunity {community_id} ({len(nodes)} papers):")
        
        # Get most frequent authors
        authors = [G.nodes[node].get(first_author_key, 'Unknown') for node in nodes if first_author_key in G.nodes[node]]
        author_count = Counter(authors)
        
        # Get year distribution
        years = [G.nodes[node].get(year_key, 'Unknown') for node in nodes if year_key in G.nodes[node]]
        year_count = Counter(years)
        
        # Print results
        print("Top authors:")
        for author, count in author_count.most_common(5):
            print(f"  {author}: {count} papers")
        
        print("Year distribution:")
        for year, count in sorted(year_count.items()):
            print(f"  {year}: {count} papers")
        
        # Calculate network metrics for this community
        subgraph = G.subgraph(nodes)
        
        # Internal vs. external citations
        internal_citations = 0
        external_citations = 0
        community_set = set(nodes)
        
        for node in nodes:
            for _, target in G.out_edges(node):
                if target in community_set:
                    internal_citations += 1
                else:
                    external_citations += 1
        
        total_citations = internal_citations + external_citations
        
        # Density of the community subgraph
        density = nx.density(subgraph)
        
        # Average clustering coefficient (local transitivity)
        try:
            # For directed graphs, convert to undirected for clustering coefficient
            undirected_subgraph = subgraph.to_undirected()
            avg_clustering = nx.average_clustering(undirected_subgraph)
        except:
            avg_clustering = 0
        
        # Store metrics
        community_metrics = {
            'community_id': community_id,
            'size': len(nodes),
            'density': density,
            'avg_clustering': avg_clustering,
            'internal_citations': internal_citations,
            'external_citations': external_citations
        }
        metrics.append(community_metrics)
        
        if total_citations > 0:
            print(f"Citation patterns:")
            print(f"  Internal citations: {internal_citations} ({internal_citations/total_citations:.1%})")
            print(f"  External citations: {external_citations} ({external_citations/total_citations:.1%})")
        
        print(f"Network metrics:")
        print(f"  Density: {density:.3f}")
        print(f"  Avg. clustering coefficient: {avg_clustering:.3f}")
    
    return metrics

def plot_community_metrics(metrics, algorithm_name, output_file=None):
    """Plot metrics for communities."""
    # Filter out single-node communities
    metrics = [m for m in metrics if m['size'] > 1]
    
    if not metrics:
        print("No multi-node communities to analyze.")
        return
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sort communities by size for plotting
    metrics.sort(key=lambda x: x['size'], reverse=True)
    
    # Community IDs for x-axis (keep only the top 10 communities for readability)
    top_metrics = metrics[:10]
    comm_ids = [str(m['community_id']) for m in top_metrics]
    
    # Plot 1: Community Size
    axs[0, 0].bar(comm_ids, [m['size'] for m in top_metrics], color='skyblue')
    axs[0, 0].set_title('Community Size')
    axs[0, 0].set_xlabel('Community ID')
    axs[0, 0].set_ylabel('Number of Papers')
    
    # Plot 2: Internal vs External Citations
    internal = [m['internal_citations'] for m in top_metrics]
    external = [m['external_citations'] for m in top_metrics]
    
    axs[0, 1].bar(comm_ids, internal, label='Internal Citations', color='green', alpha=0.7)
    axs[0, 1].bar(comm_ids, external, bottom=internal, label='External Citations', color='red', alpha=0.7)
    axs[0, 1].set_title('Citation Patterns')
    axs[0, 1].set_xlabel('Community ID')
    axs[0, 1].set_ylabel('Number of Citations')
    axs[0, 1].legend()
    
    # Plot 3: Density
    axs[1, 0].bar(comm_ids, [m['density'] for m in top_metrics], color='purple')
    axs[1, 0].set_title('Community Density')
    axs[1, 0].set_xlabel('Community ID')
    axs[1, 0].set_ylabel('Density')
    
    # Plot 4: Average Clustering Coefficient
    axs[1, 1].bar(comm_ids, [m['avg_clustering'] for m in top_metrics], color='orange')
    axs[1, 1].set_title('Avg. Clustering Coefficient')
    axs[1, 1].set_xlabel('Community ID')
    axs[1, 1].set_ylabel('Clustering Coefficient')
    
    plt.suptitle(f'Community Metrics ({algorithm_name} Algorithm)', fontsize=16)
    plt.tight_layout()
    
    if output_file:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, format="PNG", dpi=300, bbox_inches="tight")
        print(f"Metrics plot saved to {output_file}")
    
    plt.show()

def export_community_assignments(G, louvain_partition, leiden_partition, output_file):
    """
    Export node information and community assignments to a CSV file.
    
    Parameters:
    -----------
    G : networkx.Graph
        The citation graph
    louvain_partition : dict
        Mapping of node IDs to community IDs from Louvain algorithm
    leiden_partition : dict
        Mapping of node IDs to community IDs from Leiden algorithm
    output_file : str
        Path to the output CSV file
    """
    print(f"Exporting community assignments to {output_file}...")
    
    # Create a list to store the data
    data = []
    
    for node in G.nodes():
        # Get node attributes
        first_author = G.nodes[node].get('first_author', 'Unknown')
        year = G.nodes[node].get('year', 'Unknown')
        label = G.nodes[node].get('label', f"{first_author}, {year}")
        doi = node  # The node ID is the DOI
        
        # Get community assignments
        louvain_community = louvain_partition.get(node, 'Not assigned')
        leiden_community = leiden_partition.get(node, 'Not assigned')
        
        # Add to data list
        data.append({
            'DOI': doi,
            'Label': label,
            'First Author': first_author,
            'Year': year,
            'Louvain Community': louvain_community,
            'Leiden Community': leiden_community
        })
    
    # Create dataframe and export to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Exported {len(data)} nodes to {output_file}")

def main():
    # Load the citation graph
    citation_graph = load_citation_graph('output/citation_graph.graphml')
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Detect communities using Louvain algorithm
    louvain_partition, louvain_communities = detect_communities_louvain(citation_graph)
    
    # Plot the communities detected by Louvain
    plot_communities(citation_graph, louvain_partition, "Louvain", "output/louvain_communities.png", label_all_nodes=True)
    
    # Analyze Louvain communities
    louvain_metrics = analyze_communities(citation_graph, louvain_communities)
    
    # Plot metrics for Louvain communities
    plot_community_metrics(louvain_metrics, "Louvain", "output/louvain_metrics.png")
    
    # Create empty leiden_partition as a fallback
    leiden_partition = {}
    
    try:
        # Detect communities using Leiden algorithm
        leiden_partition, leiden_communities = detect_communities_leiden(citation_graph)
        
        # Plot the communities detected by Leiden
        plot_communities(citation_graph, leiden_partition, "Leiden", "output/leiden_communities.png", label_all_nodes=True)
        
        # Analyze Leiden communities
        leiden_metrics = analyze_communities(citation_graph, leiden_communities)
        
        # Plot metrics for Leiden communities
        plot_community_metrics(leiden_metrics, "Leiden", "output/leiden_metrics.png")
        
        # Compare Louvain and Leiden results
        print("\nComparison between Louvain and Leiden algorithms:")
        print(f"Louvain detected {len(louvain_communities)} communities.")
        print(f"Leiden detected {len(leiden_communities)} communities.")
        
        # Calculate community similarity using adjusted rand index
        try:
            from sklearn.metrics.cluster import adjusted_rand_score
            
            # Convert partitions to lists
            louvain_labels = [louvain_partition[node] for node in citation_graph.nodes()]
            leiden_labels = [leiden_partition[node] for node in citation_graph.nodes()]
            
            # Calculate similarity
            ari = adjusted_rand_score(louvain_labels, leiden_labels)
            print(f"Adjusted Rand Index (similarity between partitions): {ari:.3f}")
        except ImportError:
            print("sklearn not installed, skipping partition similarity calculation.")
        
    except Exception as e:
        print(f"Error running Leiden algorithm: {e}")
        print("Continuing with only Louvain results.")
    
    # Export community assignments to CSV
    export_community_assignments(citation_graph, louvain_partition, leiden_partition, "output/community_assignments.csv")

if __name__ == "__main__":
    main()
