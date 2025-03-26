#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%% Imports
import re
import time
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
import bibtexparser
from urllib.parse import quote
import random
import argparse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables
global_email = "anonymous@example.com"
opencitations_token = os.getenv("OPENCITATIONS_ACCESS_TOKEN")  # Keeping this defined but not using it

#%% Gather citations data

def extract_dois_from_bibtex(bibtex_file):
    """Extract DOIs and metadata from a BibTeX file."""
    with open(bibtex_file, 'r', encoding='utf-8') as file:
        bibtex_str = file.read()
    
    # Parse the BibTeX file
    bib_database = bibtexparser.loads(bibtex_str)
    
    # Extract DOIs and metadata
    dois = []
    doi_metadata = {}
    for entry in bib_database.entries:
        if 'doi' in entry:
            doi = entry['doi']
            dois.append(doi)
            
            # Extract author and year information
            first_author = "Unknown"
            if 'author' in entry:
                authors = entry['author'].split(' and ')
                if authors:
                    # Get the last name of the first author
                    first_author_full = authors[0].strip()
                    # Handle different name formats (Last, First or First Last)
                    if ',' in first_author_full:
                        first_author = first_author_full.split(',')[0].strip()
                    else:
                        name_parts = first_author_full.split()
                        if name_parts:
                            first_author = name_parts[-1].strip()
            
            year = entry.get('year', 'Unknown')
            
            # Store metadata
            doi_metadata[doi] = {
                'first_author': first_author,
                'year': year,
                'label': f"{first_author}, {year}"
            }
    
    return dois, doi_metadata

def make_api_request(url, max_retries=3, base_delay=1):
    """Make an API request with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            # Add a small random delay to avoid hitting rate limits
            time.sleep(base_delay + random.random())
            
            # Add email parameter for better API service
            if '?' in url:
                request_url = f"{url}&email={global_email}"
            else:
                request_url = f"{url}?email={global_email}"
                
            response = requests.get(request_url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Too Many Requests
                wait_time = (2 ** attempt) + random.random()
                print(f"Rate limit hit. Waiting {wait_time:.2f}s before retrying...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Error: Status code {response.status_code} for URL: {url}")
                return None
                
        except requests.exceptions.RequestException as e:
            wait_time = (2 ** attempt) + random.random()
            print(f"Request error: {e}. Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)
        except json.JSONDecodeError:
            print(f"Invalid JSON response from API for URL: {url}")
            return None
            
    print(f"Failed after {max_retries} attempts for URL: {url}")
    return None

def get_openalex_id_from_doi(doi):
    """Get OpenAlex ID from DOI."""
    try:
        # Properly encode the DOI for the URL
        encoded_doi = quote(doi, safe='')
        url = f"https://api.openalex.org/works/doi:{encoded_doi}"
        
        data = make_api_request(url)
        if data:
            return data.get('id')
        else:
            print(f"Could not get OpenAlex ID for DOI: {doi}")
            return None
    except Exception as e:
        print(f"Exception when fetching OpenAlex ID for DOI {doi}: {e}")
        return None

def get_referenced_works(openalex_id):
    """Get works referenced by a given OpenAlex ID."""
    if not openalex_id:
        return []
    
    try:
        # Get the work details directly from the OpenAlex API
        # The referenced_works field is part of the work object
        work_url = f"https://api.openalex.org/works/{openalex_id.split('/')[-1]}"
        work_data = make_api_request(work_url)
        
        if work_data:
            # Extract referenced works directly from the work details
            references = work_data.get('referenced_works', [])
            
            if references:
                print(f"Found {len(references)} references for {openalex_id}")
                return references
            else:
                print(f"No references found for {openalex_id} in OpenAlex data")
                return []
        else:
            print(f"Could not get work details for {openalex_id}")
            return []
    except Exception as e:
        print(f"Exception when fetching references for {openalex_id}: {e}")
        return []

def get_cited_by_works(openalex_id):
    """Get works that cite a given OpenAlex ID."""
    if not openalex_id:
        return []
    
    try:
        # Extract the OpenAlex ID number
        openalex_id_number = openalex_id.split('/')[-1]
        url = f"https://api.openalex.org/works?filter=cites:{openalex_id_number}&per_page=100"
        
        data = make_api_request(url)
        if data:
            return [work.get('id') for work in data.get('results', [])]
        else:
            return []
    except Exception as e:
        print(f"Exception when fetching citing works for {openalex_id}: {e}")
        return []

def get_crossref_work(doi):
    """Get work details from Crossref API."""
    if not doi:
        return None
    
    try:
        # Properly encode the DOI for the URL
        encoded_doi = quote(doi, safe='')
        url = f"https://api.crossref.org/works/{encoded_doi}"
        
        # Add polite pool parameter if email is provided
        if global_email and global_email != "anonymous@example.com":
            url = f"{url}?mailto={global_email}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('message')
        else:
            print(f"Error fetching work from Crossref for DOI {doi}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception when fetching work from Crossref for DOI {doi}: {e}")
        return None

def get_crossref_references(doi):
    """Get references (works cited by this paper) from Crossref."""
    work_data = get_crossref_work(doi)
    if not work_data or 'reference' not in work_data:
        return []
    
    # Extract DOIs from references
    references = []
    for ref in work_data.get('reference', []):
        if 'DOI' in ref:
            references.append(ref['DOI'].lower())  # Normalize DOI to lowercase
    
    return references

def get_paper_metadata(doi, api_name, doi_metadata=None):
    """Get paper metadata (first author, year) based on the API being used."""
    # If we already have metadata from BibTeX, use it
    if doi_metadata and doi in doi_metadata:
        return doi_metadata[doi]
    
    # Default metadata
    metadata = {
        'first_author': 'Unknown',
        'year': 'Unknown',
        'label': f"Unknown, Unknown"
    }
    
    # Try to get metadata from the appropriate API
    try:
        if api_name == "crossref":
            work_data = get_crossref_work(doi)
            if work_data:
                # Extract first author from Crossref
                first_author = "Unknown"
                if 'author' in work_data and work_data['author']:
                    author = work_data['author'][0]
                    first_author = author.get('family', 'Unknown')
                
                # Extract year from Crossref
                year = "Unknown"
                if 'published-print' in work_data and 'date-parts' in work_data['published-print']:
                    year = str(work_data['published-print']['date-parts'][0][0])
                elif 'published-online' in work_data and 'date-parts' in work_data['published-online']:
                    year = str(work_data['published-online']['date-parts'][0][0])
                elif 'created' in work_data and 'date-parts' in work_data['created']:
                    year = str(work_data['created']['date-parts'][0][0])
                
                metadata = {
                    'first_author': first_author,
                    'year': year,
                    'label': f"{first_author}, {year}"
                }
        
        elif api_name == "openalex":
            # For OpenAlex, we could make another API call, but that would add overhead
            # For now, we'll just use the DOI-based label as fallback
            pass
    
    except Exception as e:
        print(f"Error getting metadata for DOI {doi}: {e}")
    
    return metadata

def build_citation_graph(dois, doi_metadata=None, delay_range=(1, 2)):
    """Build a citation graph from a list of DOIs using OpenAlex API.
    
    Args:
        dois: List of DOIs to process
        doi_metadata: Dictionary of metadata for DOIs from BibTeX
        delay_range: Tuple of (min, max) delay between API calls in seconds
    """
    # Create a graph
    G = nx.DiGraph()
    
    # Map DOIs to OpenAlex IDs
    print("Getting OpenAlex IDs for DOIs...")
    doi_to_openalex = {}
    for doi in tqdm(dois):
        openalex_id = get_openalex_id_from_doi(doi)
        if openalex_id:
            doi_to_openalex[doi] = openalex_id
            # Get metadata for the node
            metadata = doi_metadata.get(doi, {}) if doi_metadata else {}
            
            # If label doesn't exist in metadata, create it
            if 'label' not in metadata:
                metadata['label'] = doi.split('/')[-1]
            
            # Add node to the graph with metadata
            G.add_node(openalex_id, doi=doi, **metadata)
    
    print(f"Successfully mapped {len(doi_to_openalex)} out of {len(dois)} DOIs to OpenAlex IDs.")
    
    # Get references for each paper
    print("Getting references for each paper...")
    references_count = 0
    for doi, openalex_id in tqdm(doi_to_openalex.items()):
        # Get works referenced by this paper
        referenced_works = get_referenced_works(openalex_id)
        
        # Add edges for references - paper A references paper B (A -> B)
        for ref_work in referenced_works:
            if ref_work in G:  # If the referenced work is in our graph
                G.add_edge(openalex_id, ref_work, type='references')
                references_count += 1
    
    print(f"Found {references_count} reference relationships within the dataset.")
    
    # We'll skip getting citations separately since they're just the reverse of references
    # This avoids creating duplicate edges in opposite directions
    
    return G

def build_crossref_graph(dois, doi_metadata=None, delay_range=(1, 2)):
    """Build a citation graph from a list of DOIs using Crossref API.
    
    Args:
        dois: List of DOIs to process
        doi_metadata: Dictionary of metadata for DOIs from BibTeX
        delay_range: Tuple of (min, max) delay between API calls in seconds
    """
    # Create a graph
    G = nx.DiGraph()
    
    # Normalize DOIs to lowercase
    normalized_dois = [doi.lower() for doi in dois]
    
    # Add nodes for all DOIs
    print("Adding nodes for all DOIs...")
    for i, doi in enumerate(tqdm(normalized_dois)):
        # Get metadata for the node
        metadata = doi_metadata.get(dois[i], {}) if doi_metadata else {}
        
        # If label doesn't exist in metadata, create it
        if 'label' not in metadata:
            metadata['label'] = doi.split('/')[-1]
        
        G.add_node(doi, **metadata)
    
    # Get references for each paper
    print("Getting references for each paper...")
    references_count = 0
    for doi in tqdm(normalized_dois):
        # Get works referenced by this paper
        referenced_dois = get_crossref_references(doi)
        
        # Add edges for references (from paper to its references)
        for ref_doi in referenced_dois:
            if ref_doi in G:  # If the referenced work is in our graph
                G.add_edge(doi, ref_doi, type='references')
                references_count += 1
        
        # Be gentle with the API
        time.sleep(random.uniform(*delay_range))
    
    print(f"Found {references_count} reference relationships within the dataset.")
    
    return G

def save_citation_graph(G, output_file):
    """Save the citation graph to a file."""
    # Convert the graph to a dictionary
    graph_data = {
        'nodes': [{'id': node, 'doi': G.nodes[node].get('doi', '')} for node in G.nodes],
        'edges': [{'source': u, 'target': v, 'type': G.edges[u, v]['type']} for u, v in G.edges]
    }
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)

def find_intermediary_papers(G, api_name, max_depth=3, max_papers_per_node=10):
    """Find intermediary papers that connect isolated nodes in the citation graph.
    
    Args:
        G: The citation graph (NetworkX DiGraph)
        api_name: The API to use (openalex, crossref, or semanticscholar)
        max_depth: Maximum depth to search for connections (default: 3)
        max_papers_per_node: Maximum number of intermediary papers to add per isolated node
        
    Returns:
        A new graph with intermediary papers added
    """
    # Create a copy of the graph to add intermediary papers
    G_extended = G.copy()
    
    # Find isolated nodes (nodes with no connections to other nodes in the graph)
    isolated_nodes = []
    for node in G.nodes():
        # Check if node has no connections to other nodes in the graph
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        if not neighbors:
            isolated_nodes.append(node)
    
    print(f"Found {len(isolated_nodes)} isolated nodes out of {len(G.nodes())} total nodes.")
    
    if not isolated_nodes:
        print("No isolated nodes found. No need to look for intermediary papers.")
        return G_extended
    
    # For each isolated node, try to find connections through intermediary papers
    intermediary_connections = 0
    
    for node in tqdm(isolated_nodes, desc="Finding intermediary papers"):
        # Get the DOI for the node
        if api_name == "openalex":
            # For OpenAlex, the node is the OpenAlex ID and we need to get the DOI from node attributes
            doi = G.nodes[node].get('doi', '')
        else:
            # For Crossref and SemanticScholar, the node is the DOI
            doi = node
        
        if not doi:
            continue
        
        # Initialize our search from this isolated node
        visited = set()  # Keep track of papers we've already visited
        found_connecting_paper = False
        
        # Queue for BFS with tuples of (paper_id, depth, path)
        # Path is a list of papers that connect the isolated node to the current paper
        if api_name == "openalex":
            queue = [(node, 0, [])]
        else:
            queue = [(doi, 0, [])]
        
        # Perform a breadth-first search up to max_depth
        while queue and not found_connecting_paper:
            current_paper, depth, path = queue.pop(0)
            
            # Skip if we've already visited this paper or reached max depth
            if current_paper in visited or depth > max_depth:
                continue
            
            visited.add(current_paper)
            
            # Get the DOI for the current paper if needed
            current_doi = None
            if api_name == "openalex":
                if current_paper == node:
                    current_doi = doi  # We already know the DOI of the isolated node
                else:
                    # Get DOI for the paper
                    openalex_id_number = current_paper.split('/')[-1]
                    url = f"https://api.openalex.org/works/{openalex_id_number}"
                    data = make_api_request(url)
                    if data and 'doi' in data:
                        current_doi = data['doi']
            else:
                current_doi = current_paper
            
            if not current_doi:
                continue
            
            # Get papers that cite this paper
            citing_papers = []
            if api_name == "openalex":
                citing_papers = get_cited_by_works(current_paper)
            elif api_name == "semanticscholar":
                citing_papers = get_semanticscholar_citations(current_doi)
            # No citations for Crossref
            
            # Get papers cited by this paper
            cited_papers = []
            if api_name == "openalex":
                cited_papers = get_referenced_works(current_paper)
            elif api_name == "semanticscholar":
                cited_papers = get_semanticscholar_references(current_doi)
            elif api_name == "crossref":
                cited_papers = get_crossref_references(current_doi)
            
            # Combine all connected papers
            connected_papers = citing_papers + cited_papers
            
            # Prioritize connected papers based on potential to connect to the rest of the graph
            # For OpenAlex, we can actually get additional metadata for better prioritization
            prioritized_papers = []
            
            for paper_id in connected_papers:
                # Skip if already visited
                if paper_id in visited:
                    continue
                
                if api_name == "openalex":
                    # For OpenAlex we can get citation counts and reference counts
                    openalex_id_number = paper_id.split('/')[-1]
                    url = f"https://api.openalex.org/works/{openalex_id_number}"
                    data = make_api_request(url)
                    
                    if not data:
                        continue
                        
                    # Calculate a score based on citation count and reference count
                    citation_count = data.get('cited_by_count', 0)
                    reference_count = len(data.get('referenced_works', []))
                    
                    # Papers with more citations and references are more likely to connect
                    score = citation_count + reference_count
                    
                    # Check if any authors of this paper have other papers in our graph
                    has_author_in_graph = False
                    if 'authorships' in data:
                        for authorship in data['authorships']:
                            author_id = authorship.get('author', {}).get('id')
                            if author_id:
                                # Check if this author has other papers in our graph
                                # This would require a separate function to track authors
                                # For simplicity, we'll just boost the score
                                score += 50
                                break
                    
                    prioritized_papers.append((paper_id, score))
                else:
                    # For other APIs, we don't have as much metadata
                    # We could do an additional API call but for simplicity, just add with a default score
                    # We'll use the position in the API response as a proxy for importance
                    position_score = len(connected_papers) - connected_papers.index(paper_id)
                    prioritized_papers.append((paper_id, position_score))
            
            # Sort papers by score in descending order
            prioritized_papers.sort(key=lambda x: x[1], reverse=True)
            
            # Increase max_papers_per_node for the first level of search to cast a wider net
            actual_max_papers = max_papers_per_node
            if depth == 0:
                actual_max_papers = max(30, max_papers_per_node * 2)
            
            # Limit to prevent too many API calls but use more papers at the first level
            if len(prioritized_papers) > actual_max_papers:
                prioritized_papers = prioritized_papers[:actual_max_papers]
            
            # Now continue with the rest of the algorithm using the prioritized papers
            for paper_id, score in prioritized_papers:
                # Get papers cited by this connected paper
                connected_paper_cited = []
                connected_paper_doi = None
                
                if api_name == "openalex":
                    connected_paper_cited = get_referenced_works(paper_id)
                    # Get DOI for the connected paper
                    openalex_id_number = paper_id.split('/')[-1]
                    url = f"https://api.openalex.org/works/{openalex_id_number}"
                    data = make_api_request(url)
                    if data and 'doi' in data:
                        connected_paper_doi = data['doi']
                elif api_name == "semanticscholar":
                    connected_paper_doi = paper_id
                    connected_paper_cited = get_semanticscholar_references(paper_id)
                elif api_name == "crossref":
                    connected_paper_doi = paper_id
                    connected_paper_cited = get_crossref_references(paper_id)
                
                # Check if this paper connects to any other node in the original graph
                # besides the isolated node we're working with
                connects_to_other_nodes = False
                non_isolated_nodes_connected = []
                
                for cited in connected_paper_cited:
                    # Check if the cited paper is in our original graph and is not our isolated node
                    if cited in G.nodes() and ((api_name == "openalex" and cited != node) or \
                       (api_name != "openalex" and cited != doi)):
                        connects_to_other_nodes = True
                        non_isolated_nodes_connected.append(cited)
                
                # Is this paper a good intermediary that connects our isolated node to others?
                if connects_to_other_nodes and connected_paper_doi:
                    # We found a good intermediary paper!
                    label = f"Intermediary: {connected_paper_doi.split('/')[-1]}"
                    G_extended.add_node(paper_id, label=label, is_intermediary=True, doi=connected_paper_doi)
                    
                    # Add the path of intermediary papers that led us here
                    for i, path_paper in enumerate(path):
                        # Skip if the path_paper is already in the graph
                        if path_paper in G_extended and G_extended.nodes[path_paper].get('is_intermediary', False):
                            continue
                            
                        path_paper_doi = None
                        if api_name == "openalex":
                            # Get DOI for the paper in the path
                            openalex_id_number = path_paper.split('/')[-1]
                            url = f"https://api.openalex.org/works/{openalex_id_number}"
                            data = make_api_request(url)
                            if data and 'doi' in data:
                                path_paper_doi = data['doi']
                        else:
                            path_paper_doi = path_paper
                            
                        if path_paper_doi:
                            path_label = f"Path: {path_paper_doi.split('/')[-1]}"
                            G_extended.add_node(path_paper, label=path_label, is_intermediary=True, doi=path_paper_doi)
                            
                            # Add edge to the next paper in the path
                            if i < len(path) - 1:
                                G_extended.add_edge(path_paper, path[i+1], type='references')
                            else:
                                # Connect the last paper in the path to our connecting paper
                                G_extended.add_edge(path_paper, paper_id, type='references')
                    
                    # Connect our isolated node to the first paper in the path if any
                    if api_name == "openalex":
                        if path:
                            G_extended.add_edge(node, path[0], type='references')
                        else:
                            # Direct connection
                            G_extended.add_edge(node, paper_id, type='references')
                    else:
                        if path:
                            G_extended.add_edge(doi, path[0], type='references')
                        else:
                            # Direct connection
                            G_extended.add_edge(doi, paper_id, type='references')
                    
                    # Add edges from the connecting paper to the non-isolated nodes it connects to
                    for non_isolated in non_isolated_nodes_connected:
                        G_extended.add_edge(paper_id, non_isolated, type='references')
                    
                    intermediary_connections += 1
                    found_connecting_paper = True
                    print(f"Found connecting paper for {node if api_name == 'openalex' else doi} at depth {depth+1} (score: {score})")
                    break
                
                # If we haven't reached max depth, add this paper to the queue
                if depth + 1 < max_depth and not found_connecting_paper:
                    # Add this paper to the path
                    new_path = path + [paper_id]
                    queue.append((paper_id, depth + 1, new_path))
            
            # Add a small delay to be nice to the API
            time.sleep(random.uniform(0.5, 1))
        
        if not found_connecting_paper:
            print(f"Could not find connecting papers for node {node if api_name == 'openalex' else doi} within {max_depth} hops")
    
    print(f"Added {intermediary_connections} intermediary connections to connect isolated nodes.")
    return G_extended

def visualize_citation_graph(G, output_file):
    """Visualize the citation graph and save to a file."""
    plt.figure(figsize=(15, 10))
    
    # Use a layout that works well for directed graphs
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Separate nodes into different categories for styling
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    intermediary_nodes = [node for node in G.nodes() if G.nodes[node].get('is_intermediary', False) and node not in isolated_nodes]
    regular_nodes = [node for node in G.nodes() if node not in isolated_nodes and node not in intermediary_nodes]
    
    # Draw regular nodes
    nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_size=300, alpha=0.8, node_color='blue')
    
    # Draw intermediary nodes with a different color
    if intermediary_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=intermediary_nodes, node_size=300, alpha=0.8, node_color='red')
    
    # Draw isolated nodes with yet another color
    if isolated_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=isolated_nodes, node_size=300, alpha=0.8, node_color='green')
    
    # Draw edges - all represent one paper citing another
    edges = list(G.edges(data=True))
    
    # Use clearer arrow styling - arrow points FROM the citing paper TO the cited paper
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in edges], 
                          edge_color='blue', alpha=0.6, arrows=True, 
                          arrowstyle='->', arrowsize=15, connectionstyle='arc3,rad=0.1')
    
    # Add labels using "First Author, Year" format
    labels = {node: G.nodes[node].get('label', '') for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.axis('off')
    plt.title('Citation Graph')
    plt.tight_layout()
    
    # Add a clearer legend
    blue_line = plt.Line2D([0], [0], color='blue', lw=2, label='Paper A cites Paper B (A → B)')
    blue_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Connected Papers')
    green_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Isolated Papers (No Connections)')
    
    legend_elements = [blue_line, blue_dot, green_dot]
    if intermediary_nodes:
        red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Intermediary Papers')
        legend_elements.append(red_dot)
    
    plt.legend(handles=legend_elements)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def generate_citation_report(G, output_file):
    """Generate a report of the citation relationships."""
    # Count citations for each paper
    citation_counts = {}
    for node in G.nodes:
        # Get the DOI (either directly if the node is a DOI, or from node attributes)
        if 'doi' in G.nodes[node]:
            doi = G.nodes[node]['doi']
        else:
            doi = node  # Node itself is the DOI
            
        # Count incoming and outgoing edges
        # Incoming edges = cited by others (other papers reference this one)
        # Outgoing edges = references (this paper references others)
        cited_by_count = len(G.in_edges(node))
        references_count = len(G.out_edges(node))
        
        citation_counts[doi] = {
            'cited_by_count': cited_by_count,
            'references_count': references_count
        }
    
    # Create a DataFrame for better viewing
    df = pd.DataFrame.from_dict(citation_counts, orient='index')
    df.index.name = 'DOI'
    df.reset_index(inplace=True)
    df.sort_values(by='cited_by_count', ascending=False, inplace=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)

def get_semanticscholar_paper(doi):
    """Get paper details from Semantic Scholar API."""
    if not doi:
        return None
    
    try:
        # Properly encode the DOI for the URL
        encoded_doi = quote(doi, safe='')
        url = f"https://api.semanticscholar.org/v1/paper/DOI:{encoded_doi}"
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(random.uniform(1, 2))
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:  # Too Many Requests
            print("Rate limit hit on Semantic Scholar API. Waiting before retrying...")
            time.sleep(10 + random.random() * 5)
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching paper from Semantic Scholar for DOI {doi} after retry: {response.status_code}")
                return None
        else:
            print(f"Error fetching paper from Semantic Scholar for DOI {doi}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception when fetching paper from Semantic Scholar for DOI {doi}: {e}")
        return None

def get_semanticscholar_references(doi):
    """Get references (works cited by this paper) from Semantic Scholar."""
    paper_data = get_semanticscholar_paper(doi)
    if not paper_data or 'references' not in paper_data:
        return []
    
    # Extract DOIs from references
    references = []
    for ref in paper_data.get('references', []):
        if ref.get('doi'):
            references.append(ref['doi'].lower())  # Normalize DOI to lowercase
    
    return references

def get_semanticscholar_citations(doi):
    """Get citations (works that cite this paper) from Semantic Scholar."""
    paper_data = get_semanticscholar_paper(doi)
    if not paper_data or 'citations' not in paper_data:
        return []
    
    # Extract DOIs from citations
    citations = []
    for cit in paper_data.get('citations', []):
        if cit.get('doi'):
            citations.append(cit['doi'].lower())  # Normalize DOI to lowercase
    
    return citations

def build_semanticscholar_graph(dois, doi_metadata=None, delay_range=(1, 2)):
    """Build a citation graph from a list of DOIs using Semantic Scholar API.
    
    Args:
        dois: List of DOIs to process
        doi_metadata: Dictionary of metadata for DOIs from BibTeX
        delay_range: Tuple of (min, max) delay between API calls in seconds
    """
    # Create a graph
    G = nx.DiGraph()
    
    # Normalize DOIs to lowercase
    normalized_dois = [doi.lower() for doi in dois]
    
    # Add nodes for all DOIs
    print("Adding nodes for all DOIs...")
    for i, doi in enumerate(tqdm(normalized_dois)):
        # Get metadata for the node
        metadata = doi_metadata.get(dois[i], {}) if doi_metadata else {}
        
        # If label doesn't exist in metadata, create it
        if 'label' not in metadata:
            metadata['label'] = doi.split('/')[-1]
        
        G.add_node(doi, **metadata)
    
    # Get references for each paper
    print("Getting references for each paper...")
    references_count = 0
    for doi in tqdm(normalized_dois):
        # Get works referenced by this paper
        referenced_dois = get_semanticscholar_references(doi)
        
        # Add edges for references (from paper to its references)
        for ref_doi in referenced_dois:
            if ref_doi in G:  # If the referenced work is in our graph
                G.add_edge(doi, ref_doi, type='references')
                references_count += 1
        
        # Be gentle with the API
        time.sleep(random.uniform(*delay_range))
    
    print(f"Found {references_count} reference relationships within the dataset.")
    
    return G

def merge_citation_graphs(graphs, prefer_metadata=None):
    """Merge multiple citation graphs into one comprehensive graph.
    
    Args:
        graphs: List of (graph, api_name) tuples to merge
        prefer_metadata: API name to prefer for node metadata in case of conflicts
        
    Returns:
        Merged NetworkX DiGraph
    """
    if not graphs:
        return nx.DiGraph()
    
    # Create a new graph for the merged result
    merged_graph = nx.DiGraph()
    
    print("Merging citation graphs...")
    
    # First, collect all nodes across all graphs
    all_nodes = set()
    node_metadata = {}
    
    for graph, api_name in graphs:
        for node in graph.nodes():
            # For OpenAlex, the node is the OpenAlex ID and we need to get the DOI
            if api_name == "openalex":
                doi = graph.nodes[node].get('doi', '')
                if doi:
                    all_nodes.add(doi.lower())
                    # Collect metadata if not already present or if from preferred API
                    if doi.lower() not in node_metadata or (prefer_metadata and api_name == prefer_metadata):
                        node_metadata[doi.lower()] = {k: v for k, v in graph.nodes[node].items() if k != 'doi'}
            else:
                # For other APIs, the node is already the DOI
                all_nodes.add(node.lower())
                # Collect metadata if not already present or if from preferred API
                if node.lower() not in node_metadata or (prefer_metadata and api_name == prefer_metadata):
                    node_metadata[node.lower()] = dict(graph.nodes[node])
    
    print(f"Found {len(all_nodes)} unique DOIs across all APIs.")
    
    # Add all nodes to the merged graph with their metadata
    for doi in all_nodes:
        metadata = node_metadata.get(doi, {})
        if 'label' not in metadata:
            metadata['label'] = doi.split('/')[-1]
        merged_graph.add_node(doi, **metadata)
    
    # Now add all edges from all graphs
    edge_count = 0
    for graph, api_name in graphs:
        for u, v, data in graph.edges(data=True):
            # For OpenAlex, convert the OpenAlex IDs to DOIs
            if api_name == "openalex":
                u_doi = graph.nodes[u].get('doi', '').lower()
                v_doi = graph.nodes[v].get('doi', '').lower()
                if u_doi and v_doi and u_doi in merged_graph and v_doi in merged_graph:
                    # Only add if both nodes are in the merged graph and edge doesn't already exist
                    if not merged_graph.has_edge(u_doi, v_doi):
                        merged_graph.add_edge(u_doi, v_doi, type='references')
                        edge_count += 1
            else:
                # For other APIs, the nodes are already DOIs
                u_lower = u.lower()
                v_lower = v.lower()
                if u_lower in merged_graph and v_lower in merged_graph:
                    # Only add if both nodes are in the merged graph and edge doesn't already exist
                    if not merged_graph.has_edge(u_lower, v_lower):
                        merged_graph.add_edge(u_lower, v_lower, type='references')
                        edge_count += 1
    
    print(f"Added {edge_count} unique edges to the merged graph.")
    return merged_graph

def find_direct_connections(nodes, api_name="openalex"):
    """Find direct connections between nodes by looking for title/author matches in reference lists.
    
    This is a more direct approach that bypasses API limitations
    and instead looks at the actual reference text and bibliographic data for each paper.
    
    Args:
        nodes: List of node IDs (OpenAlex IDs or DOIs)
        api_name: API name to use ("openalex", "opencitations", "crossref", etc.)
    
    Returns:
        List of (source, target) edge pairs
    """
    # Skip if we have no nodes
    if not nodes:
        return []
    
    print(f"Looking for direct connections between {len(nodes)} papers...")
    
    # Get paper details with their full reference lists
    paper_details = {}
    node_doi_map = {}  # Map DOIs to node IDs
    
    for node in tqdm(nodes, desc="Getting paper details"):
        try:
            if api_name == "openalex":
                # Get the work details
                work_url = f"https://api.openalex.org/works/{node.split('/')[-1]}"
                data = make_api_request(work_url)
                
                if data:
                    # Store the DOI for mapping
                    if 'doi' in data and data['doi']:
                        node_doi_map[data['doi'].lower()] = node
                    
                    # Extract useful attributes
                    paper_details[node] = {
                        'title': data.get('title', '').lower(),
                        'authors': [auth.get('author', {}).get('display_name', '').lower() 
                                   for auth in data.get('authorships', [])],
                        'year': data.get('publication_year'),
                        'doi': data.get('doi', ''),
                        # Get the bibliographic information
                        'biblio': data.get('biblio', {}),
                        # Get the referenced works directly
                        'referenced_works': data.get('referenced_works', []),
                        # Get the cited_by count - papers with more citations are 
                        # more likely to be referenced by other papers in our dataset
                        'cited_by_count': data.get('cited_by_count', 0),
                        # Get the abstract for potential text matching
                        'abstract': data.get('abstract_inverted_index', {})
                    }
                    
                    # If we have a DOI, try OpenCitations as a backup for references
                    if data.get('doi') and not data.get('referenced_works'):
                        opencitations_refs = get_opencitations_references(data['doi'])
                        if opencitations_refs:
                            paper_details[node]['opencitations_refs'] = opencitations_refs
            
            elif api_name == "crossref" and node.startswith("10."):
                # For Crossref, the node is the DOI
                work_data = get_crossref_work(node)
                if work_data:
                    node_doi_map[node.lower()] = node
                    
                    paper_details[node] = {
                        'title': work_data.get('title', [''])[0].lower() if isinstance(work_data.get('title'), list) else '',
                        'authors': [],
                        'year': work_data.get('published-print', {}).get('date-parts', [['']])[0][0] if 'published-print' in work_data else '',
                        'doi': node,
                    }
                    
                    # Extract authors
                    if 'author' in work_data:
                        paper_details[node]['authors'] = [
                            author.get('family', '').lower() 
                            for author in work_data['author'] 
                            if 'family' in author
                        ]
                    
                    # Extract references
                    if 'reference' in work_data:
                        ref_dois = []
                        for ref in work_data['reference']:
                            if 'DOI' in ref:
                                ref_dois.append(ref['DOI'].lower())
                        
                        paper_details[node]['references'] = ref_dois
        
        except Exception as e:
            print(f"Error getting details for {node}: {e}")
    
    # Now look for direct connections
    print("Analyzing papers to find direct connections...")
    direct_connections = []
    
    # First, map any DOIs to node IDs
    doi_connections = []
    for source_node, source_details in paper_details.items():
        # If we have referenced_works in OpenAlex, use those directly
        if source_details.get('referenced_works'):
            for ref_id in source_details['referenced_works']:
                if ref_id in nodes:
                    direct_connections.append((source_node, ref_id))
        
        # If we have OpenCitations references (which are DOIs), map them to nodes
        if source_details.get('opencitations_refs'):
            for ref_doi in source_details['opencitations_refs']:
                ref_doi_lower = ref_doi.lower()
                # Try to find this DOI in our node_doi_map
                for node_doi, node_id in node_doi_map.items():
                    if ref_doi_lower == node_doi.lower():
                        doi_connections.append((source_node, node_id))
                        break
    
    # Add DOI connections to our list
    if doi_connections:
        print(f"Found {len(doi_connections)} connections via DOI matching")
        direct_connections.extend(doi_connections)
    
    # Next, try matching by content (title and authors)
    content_connections = []
    for source_node, source_details in tqdm(paper_details.items(), desc="Finding title/author matches"):
        source_title = source_details.get('title', '').lower()
        source_authors = [a.lower() for a in source_details.get('authors', []) if a]
        
        # Skip if we don't have good source data
        if not source_title or not source_authors:
            continue
        
        # Check each potential target
        for target_node, target_details in paper_details.items():
            # Skip self-references
            if source_node == target_node:
                continue
                
            target_title = target_details.get('title', '').lower()
            target_authors = [a.lower() for a in target_details.get('authors', []) if a]
            target_year = target_details.get('year')
            
            # Skip if we don't have good target data
            if not target_title or not target_authors:
                continue
                
            # Skip if target is newer than source (can't cite something that doesn't exist yet)
            if source_details.get('year') and target_year and int(source_details['year']) < int(target_year):
                continue
            
            # Check if the target title appears in the source biblio
            biblio = source_details.get('biblio', {})
            if biblio:
                biblio_str = ' '.join(str(val).lower() for val in biblio.values() if val)
                
                # Check if target title appears in the biblio text
                if target_title in biblio_str:
                    # Check if at least one author also appears
                    for author in target_authors:
                        if author and len(author) > 3 and author in biblio_str:
                            content_connections.append((source_node, target_node))
                            break
    
    # Add content-based connections
    if content_connections:
        print(f"Found {len(content_connections)} connections via title/author matching")
        direct_connections.extend(content_connections)
    
    # Remove duplicates (if any)
    direct_connections = list(set(direct_connections))
    
    print(f"Found a total of {len(direct_connections)} potential direct connections")
    return direct_connections

def export_graph_for_visualization(G, output_dir):
    """Export the graph to formats compatible with visualization tools like Gephi or Cytoscape.
    
    Args:
        G: NetworkX graph to export
        output_dir: Directory to save the exported files
    """
    # Create a copy of the graph for export to avoid modifying the original
    G_export = G.copy()
    
    # For GEXF export: Convert edge 'type' attribute to 'edge_type' to avoid reserved word issues
    # Also ensure all attributes are properly formatted for GEXF
    for u, v, data in G_export.edges(data=True):
        # Change 'type' attribute to 'edge_type' for GEXF compatibility
        if 'type' in data:
            data['edge_type'] = data['type']
            del data['type']
    
    # Export to GEXF format (for Gephi)
    gexf_file = os.path.join(output_dir, "citation_graph.gexf")
    print(f"Exporting graph to GEXF format for Gephi: {gexf_file}")
    nx.write_gexf(G_export, gexf_file)
    
    # For GraphML: No need to change attribute names
    # Reset the copy to retain original 'type' attribute for GraphML
    G_export = G.copy()
    
    # Export to GraphML format (for Cytoscape and others)
    graphml_file = os.path.join(output_dir, "citation_graph.graphml")
    print(f"Exporting graph to GraphML format for Cytoscape: {graphml_file}")
    nx.write_graphml(G_export, graphml_file)
    
    print("Graph exported successfully. You can now open these files in Gephi or Cytoscape for interactive exploration.")

#%% Main execution
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate a citation graph from a BibTeX file using citation APIs.')
    parser.add_argument('--bibtex', default="Exported Items.bib", help='Path to the BibTeX file (default: Exported Items.bib)')
    parser.add_argument('--cache', action='store_true', help='Use cached API IDs if available')
    parser.add_argument('--email', help='Email to use for API requests (helps with rate limits)')
    parser.add_argument('--output-dir', default="output", help='Directory to save output files (default: output)')
    parser.add_argument('--api', default="openalex", 
                      choices=["openalex", "crossref", "semanticscholar", "all"], 
                      help='API to use for citation data (default: openalex, "all" to use and merge all APIs)')
    parser.add_argument('--prefer-metadata', default="openalex",
                      choices=["openalex", "crossref", "semanticscholar"],
                      help='Preferred API for metadata when merging (default: openalex)')
    parser.add_argument('--find-intermediaries', action='store_true', help='Find intermediary papers to connect isolated nodes')
    parser.add_argument('--max-intermediaries', type=int, default=10, help='Maximum number of intermediary papers to check per isolated node')
    args = parser.parse_args()
    
    # Set email for API requests if provided
    if args.email:
        global_email = args.email
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set file paths
    bibtex_file = args.bibtex
    api_name = args.api
    cache_file = os.path.join(args.output_dir, f"{api_name}_ids_cache.json")
    graph_file = os.path.join(args.output_dir, "citation_graph.json")
    intermediary_graph_file = os.path.join(args.output_dir, "citation_graph_with_intermediaries.json")
    visualization_file = os.path.join(args.output_dir, "citation_graph.png")
    intermediary_visualization_file = os.path.join(args.output_dir, "citation_graph_with_intermediaries.png")
    report_file = os.path.join(args.output_dir, "citation_report.csv")
    
    print(f"Extracting DOIs from {bibtex_file}...")
    dois, doi_metadata = extract_dois_from_bibtex(bibtex_file)
    print(f"Found {len(dois)} DOIs.")
    
    # If using all APIs, build separate graphs and merge them
    if api_name == "all":
        graphs_to_merge = []
        
        # Build OpenAlex graph
        print("Building citation graph using OpenAlex...")
        try:
            openalex_cache = os.path.join(args.output_dir, "openalex_ids_cache.json")
            use_cache = False
            if args.cache and os.path.exists(openalex_cache):
                try:
                    with open(openalex_cache, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Check if cache contains the new format with references
                    if isinstance(cache_data, dict) and 'doi_to_openalex' in cache_data and 'references' in cache_data:
                        doi_to_openalex = cache_data['doi_to_openalex']
                        reference_pairs = cache_data['references']
                        print(f"Loaded {len(doi_to_openalex)} OpenAlex IDs and {len(reference_pairs)} reference relationships from cache.")
                        
                        # Create initial graph from cached data
                        openalex_graph = nx.DiGraph()
                        for doi, openalex_id in doi_to_openalex.items():
                            # Get metadata if available
                            metadata = doi_metadata.get(doi, {}) if doi_metadata else {}
                            # If label doesn't exist in metadata, create it
                            if 'label' not in metadata:
                                metadata['label'] = f"{doi.split('/')[-1]}"
                            openalex_graph.add_node(openalex_id, doi=doi, **metadata)
                        
                        # Add cached reference relationships
                        references_count = 0
                        for source, target in reference_pairs:
                            if source in openalex_graph and target in openalex_graph:  # Ensure both nodes are in the graph
                                openalex_graph.add_edge(source, target, type='references')
                                references_count += 1
                        
                        print(f"Added {references_count} reference relationships from cache.")
                        graphs_to_merge.append((openalex_graph, "openalex"))
                    else:
                        # Handle legacy cache format (only DOI to OpenAlex mappings)
                        doi_to_openalex = cache_data
                        print(f"Loaded {len(doi_to_openalex)} OpenAlex IDs from cache (legacy format).")
                        
                        # Create initial graph from cached data
                        openalex_graph = nx.DiGraph()
                        for doi, openalex_id in doi_to_openalex.items():
                            # Get metadata if available
                            metadata = doi_metadata.get(doi, {}) if doi_metadata else {}
                            # If label doesn't exist in metadata, create it
                            if 'label' not in metadata:
                                metadata['label'] = f"{doi.split('/')[-1]}"
                            openalex_graph.add_node(openalex_id, doi=doi, **metadata)
                        
                        # Add this code to debug potential connections
                        # Analyze how many DOIs are from the same journals or have the same authors
                        print("Analyzing potential connections between papers...")
                        # Get the list of DOIs in our graph
                        our_dois = list(doi_to_openalex.keys())
                        
                        # Debug: Check how many papers we should expect to have connections
                        potential_connections = 0
                        # First, get metadata for each paper to see which are from same journals
                        paper_metadata = {}
                        for doi, openalex_id in doi_to_openalex.items():
                            try:
                                # Get the paper's metadata
                                paper_url = f"https://api.openalex.org/works/{openalex_id.split('/')[-1]}"
                                data = make_api_request(paper_url)
                                if data:
                                    # Extract journal/venue info
                                    if 'primary_location' in data and 'source' in data['primary_location']:
                                        source = data['primary_location']['source']
                                        if source and 'id' in source:
                                            paper_metadata[doi] = {'source_id': source['id']}
                                    
                                    # Extract author IDs
                                    if 'authorships' in data:
                                        paper_metadata[doi]['author_ids'] = [
                                            auth.get('author', {}).get('id') 
                                            for auth in data['authorships'] 
                                            if auth.get('author', {}).get('id')
                                        ]
                                    
                                    # Extract publication year
                                    if 'publication_year' in data:
                                        paper_metadata[doi]['year'] = data['publication_year']
                            except Exception as e:
                                print(f"Error getting metadata for {doi}: {e}")
                        
                        # Check for papers from same sources/journals
                        sources = {}
                        for doi, metadata in paper_metadata.items():
                            if 'source_id' in metadata:
                                source_id = metadata['source_id']
                                if source_id in sources:
                                    sources[source_id].append(doi)
                                else:
                                    sources[source_id] = [doi]
                        
                        # Print source statistics
                        print(f"Found metadata for {len(paper_metadata)} papers")
                        print(f"Papers are from {len(sources)} different sources/journals")
                        
                        # Print sources with multiple papers (these are more likely to have connections)
                        sources_with_multiple = {k: v for k, v in sources.items() if len(v) > 1}
                        print(f"Found {len(sources_with_multiple)} sources with multiple papers:")
                        for source_id, dois in sources_with_multiple.items():
                            print(f"  Source {source_id}: {len(dois)} papers")
                            potential_connections += len(dois) * (len(dois) - 1) // 2  # n*(n-1)/2 potential connections
                        
                        # Check for papers with same authors
                        author_papers = {}
                        for doi, metadata in paper_metadata.items():
                            if 'author_ids' in metadata:
                                for author_id in metadata['author_ids']:
                                    if author_id in author_papers:
                                        author_papers[author_id].append(doi)
                                    else:
                                        author_papers[author_id] = [doi]
                        
                        # Print author statistics
                        authors_with_multiple = {k: v for k, v in author_papers.items() if len(v) > 1}
                        print(f"Found {len(authors_with_multiple)} authors with multiple papers:")
                        for author_id, dois in authors_with_multiple.items():
                            print(f"  Author {author_id}: {len(dois)} papers")
                            potential_connections += len(dois) * (len(dois) - 1) // 2  # More potential connections
                        
                        print(f"Based on common sources and authors, we should expect approximately {potential_connections} connections")
                        print("These are just potential connections - actual citation links depend on papers citing each other")
                        
                        # Continue with getting references for each paper
                        print("Getting references for each paper...")
                        references_count = 0
                        reference_pairs = []
                        for node in tqdm(openalex_graph.nodes()):
                            referenced_works = get_referenced_works(node)
                            
                            # Add edges for references
                            for ref_work in referenced_works:
                                if ref_work in openalex_graph:  # If the referenced work is in our graph
                                    openalex_graph.add_edge(node, ref_work, type='references')
                                    reference_pairs.append((node, ref_work))
                                    references_count += 1
                        
                        print(f"Found {references_count} reference relationships within the OpenAlex dataset.")
                        
                        # Save the updated cache with references
                        cache_data = {
                            'doi_to_openalex': doi_to_openalex,
                            'references': reference_pairs
                        }
                        with open(openalex_cache, 'w', encoding='utf-8') as f:
                            json.dump(cache_data, f, indent=2)
                        print(f"Updated cache with {len(reference_pairs)} reference relationships.")
                        graphs_to_merge.append((openalex_graph, "openalex"))
                except Exception as e:
                    print(f"Error loading OpenAlex cache: {e}")
                    print("Building OpenAlex citation graph from scratch...")
                    openalex_graph = build_citation_graph(dois, doi_metadata)
                    graphs_to_merge.append((openalex_graph, "openalex"))
                    
                    # Save OpenAlex IDs and reference relationships for caching
                    doi_to_openalex = {openalex_graph.nodes[node]['doi']: node for node in openalex_graph.nodes() 
                                     if 'doi' in openalex_graph.nodes[node]}
                    
                    # Extract reference relationships as pairs of OpenAlex IDs
                    reference_pairs = [(u, v) for u, v in openalex_graph.edges()]
                    
                    # Create new cache format with both mappings and references
                    cache_data = {
                        'doi_to_openalex': doi_to_openalex,
                        'references': reference_pairs
                    }
                    
                    with open(openalex_cache, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2)
                    print(f"Saved {len(doi_to_openalex)} OpenAlex IDs and {len(reference_pairs)} reference relationships to cache.")
            else:
                openalex_graph = build_citation_graph(dois, doi_metadata)
                graphs_to_merge.append((openalex_graph, "openalex"))
                
                # Save OpenAlex IDs and reference relationships for caching
                doi_to_openalex = {openalex_graph.nodes[node]['doi']: node for node in openalex_graph.nodes() 
                                if 'doi' in openalex_graph.nodes[node]}
                
                # Extract reference relationships as pairs of OpenAlex IDs
                reference_pairs = [(u, v) for u, v in openalex_graph.edges()]
                
                # Create new cache format with both mappings and references
                cache_data = {
                    'doi_to_openalex': doi_to_openalex,
                    'references': reference_pairs
                }
                
                with open(openalex_cache, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
                print(f"Saved {len(doi_to_openalex)} OpenAlex IDs and {len(reference_pairs)} reference relationships to cache.")
        except Exception as e:
            print(f"Error building OpenAlex graph: {e}")
        
        # Build Crossref graph
        print("Building citation graph using Crossref...")
        try:
            crossref_cache = os.path.join(args.output_dir, "crossref_ids_cache.json")
            if args.cache and os.path.exists(crossref_cache):
                try:
                    with open(crossref_cache, 'r') as f:
                        cached_graph_data = json.load(f)
                    print(f"Loaded cached Crossref graph data from {crossref_cache}")
                    
                    # Recreate graph from cached data
                    crossref_graph = nx.DiGraph()
                    
                    # Add nodes
                    for node_data in cached_graph_data['nodes']:
                        crossref_graph.add_node(node_data['id'], label=node_data.get('label', ''))
                    
                    # Add edges - keep only 'references' type edges for consistency
                    for edge_data in cached_graph_data['edges']:
                        if edge_data.get('type') == 'references':
                            crossref_graph.add_edge(edge_data['source'], edge_data['target'], type='references')
                    
                    print(f"Recreated Crossref graph with {len(crossref_graph.nodes)} nodes and {len(crossref_graph.edges)} edges.")
                    graphs_to_merge.append((crossref_graph, "crossref"))
                except Exception as e:
                    print(f"Error loading Crossref cache: {e}")
                    print("Building Crossref citation graph from scratch...")
                    crossref_graph = build_crossref_graph(dois, doi_metadata)
                    graphs_to_merge.append((crossref_graph, "crossref"))
                    
                    # Save graph data for caching
                    save_citation_graph(crossref_graph, crossref_cache)
                    print(f"Saved Crossref graph data to cache file: {crossref_cache}")
            else:
                crossref_graph = build_crossref_graph(dois, doi_metadata)
                graphs_to_merge.append((crossref_graph, "crossref"))
                
                # Save graph data for caching
                save_citation_graph(crossref_graph, crossref_cache)
                print(f"Saved Crossref graph data to cache file: {crossref_cache}")
        except Exception as e:
            print(f"Error building Crossref graph: {e}")
        
        # Build Semantic Scholar graph
        print("Building citation graph using Semantic Scholar...")
        try:
            semanticscholar_cache = os.path.join(args.output_dir, "semanticscholar_ids_cache.json")
            if args.cache and os.path.exists(semanticscholar_cache):
                try:
                    with open(semanticscholar_cache, 'r') as f:
                        cached_graph_data = json.load(f)
                    print(f"Loaded cached Semantic Scholar graph data from {semanticscholar_cache}")
                    
                    # Recreate graph from cached data
                    semanticscholar_graph = nx.DiGraph()
                    
                    # Add nodes
                    for node_data in cached_graph_data['nodes']:
                        semanticscholar_graph.add_node(node_data['id'], label=node_data.get('label', ''))
                    
                    # Add edges - keep only 'references' type edges for consistency
                    for edge_data in cached_graph_data['edges']:
                        if edge_data.get('type') == 'references':
                            semanticscholar_graph.add_edge(edge_data['source'], edge_data['target'], type='references')
                    
                    print(f"Recreated Semantic Scholar graph with {len(semanticscholar_graph.nodes)} nodes and {len(semanticscholar_graph.edges)} edges.")
                    graphs_to_merge.append((semanticscholar_graph, "semanticscholar"))
                except Exception as e:
                    print(f"Error loading Semantic Scholar cache: {e}")
                    print("Building Semantic Scholar citation graph from scratch...")
                    semanticscholar_graph = build_semanticscholar_graph(dois, doi_metadata)
                    graphs_to_merge.append((semanticscholar_graph, "semanticscholar"))
                    
                    # Save graph data for caching
                    save_citation_graph(semanticscholar_graph, semanticscholar_cache)
                    print(f"Saved Semantic Scholar graph data to cache file: {semanticscholar_cache}")
            else:
                semanticscholar_graph = build_semanticscholar_graph(dois, doi_metadata)
                graphs_to_merge.append((semanticscholar_graph, "semanticscholar"))
                
                # Save graph data for caching
                save_citation_graph(semanticscholar_graph, semanticscholar_cache)
                print(f"Saved Semantic Scholar graph data to cache file: {semanticscholar_cache}")
        except Exception as e:
            print(f"Error building Semantic Scholar graph: {e}")
                
        # Merge all graphs
        print("Merging citation graphs from all APIs...")
        citation_graph = merge_citation_graphs(graphs_to_merge, prefer_metadata=args.prefer_metadata)
            
    elif api_name == "semanticscholar":
        # Handle Semantic Scholar API
        if args.cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_graph_data = json.load(f)
                print(f"Loaded cached graph data from {cache_file}")
                
                # Recreate graph from cached data
                citation_graph = nx.DiGraph()
                
                # Add nodes
                for node_data in cached_graph_data['nodes']:
                    citation_graph.add_node(node_data['id'], label=node_data.get('label', ''))
                
                # Add edges - keep only 'references' type edges for consistency
                for edge_data in cached_graph_data['edges']:
                    if edge_data.get('type') == 'references':
                        citation_graph.add_edge(edge_data['source'], edge_data['target'], type='references')
                
                print(f"Recreated graph with {len(citation_graph.nodes)} nodes and {len(citation_graph.edges)} edges.")
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Building citation graph using Semantic Scholar...")
                citation_graph = build_semanticscholar_graph(dois, doi_metadata)
                
                # Save graph data for caching
                save_citation_graph(citation_graph, cache_file)
                print(f"Saved graph data to cache file: {cache_file}")
        else:
            print("Building citation graph using Semantic Scholar...")
            citation_graph = build_semanticscholar_graph(dois, doi_metadata)
            
            # Save graph data for caching
            save_citation_graph(citation_graph, cache_file)
            print(f"Saved graph data to cache file: {cache_file}")
    else:
        # Continue with existing code for OpenAlex, OpenCitations, and Crossref
        if api_name == "crossref":
            # For Crossref, we use DOIs directly
            if args.cache and os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_graph_data = json.load(f)
                    print(f"Loaded cached graph data from {cache_file}")
                    
                    # Recreate graph from cached data
                    citation_graph = nx.DiGraph()
                    
                    # Add nodes
                    for node_data in cached_graph_data['nodes']:
                        citation_graph.add_node(node_data['id'], label=node_data.get('label', ''))
                    
                    # Add edges - keep only 'references' type edges for consistency
                    for edge_data in cached_graph_data['edges']:
                        if edge_data.get('type') == 'references':
                            citation_graph.add_edge(edge_data['source'], edge_data['target'], type='references')
                    
                    print(f"Recreated graph with {len(citation_graph.nodes)} nodes and {len(citation_graph.edges)} edges.")
                except Exception as e:
                    print(f"Error loading cache: {e}")
                    print("Building citation graph using Crossref...")
                    citation_graph = build_crossref_graph(dois, doi_metadata)
                    
                    # Save graph data for caching
                    save_citation_graph(citation_graph, cache_file)
                    print(f"Saved graph data to cache file: {cache_file}")
            else:
                print("Building citation graph using Crossref...")
                citation_graph = build_crossref_graph(dois, doi_metadata)
                
                # Save graph data for caching
                save_citation_graph(citation_graph, cache_file)
                print(f"Saved graph data to cache file: {cache_file}")
        else:  # OpenAlex API (default)
            # Use cached OpenAlex IDs if available and requested
            use_cache = False
            if args.cache and os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Check if cache contains the new format with references
                    if isinstance(cache_data, dict) and 'doi_to_openalex' in cache_data and 'references' in cache_data:
                        doi_to_openalex = cache_data['doi_to_openalex']
                        reference_pairs = cache_data['references']
                        print(f"Loaded {len(doi_to_openalex)} OpenAlex IDs and {len(reference_pairs)} reference relationships from cache.")
                        
                        # Create initial graph from cached data
                        G = nx.DiGraph()
                        for doi, openalex_id in doi_to_openalex.items():
                            # Get metadata if available
                            metadata = doi_metadata.get(doi, {}) if doi_metadata else {}
                            # If label doesn't exist in metadata, create it
                            if 'label' not in metadata:
                                metadata['label'] = f"{doi.split('/')[-1]}"
                            G.add_node(openalex_id, doi=doi, **metadata)
                        
                        # Add cached reference relationships
                        references_count = 0
                        for source, target in reference_pairs:
                            if source in G and target in G:  # Ensure both nodes are in the graph
                                G.add_edge(source, target, type='references')
                                references_count += 1
                        
                        print(f"Added {references_count} reference relationships from cache.")
                        use_cache = True
                        citation_graph = G
                    else:
                        # Handle legacy cache format (only DOI to OpenAlex mappings)
                        doi_to_openalex = cache_data
                        print(f"Loaded {len(doi_to_openalex)} OpenAlex IDs from cache (legacy format).")
                        
                        # Create initial graph from cached data
                        G = nx.DiGraph()
                        for doi, openalex_id in doi_to_openalex.items():
                            # Get metadata if available
                            metadata = doi_metadata.get(doi, {}) if doi_metadata else {}
                            # If label doesn't exist in metadata, create it
                            if 'label' not in metadata:
                                metadata['label'] = f"{doi.split('/')[-1]}"
                            G.add_node(openalex_id, doi=doi, **metadata)
                        
                        use_cache = True
                except Exception as e:
                    print(f"Error loading cache: {e}")
                    use_cache = False
            
            if not use_cache:
                print("Building citation graph using OpenAlex...")
                citation_graph = build_citation_graph(dois, doi_metadata)
                
                # Save OpenAlex IDs and reference relationships for caching
                doi_to_openalex = {citation_graph.nodes[node]['doi']: node for node in citation_graph.nodes() 
                                   if 'doi' in citation_graph.nodes[node]}
                
                # Extract reference relationships as pairs of OpenAlex IDs
                reference_pairs = [(u, v) for u, v in citation_graph.edges()]
                
                # Create new cache format with both mappings and references
                cache_data = {
                    'doi_to_openalex': doi_to_openalex,
                    'references': reference_pairs
                }
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
                print(f"Saved {len(doi_to_openalex)} OpenAlex IDs and {len(reference_pairs)} reference relationships to cache.")
            elif use_cache and not 'references' in locals():
                # If using legacy cache format, still need to get references
                print("Building citation graph from cached OpenAlex IDs...")
                citation_graph = G
                
                # Add this line to try direct connections when the analysis is done
                print("Trying to find direct connections between papers...")
                reference_pairs = []  # Initialize before using
                direct_connections = find_direct_connections(list(citation_graph.nodes()), api_name="openalex")
                
                # If we found direct connections, add them to the graph
                if direct_connections:
                    print(f"Adding {len(direct_connections)} direct connections to the graph")
                    direct_connection_count = 0
                    for source, target in direct_connections:
                        if source in citation_graph and target in citation_graph:
                            citation_graph.add_edge(source, target, type='references')
                            reference_pairs.append((source, target))
                            direct_connection_count += 1
                    print(f"Added {direct_connection_count} direct connections to the graph")
                
                # Continue with normal reference fetching
                print("Getting references for each paper...")
                references_count = 0
                reference_pairs = []
                for node in tqdm(citation_graph.nodes()):
                    referenced_works = get_referenced_works(node)
                    
                    # Add edges for references
                    for ref_work in referenced_works:
                        if ref_work in citation_graph:  # If the referenced work is in our graph
                            citation_graph.add_edge(node, ref_work, type='references')
                            reference_pairs.append((node, ref_work))
                            references_count += 1
                
                print(f"Found {references_count} reference relationships within the dataset.")
                
                # Save the updated cache with references
                cache_data = {
                    'doi_to_openalex': doi_to_openalex,
                    'references': reference_pairs
                }
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
                print(f"Updated cache with {len(reference_pairs)} reference relationships.")
    
    print(f"Graph has {len(citation_graph.nodes)} nodes and {len(citation_graph.edges)} edges.")
    
    # Find intermediary papers if requested
    if args.find_intermediaries:
        print("Finding intermediary papers to connect isolated nodes...")
        extended_graph = find_intermediary_papers(citation_graph, api_name if api_name != "all" else "crossref", max_papers_per_node=args.max_intermediaries)
        print(f"Extended graph has {len(extended_graph.nodes)} nodes and {len(extended_graph.edges)} edges.")
        
        # Save extended graph
        save_citation_graph(extended_graph, intermediary_graph_file)
        
        # Visualize extended graph
        print("Visualizing extended citation graph...")
        visualize_citation_graph(extended_graph, intermediary_visualization_file)
    
    print("Saving citation graph...")
    save_citation_graph(citation_graph, graph_file)
    
    print("Visualizing citation graph...")
    visualize_citation_graph(citation_graph, visualization_file)
    
    print("Generating citation report...")
    generate_citation_report(citation_graph, report_file)
    
    print("Exporting graph for visualization tools...")
    export_graph_for_visualization(citation_graph, args.output_dir)
    
    # If we found intermediary papers, also export the extended graph
    if args.find_intermediaries:
        print("Exporting extended graph for visualization tools...")
        export_graph_for_visualization(extended_graph, args.output_dir)
    
    print(f"Done! Output files are in {args.output_dir}/")
# %%
