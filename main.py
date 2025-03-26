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

# Global variable for email
global_email = "anonymous@example.com"

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
        # Properly format the URL for the references endpoint
        url = f"https://api.openalex.org/works/{openalex_id.split('/')[-1]}/referenced_works"
        
        data = make_api_request(url)
        if data:
            return [ref.get('id') for ref in data.get('results', [])]
        else:
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

def get_opencitations_references(doi):
    """Get references (works cited by this paper) from OpenCitations."""
    if not doi:
        return []
    
    try:
        # Properly encode the DOI for the URL
        encoded_doi = quote(doi, safe='')
        url = f"https://opencitations.net/index/coci/api/v1/references/{encoded_doi}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Extract the DOIs of the references
            return [item.get('cited') for item in data if item.get('cited')]
        else:
            print(f"Error fetching references from OpenCitations for DOI {doi}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception when fetching references from OpenCitations for DOI {doi}: {e}")
        return []

def get_opencitations_citations(doi):
    """Get citations (works that cite this paper) from OpenCitations."""
    if not doi:
        return []
    
    try:
        # Properly encode the DOI for the URL
        encoded_doi = quote(doi, safe='')
        url = f"https://opencitations.net/index/coci/api/v1/citations/{encoded_doi}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Extract the DOIs of the citing works
            return [item.get('citing') for item in data if item.get('citing')]
        else:
            print(f"Error fetching citations from OpenCitations for DOI {doi}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception when fetching citations from OpenCitations for DOI {doi}: {e}")
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

def build_opencitations_graph(dois, doi_metadata=None, delay_range=(1, 2)):
    """Build a citation graph from a list of DOIs using OpenCitations API.
    
    Args:
        dois: List of DOIs to process
        doi_metadata: Dictionary of metadata for DOIs from BibTeX
        delay_range: Tuple of (min, max) delay between API calls in seconds
    """
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes for all DOIs
    print("Adding nodes for all DOIs...")
    for doi in tqdm(dois):
        # Get metadata for the node
        metadata = doi_metadata.get(doi, {}) if doi_metadata else {}
        
        # If label doesn't exist in metadata, create it
        if 'label' not in metadata:
            metadata['label'] = doi.split('/')[-1]
        
        G.add_node(doi, **metadata)
    
    # Get references for each paper
    print("Getting references for each paper...")
    references_count = 0
    for doi in tqdm(dois):
        # Get works referenced by this paper
        referenced_dois = get_opencitations_references(doi)
        
        # Add edges for references - paper A references paper B (A -> B)
        for ref_doi in referenced_dois:
            if ref_doi in G:  # If the referenced work is in our graph
                G.add_edge(doi, ref_doi, type='references')
                references_count += 1
        
        # Be gentle with the API
        time.sleep(random.uniform(*delay_range))
    
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
        api_name: The API to use (openalex, opencitations, or crossref)
        max_depth: Maximum depth to search for connections (default: 1)
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
            # For OpenCitations and Crossref, the node is the DOI
            doi = node
        
        if not doi:
            continue
        
        # Get papers that cite this paper
        citing_papers = []
        if api_name == "openalex":
            citing_papers = get_cited_by_works(node)
        elif api_name == "opencitations":
            citing_papers = get_opencitations_citations(doi)
        elif api_name == "crossref":
            # Crossref doesn't have a direct API for citations, so we skip this for Crossref
            pass
        
        # Get papers cited by this paper
        cited_papers = []
        if api_name == "openalex":
            cited_papers = get_referenced_works(node)
        elif api_name == "opencitations":
            cited_papers = get_opencitations_references(doi)
        elif api_name == "crossref":
            cited_papers = get_crossref_references(doi)
        
        # Combine all potential intermediary papers
        potential_intermediaries = citing_papers + cited_papers
        
        # Limit to prevent too many API calls
        if len(potential_intermediaries) > max_papers_per_node:
            potential_intermediaries = potential_intermediaries[:max_papers_per_node]
        
        papers_added = False
        
        # For each potential intermediary
        for intermediary in potential_intermediaries:
            # Skip if already in graph
            if intermediary in G_extended:
                continue
            
            # Get cited papers for this intermediary
            intermediary_cited = []
            intermediary_doi = None
            
            if api_name == "openalex":
                intermediary_cited = get_referenced_works(intermediary)
                # Get DOI for the intermediary paper
                # Extract the OpenAlex ID number
                openalex_id_number = intermediary.split('/')[-1]
                url = f"https://api.openalex.org/works/{openalex_id_number}"
                data = make_api_request(url)
                if data and 'doi' in data:
                    intermediary_doi = data['doi']
            elif api_name == "opencitations":
                intermediary_doi = intermediary
                intermediary_cited = get_opencitations_references(intermediary)
            elif api_name == "crossref":
                intermediary_doi = intermediary
                intermediary_cited = get_crossref_references(intermediary)
            
            # Check if this intermediary connects to any other node in the graph
            connections_found = False
            for cited in intermediary_cited:
                if cited in G.nodes() and cited != node:
                    connections_found = True
                    break
            
            if connections_found:
                # Add the intermediary node to the graph
                if intermediary_doi:
                    label = f"Intermediary: {intermediary_doi.split('/')[-1]}"
                    G_extended.add_node(intermediary, label=label, is_intermediary=True, doi=intermediary_doi)
                    
                    # Add edges
                    if api_name == "openalex":
                        # Add edge from node to intermediary if node cites intermediary
                        if intermediary in cited_papers:
                            G_extended.add_edge(node, intermediary, type='references')
                        # Add edge from intermediary to node if intermediary cites node
                        if intermediary in citing_papers:
                            G_extended.add_edge(intermediary, node, type='references')
                    else:
                        # For other APIs, use similar logic but with DOIs
                        if intermediary in cited_papers:
                            G_extended.add_edge(doi, intermediary, type='references')
                        if intermediary in citing_papers:
                            G_extended.add_edge(intermediary, doi, type='references')
                    
                    # Add edges to other nodes in the graph
                    for cited in intermediary_cited:
                        if cited in G.nodes() and cited != node:
                            G_extended.add_edge(intermediary, cited, type='references')
                    
                    papers_added = True
                    intermediary_connections += 1
                    break  # Only add one intermediary per isolated node to keep the graph clean
            
            # Add a small delay to be nice to the API
            time.sleep(random.uniform(1, 2))
        
        if not papers_added:
            print(f"Could not find intermediary papers for node {node}")
    
    print(f"Added {intermediary_connections} intermediary papers to connect isolated nodes.")
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

#%% Main execution
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate a citation graph from a BibTeX file using citation APIs.')
    parser.add_argument('--bibtex', default="Exported Items.bib", help='Path to the BibTeX file (default: Exported Items.bib)')
    parser.add_argument('--cache', action='store_true', help='Use cached API IDs if available')
    parser.add_argument('--email', help='Email to use for API requests (helps with rate limits)')
    parser.add_argument('--output-dir', default="output", help='Directory to save output files (default: output)')
    parser.add_argument('--api', default="openalex", 
                      choices=["openalex", "opencitations", "crossref", "semanticscholar", "all"], 
                      help='API to use for citation data (default: openalex, "all" to use and merge all APIs)')
    parser.add_argument('--prefer-metadata', default="openalex",
                      choices=["openalex", "opencitations", "crossref", "semanticscholar"],
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
                        doi_to_openalex = json.load(f)
                    print(f"Loaded {len(doi_to_openalex)} OpenAlex IDs from cache.")
                    
                    # Create initial graph from cached data
                    openalex_graph = nx.DiGraph()
                    for doi, openalex_id in doi_to_openalex.items():
                        # Get metadata if available
                        metadata = doi_metadata.get(doi, {}) if doi_metadata else {}
                        # If label doesn't exist in metadata, create it
                        if 'label' not in metadata:
                            metadata['label'] = f"{doi.split('/')[-1]}"
                        openalex_graph.add_node(openalex_id, doi=doi, **metadata)
                    
                    # Get references for each paper
                    print("Getting references for each paper from OpenAlex...")
                    references_count = 0
                    for node in tqdm(openalex_graph.nodes()):
                        referenced_works = get_referenced_works(node)
                        
                        # Add edges for references
                        for ref_work in referenced_works:
                            if ref_work in openalex_graph:  # If the referenced work is in our graph
                                openalex_graph.add_edge(node, ref_work, type='references')
                                references_count += 1
                    
                    print(f"Found {references_count} reference relationships within the OpenAlex dataset.")
                    graphs_to_merge.append((openalex_graph, "openalex"))
                except Exception as e:
                    print(f"Error loading OpenAlex cache: {e}")
                    print("Building OpenAlex citation graph from scratch...")
                    openalex_graph = build_citation_graph(dois, doi_metadata)
                    graphs_to_merge.append((openalex_graph, "openalex"))
                    
                    # Save OpenAlex IDs for caching
                    doi_to_openalex = {openalex_graph.nodes[node]['doi']: node for node in openalex_graph.nodes() 
                                    if 'doi' in openalex_graph.nodes[node]}
                    with open(openalex_cache, 'w', encoding='utf-8') as f:
                        json.dump(doi_to_openalex, f, indent=2)
                    print(f"Saved {len(doi_to_openalex)} OpenAlex IDs to cache.")
            else:
                openalex_graph = build_citation_graph(dois, doi_metadata)
                graphs_to_merge.append((openalex_graph, "openalex"))
                
                # Save OpenAlex IDs for caching
                doi_to_openalex = {openalex_graph.nodes[node]['doi']: node for node in openalex_graph.nodes() 
                                if 'doi' in openalex_graph.nodes[node]}
                with open(openalex_cache, 'w', encoding='utf-8') as f:
                    json.dump(doi_to_openalex, f, indent=2)
                print(f"Saved {len(doi_to_openalex)} OpenAlex IDs to cache.")
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
        if api_name == "opencitations":
            # For OpenCitations, we don't need to map DOIs to anything
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
                    print("Building citation graph using OpenCitations...")
                    citation_graph = build_opencitations_graph(dois, doi_metadata)
                    
                    # Save graph data for caching
                    save_citation_graph(citation_graph, cache_file)
                    print(f"Saved graph data to cache file: {cache_file}")
            else:
                print("Building citation graph using OpenCitations...")
                citation_graph = build_opencitations_graph(dois, doi_metadata)
                
                # Save graph data for caching
                save_citation_graph(citation_graph, cache_file)
                print(f"Saved graph data to cache file: {cache_file}")
        elif api_name == "crossref":
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
                        doi_to_openalex = json.load(f)
                    print(f"Loaded {len(doi_to_openalex)} OpenAlex IDs from cache.")
                    
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
                
                # Save OpenAlex IDs for caching
                doi_to_openalex = {citation_graph.nodes[node]['doi']: node for node in citation_graph.nodes() 
                                   if 'doi' in citation_graph.nodes[node]}
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(doi_to_openalex, f, indent=2)
                print(f"Saved {len(doi_to_openalex)} OpenAlex IDs to cache.")
            else:
                print("Building citation graph from cached OpenAlex IDs...")
                citation_graph = G
                
                # Get references for each paper
                print("Getting references for each paper...")
                references_count = 0
                for node in tqdm(citation_graph.nodes()):
                    referenced_works = get_referenced_works(node)
                    
                    # Add edges for references
                    for ref_work in referenced_works:
                        if ref_work in citation_graph:  # If the referenced work is in our graph
                            citation_graph.add_edge(node, ref_work, type='references')
                            references_count += 1
                
                print(f"Found {references_count} reference relationships within the dataset.")
                
                # We'll skip getting citations separately since they're just the reverse of references
                # This avoids creating duplicate edges in opposite directions
    
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
    
    print(f"Done! Output files are in {args.output_dir}/")
# %%
