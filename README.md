# Citation Graph Generator

This script generates a citation graph from a BibTeX file containing DOIs. It uses either the OpenAlex API, OpenCitations API, or Crossref API to retrieve citation data and visualize the relationships between papers.

## Features

- Extracts DOIs from a BibTeX file
- Retrieves citation information from multiple API sources (OpenAlex, OpenCitations, or Crossref)
- Builds a graph representing citation relationships
- Visualizes the citation network
- Generates a citation report with statistics
- Caches API responses to avoid redundant API calls
- Robust error handling with retries and exponential backoff
- Command line arguments for flexible usage
- Community detection and analysis of citation networks

## Requirements

- Python 3.7+
- Packages listed in `requirements.txt`
- `datasci` conda environment

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the script with default settings (uses OpenAlex API):

```bash
conda activate datasci
python 1_citation_graph.py
```

### Choosing a Citation API

You can choose between three different citation APIs:

```bash
# Use OpenAlex API (default)
python 1_citation_graph.py --api openalex

# Use OpenCitations API
python 1_citation_graph.py --api opencitations

# Use Crossref API
python 1_citation_graph.py --api crossref

# Use all APIs together for maximum coverage
python 1_citation_graph.py --api all
```

### Advanced Usage

```bash
python 1_citation_graph.py --bibtex "path/to/your/bibliography.bib" --cache --email "your.email@example.com" --output-dir "results" --api crossref
```

Example with a specific bibliography file:

```bash
python 1_citation_graph.py --bibtex "displacement review.bib" --api all
```

### Community Detection

After generating the citation graph, you can run community detection and analysis:

```bash
conda activate datasci
python 2_community_detection.py
```

This will:
- Load the citation graph from `output/citation_graph.graphml`
- Detect communities using both Louvain and Leiden algorithms
- Generate visualizations of the communities
- Analyze community structures and metrics
- Export community assignments to CSV

### Command Line Arguments

- `--bibtex`: Path to your BibTeX file (default: "Exported Items.bib")
- `--cache`: Use cached API data if available (saves and reuses API calls)
- `--email`: Your email address for API requests (recommended for better service)
- `--output-dir`: Directory to save output files (default: "output")
- `--api`: API to use for citation data - "openalex", "opencitations", "crossref", or "all" (default: "openalex")

## Output Files

The script generates the following files in the output directory:

### Citation Graph Files
- Cache files (depending on API choice):
  - `openalex_ids_cache.json`
  - `opencitations_ids_cache.json`
  - `crossref_ids_cache.json`
- `citation_graph.json`: A JSON file containing the graph data
- `citation_graph.png`: A visualization of the citation network
- `citation_report.csv`: A CSV file with citation statistics for each paper
- `citation_graph.graphml`: A GraphML file for use with community detection

### Community Detection Files
- `louvain_communities.png`: Visualization of communities detected with Louvain algorithm
- `leiden_communities.png`: Visualization of communities detected with Leiden algorithm
- `louvain_metrics.png`: Metrics analysis of Louvain communities
- `leiden_metrics.png`: Metrics analysis of Leiden communities
- `community_assignments.csv`: CSV file with community assignments for each paper

## How It Works

1. The script extracts DOIs from the BibTeX file
2. Depending on the chosen API:
   - **OpenAlex**: Maps DOIs to OpenAlex IDs and uses them to find references and citations
   - **OpenCitations**: Uses DOIs directly to find reference and citation relationships
   - **Crossref**: Uses DOIs to lookup works and extract their reference lists
   - **All**: Combines results from all three APIs for maximum coverage
3. It creates a directed graph where:
   - Nodes represent papers
   - Edges represent citation relationships (colored by type)
4. The results are saved and visualized
5. Optionally, community detection algorithms analyze the citation network structure

## Community Detection

The community detection script (`2_community_detection.py`) provides:

- Detection of topical communities in the citation network
- Two different algorithms (Louvain and Leiden) for comparison
- Detailed analysis of each community:
  - Most frequent authors
  - Year distribution
  - Citation patterns (internal vs. external)
  - Network metrics (density, clustering coefficient)
- Visualizations of communities with color coding
- Comparison between algorithm results

## API Comparison

### OpenAlex
- Advantages: Larger database, more metadata available, good coverage
- Disadvantages: May require more API calls (DOI to ID mapping)

### OpenCitations
- Advantages: Direct DOI-to-DOI citation relationships, simpler API
- Disadvantages: May have less coverage than OpenAlex for some disciplines

### Crossref
- Advantages: Comprehensive reference lists, primary source of DOI registration
- Disadvantages: Often missing DOIs for references, primarily provides references but not citing works

## Error Handling

The script includes robust error handling:
- Retries API requests with exponential backoff
- Caches intermediate results to avoid unnecessary API calls
- Provides detailed error messages for debugging

## Notes

- Using the `datasci` conda environment is recommended (`conda activate datasci`)
- Including your email with API requests helps provide better service and may give access to better API rate limits
- For large BibTeX files, using the `--cache` option is recommended to avoid repeating API requests
- The script uses rate limiting to be respectful to the APIs
- Different APIs may give different results based on their coverage - try using `--api all` for the most comprehensive results

## API References

- [OpenAlex API](https://docs.openalex.org/): An open scientific knowledge graph with structured data about scientific publications
- [OpenCitations API](https://opencitations.net/index/coci/api/v1): An open index of DOI-to-DOI citation links derived from Crossref's open reference lists
- [Crossref API](https://github.com/CrossRef/rest-api-doc): The official API for Crossref, the registration agency for scholarly publication DOIs 