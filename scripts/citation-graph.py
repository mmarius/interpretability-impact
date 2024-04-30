import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from itertools import count
import requests
import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm
from utils import get_citation_details, bulk_get_paper_details, get_reference_details, get_abstract
from classifier import is_interpretability_title_and_abstract, is_mt_title_and_abstract

df = pd.read_csv("./data/cl_papers.csv", index_col=0)
df = df.dropna(subset=['semantic_scholar_id'])

# we'll consider the gold label from the conference,
# or the classifier's prediction as a fallback

# we now create the graph
G = nx.DiGraph()


# we first add all the papers in our database
semantic_scholar_papers = bulk_get_paper_details(list(df['semantic_scholar_id']))
for (index, row), paper in zip(df.iterrows(), semantic_scholar_papers):
    if paper is None:
        print(f"Paper {row['doi']} was not found in the semantic scholar API")
        continue
    year = paper.year
    citation_count = paper.citation_count
    influential_citation_count = paper.influential_citation_count
    venue = paper.venue
    paper_id = paper.paper_id
    title = row['title']
    abstract = row['abstract']
    interpretability = is_interpretability_title_and_abstract(title, abstract)
    mt = is_mt_title_and_abstract(title, abstract)

    G.add_node(
        paper_id,
        year=year,
        citation_count=citation_count,
        influential_citation_count=influential_citation_count,
        venue=venue,
        originally_from_dataset=True,
        area=row['area'],
        interpreability_prediction=interpretability,
        mt_prediction=mt
    )


# uncomment this cell to do an actual run
import random
SAMPLE_SIZE = 5
random_nodes = random.sample(list(G.nodes), SAMPLE_SIZE)
G = nx.Graph(G.subgraph(random_nodes))



# for each node in our graph (i.e., a paper), get all the papers that cite it
# NOTE: running this will take some time, and it mostly depends on the
# quality of the internet connection
nodes = dict(G.nodes.data())
for nid, attributes in tqdm(nodes.items(), desc="retrieving citations", total=len(nodes)):
    citing_papers = get_citation_details(nid, include_abstract=True)
    for paper in citing_papers:
        title = paper.title
        abstract = paper.abstract

        if abstract is None and paper.doi is not None:
            doi = paper.doi
            abstract = get_abstract(doi)
            if abstract is not None:
                print('abstract found for', title)

        if abstract is not None:
            interpretability = is_interpretability_title_and_abstract(title, abstract)
            mt = is_mt_title_and_abstract(title, abstract)
        else:
            interpretability = None
            mt = None

        year = paper.year
        citation_count = paper.citation_count
        influential_citation_count = paper.influential_citation_count
        venue = paper.venue
        paper_id = paper.paper_id
        is_influential_citation = paper.is_influential_citation

        if paper_id is None:
            print('paper with no id:', title)
            continue
        if paper_id not in G:
            G.add_node(
                paper_id,
                year=year,
                citation_count=citation_count,
                influential_citation_count=influential_citation_count,
                venue=venue,
                interpretability_prediction=interpretability,
                mt_prediction=mt,
                originally_from_dataset=False
            )
        G.add_edge(
            nid,
            paper_id,
            is_influential=is_influential_citation,
            result=paper.cites_result,
            methodology=paper.cites_methodology,
            background=paper.cites_background,
        )
    cited_papers = get_reference_details(nid, include_abstract=True)
    for paper in cited_papers:
        title = paper.title
        abstract = paper.abstract
        if abstract is None:
            doi = paper.doi
            abstract = get_abstract(doi)

        if abstract is not None:
            interpretability = is_interpretability_title_and_abstract(title, abstract)
            mt = is_mt_title_and_abstract(title, abstract)
        else:
            interpretability = None
            mt = None


        year = paper.year
        citation_count = paper.citation_count
        influential_citation_count = paper.influential_citation_count
        venue = paper.venue
        paper_id = paper.paper_id
        is_influential_citation = paper.is_influential_citation

        if paper_id is None:
            print('paper with no id:', title)
            continue
        if paper_id not in G:
            G.add_node(
                paper_id,
                year=year,
                citation_count=citation_count,
                influential_citation_count=influential_citation_count,
                venue=venue,
                interpretability_prediction=interpretability,
                mt_prediction=mt,
                originally_from_dataset=False
            )
        G.add_edge(
            nid,
            paper_id,
            is_influential=is_influential_citation,
            result=paper.cites_result,
            methodology=paper.cites_methodology,
            background=paper.cites_background,
        )

# we save the graph
G_json = nx.cytoscape_data(G)
with open('./citationgraph/graph.test.json', 'w') as f:
    json.dump(G_json, f)
