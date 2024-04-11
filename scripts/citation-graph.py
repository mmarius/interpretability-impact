import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from itertools import count
import requests
import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm
from utils import get_citation_details, bulk_get_paper_details, get_reference_details

df = pd.read_csv("./data/clean_data.csv", index_col=0)

# we'll consider the gold label from the conference,
# or the classifier's prediction as a fallback

df['interpretability_boolean'] = df.apply(
    lambda row: row['interpretability'] if not pd.isna(row['interpretability']) else row['classifier_interpretability_prediction'],
    axis=1
)


# we now create the graph
G = nx.DiGraph()


# we first add all the papers in our database
semantic_scholar_papers = bulk_get_paper_details(list(df['doi']))
for (index, row), paper in zip(df.iterrows(), semantic_scholar_papers):
    if paper is None:
        print(f"Paper {row['doi']} was not found in the semantic scholar API")
        continue
    year = paper.year
    citation_count = paper.citation_count
    influential_citation_count = paper.influential_citation_count
    venue = paper.venue
    paper_id = paper.paper_id
    interpretability_boolean = row['interpretability_boolean']
    G.add_node(
        paper_id,
        year=year,
        citation_count=citation_count,
        influential_citation_count=influential_citation_count,
        venue=venue,
        interpretability_boolean=interpretability_boolean,
        originally_from_dataset=True
    )


# uncomment this cell to do an actual run
# import random
# SAMPLE_SIZE = 5
# random_nodes = random.sample(list(G.nodes), SAMPLE_SIZE)
# G = nx.Graph(G.subgraph(random_nodes))



from classifier import is_interpretability_title_and_abstract

# for each node in our graph (i.e., a paper), get all the papers that cite it
# NOTE: running this will take some time, and it mostly depends on the
# quality of the internet connection
nodes = dict(G.nodes.data())
for nid, attributes in tqdm(nodes.items(), desc="retrieving citations", total=len(nodes)):
    citing_papers = get_citation_details(nid, include_abstract=True)
    for paper in citing_papers:
        title = paper.title
        abstract = paper.abstract
        if abstract is not None:
            interpretability_boolean = is_interpretability_title_and_abstract(title, abstract)
        else:
            interpretability_boolean = None

        year = paper.year
        citation_count = paper.citation_count
        influential_citation_count = paper.influential_citation_count
        venue = paper.venue
        paper_id = paper.paper_id

        if paper_id is None:
            print('paper with no id:', title)
            continue
        G.add_node(
            paper_id,
            year=year,
            citation_count=citation_count,
            influential_citation_count=influential_citation_count,
            venue=venue,
            interpretability_boolean=interpretability_boolean,
            originally_from_dataset=False
        )
        G.add_edge(nid, paper_id)

    cited_papers = get_reference_details(nid, include_abstract=True)
    for paper in cited_papers:
        title = paper.title
        abstract = paper.abstract
        if abstract is not None:
            interpretability_boolean = is_interpretability_title_and_abstract(title, abstract)
        else:
            interpretability_boolean = None

        year = paper.year
        citation_count = paper.citation_count
        influential_citation_count = paper.influential_citation_count
        venue = paper.venue
        paper_id = paper.paper_id

        if paper_id is None:
            print('paper with no id:', title)
            continue
        G.add_node(
            paper_id,
            year=year,
            citation_count=citation_count,
            influential_citation_count=influential_citation_count,
            venue=venue,
            interpretability_boolean=interpretability_boolean,
            originally_from_dataset=False
        )
        G.add_edge(paper_id, nid)



# we save the graph
G_json = nx.cytoscape_data(G)
with open('./citationgraph/graph2.json', 'w') as f:
    json.dump(G_json, f)
