import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import networkx as nx
import json
from tqdm import tqdm
from utils import bulk_get_paper_details
from utils import get_abstract
import zipfile

json_path = './citationgraph/graph.json'
zip_path = './citationgraph/graph.zip'

if not os.path.exists(json_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))
    print("ZIP file extracted.")

with open(json_path) as f:
    graph_json = json.load(f)
    G = nx.cytoscape_graph(graph_json)

G.number_of_nodes()

ssid_to_title_and_abstract = {}


ssids = list(node for node in G.nodes())

papers = bulk_get_paper_details(ssids, include_abstracts=True)

i = 0
for ssid, paper in tqdm(zip(ssids, papers), total=len(ssids)):
    i += 1
    if ssid in ssid_to_title_and_abstract:
        continue
    if not paper:
        continue
    if paper.abstract and paper.title:
        ssid_to_title_and_abstract[ssid] = { 'title': paper.title, 'abstract': paper.abstract }
        continue
    if paper.doi and paper.title:
        abstract = get_abstract(paper.doi)
        if abstract:
            ssid_to_title_and_abstract[ssid] = { 'title': paper.title, 'abstract': abstract }
            continue
    if i % 1000 == 0:
        with open('titles-abstracts.json', 'w') as json_file:
            json.dump(ssid_to_title_and_abstract, json_file)

with open('titles-abstracts.json', 'w') as json_file:
    json.dump(ssid_to_title_and_abstract, json_file)
