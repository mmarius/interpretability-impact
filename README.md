# From Insights to Actions: The Impact of Interpretability and Analysis Research on NLP

## About

This repository contains the code for the paper [From Insights to Actions: The Impact of Interpretability and Analysis Research on NLP
](https://arxiv.org/abs/2406.12618) published at EMNLP 2024.

## Abstract

Interpretability and analysis (IA) research is a growing subfield within NLP with the goal of developing a deeper understanding of the behavior or inner workings of NLP systems and methods. Despite growing interest in the subfield, a criticism of this work is that it lacks actionable insights and therefore has little impact on NLP. In this paper, we seek to quantify the impact of IA research on the broader field of NLP. We approach this with a mixed-methods analysis of: (1) a citation graph of 185K+ papers built from all papers published at ACL and EMNLP conferences from 2018 to 2023, and their references and citations, and (2) a survey of 138 members of the NLP community. Our quantitative results show that IA work is well-cited outside of IA, and central in the NLP citation graph. Through qualitative analysis of survey responses and manual annotation of 556 papers, we find that NLP researchers build on findings from IA work and perceive it as important for progress in NLP, multiple subfields, and rely on its findings and terminology for their own work. Many novel methods are proposed based on IA findings and highly influenced by them, but highly influential non-IA work cites IA findings without being driven by them. We end by summarizing what is missing in IA work today and provide a call to action, to pave the way for a more impactful future of IA research.

## Citation graph

Our final citation graph build from all papers published at ACL and EMNLP from 2018 to 2023 can be found in `citationgraph/graph.zip`. Once you decompress this, you can use the data as:

```py
import networkx as nx
import json

with open('./graph.json') as f:
    graph_json = json.load(f)
    G = nx.cytoscape_graph(graph_json)

G.number_of_nodes() # 185384
```

Each node is indexed by its Semantic Scholar ID. This is how an example node looks:

```py
{
    'year': 2020,
    'citation_count': 19,
    'influential_citation_count': 2,
    'venue': 'Conference on Empirical Methods in Natural Language Processing',
    'originally_from_dataset': True,
    'area': 'Question Answering',
    'id': '4c61df1b4b9a164fec1a34587b4fffae029cd18c',
    'value': '4c61df1b4b9a164fec1a34587b4fffae029cd18c',
    'name': '4c61df1b4b9a164fec1a34587b4fffae029cd18c',
    'mt_prediction': False,
    'interpretability_prediction': False,
    'predicted_track': 'Inconclusive'
 }
```

To retrieve additional information for each paper, such as its title, abstract, authors, etc., you can query the semantic scholar API using the node `id`. 

## ACL and EMNLP papers

You find the raw data for all ACL and EMNLP papers we collected in `data/cl_papers.csv`.
