# Impact of interpretability and analysis research

Our final citation graph can be found in `citationgraph/graph.zip`. Once you decompress this, you can use the data as:

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

A lot of information is missing from the paper (particularly its title and abstract), as the graph started to take a lot of storage. These should be available by its semantic scholar.

Alternatively, we also have the ACL and EMNLP papers in `data/cl_papers.csv` (but not the citations and references of them), in case you need access to the titles and abstracts.
