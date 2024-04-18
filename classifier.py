from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import torch
import torch.nn as nn
import numpy as np


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=-1)
        return out

INPUT_SIZE = 768
HIDDEN_SIZE = 400
MODEL = MLPClassifier(INPUT_SIZE, HIDDEN_SIZE, 2)
MODEL.load_state_dict(torch.load('./notebooks/classifier-weights.pt'))
KEYWORDS = ['interpretability',
            'interpretable',
            'dimension',
            'subspace',
            'inner workings',
            'circuit',
            'probe',
            'probing',
            'counterfactual',
            'attribution',
            'subnetwork',
            'intrinsic',
            'explanation',
            'factual',
            'causal',
            'role of ',
            'why',
            'encode',
            'underlying',
            'explainable',
            'shortcut',
            'encodings']

def is_interpretability_paper(row):
    x = row['embedding']
    vector = torch.tensor(np.fromstring(x[1:-1], sep='\n'), dtype=torch.float32)
    output = MODEL(vector)
    pred = torch.argmax(output)
    has_keyword = any([word in row['abstract'].lower() for word in KEYWORDS])
    return bool(pred and has_keyword)


tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
embedding_model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
adapter_name = embedding_model.load_adapter("allenai/specter2_classification", source="hf", set_active=True)


def get_embedding(title, abstract):
    text = title + tokenizer.sep_token + abstract
    inputs = tokenizer(text,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       return_token_type_ids=False,
                       max_length=512)
    output = embedding_model(**inputs)
    embeddings = output.last_hidden_state[:, 0, :][0].detach().numpy()
    return embeddings

def is_interpretability_title_and_abstract(title, abstract):
    embedding = get_embedding(title, abstract)
    embedding = torch.tensor(embedding, dtype=torch.float32)
    output = MODEL(embedding)
    pred = torch.argmax(output)
    has_keyword = any([word in abstract.lower() for word in KEYWORDS])
    return bool(pred and has_keyword)

MT_KEYWORDS = ['translation']
MT_MODEL = MLPClassifier(INPUT_SIZE, HIDDEN_SIZE, 2)
MT_MODEL.load_state_dict(torch.load('./notebooks/mt-classifier-weights.pt'))

def is_mt_title_and_abstract(title, abstract):
    embedding = get_embedding(title, abstract)
    embedding = torch.tensor(embedding, dtype=torch.float32)
    output = MT_MODEL(embedding)
    pred = torch.argmax(output)
    has_keyword = any([word in abstract.lower() for word in MT_KEYWORDS])
    return bool(pred and has_keyword)
