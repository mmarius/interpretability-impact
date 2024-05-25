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
MODEL.load_state_dict(torch.load('../notebooks/classifier-weights.pt'))
# KEYWORDS = ['interpretability',
#             'interpretable',
#             'dimension',
#             'subspace',
#             'inner workings',
#             'circuit',
#             'probe',
#             'probing',
#             'counterfactual',
#             'attribution',
#             'subnetwork',
#             'intrinsic',
#             'explanation',
#             'factual',
#             'causal',
#             'role of ',
#             'why',
#             'encode',
#             'underlying',
#             'explainable',
#             'shortcut',
#             'encodings']

def is_interpretability_paper(row):
    x = row['embedding']
    vector = torch.tensor(np.fromstring(x[1:-1], sep='\n'), dtype=torch.float32)
    output = MODEL(vector)
    pred = torch.argmax(output)
    # has_keyword = any([word in row['abstract'].lower() for word in KEYWORDS])
    return bool(pred)


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
    # has_keyword = any([word in abstract.lower() for word in KEYWORDS])
    return bool(pred)

# MT_KEYWORDS = ['translation']
MT_MODEL = MLPClassifier(INPUT_SIZE, HIDDEN_SIZE, 2)
MT_MODEL.load_state_dict(torch.load('../notebooks/mt-classifier-weights.pt'))

def is_mt_title_and_abstract(title, abstract):
    embedding = get_embedding(title, abstract)
    embedding = torch.tensor(embedding, dtype=torch.float32)
    output = MT_MODEL(embedding)
    pred = torch.argmax(output)
    return bool(pred)


IE_MODEL = MLPClassifier(INPUT_SIZE, HIDDEN_SIZE, 2)
IE_MODEL.load_state_dict(torch.load('../notebooks/info-extraction-classifier-weights.pt'))

def is_ie_title_and_abstract(title, abstract):
    embedding = get_embedding(title, abstract)
    embedding = torch.tensor(embedding, dtype=torch.float32)
    output = IE_MODEL(embedding)
    pred = torch.argmax(output)
    return bool(pred)

DIALOGUE_MODEL = MLPClassifier(INPUT_SIZE, HIDDEN_SIZE, 2)
DIALOGUE_MODEL.load_state_dict(torch.load('../notebooks/dialogue-classifier-weights.pt'))

def is_dialogue_title_and_abstract(title, abstract):
    embedding = get_embedding(title, abstract)
    embedding = torch.tensor(embedding, dtype=torch.float32)
    output = DIALOGUE_MODEL(embedding)
    pred = torch.argmax(output)
    return bool(pred)


LABELS = ['Dialogue',
 'Generation',
 'Information Extraction/Retrieval',
 'Interpretability and Analysis',
 'Machine Learning',
 'Machine Translation and Multilinguality',
 'Multimodality, Speech and Grounding',
 'Other',
 'Question Answering',
 'Sentiment Analysis',
 'Social Science',
 'Summarization']
GENERAL_TRACK_MODEL = MLPClassifier(INPUT_SIZE, HIDDEN_SIZE, len(LABELS))
GENERAL_TRACK_MODEL.load_state_dict(torch.load('../notebooks/general-classifier-weights.pt'))
def predict_track(title, abstract):
    embedding = get_embedding(title, abstract)
    embedding = torch.tensor(embedding, dtype=torch.float32)
    output = GENERAL_TRACK_MODEL(embedding)
    pred = torch.argmax(output)
    return LABELS[pred]



def get_batch_embeddings(titles, abstracts):
    assert len(titles) == len(abstracts), "Titles and abstracts lists must have the same length"

    texts = [title + tokenizer.sep_token + abstract for title, abstract in zip(titles, abstracts)]

    inputs = tokenizer(texts,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       return_token_type_ids=False,
                       max_length=512)
    with torch.no_grad():
        output = embedding_model(**inputs)

    embeddings = output.last_hidden_state[:, 0, :]

    return embeddings

def predict_batch_tracks(titles, abstracts):
    embeddings = get_batch_embeddings(titles, abstracts)
    embeddings = embeddings.to(torch.float32)

    with torch.no_grad():
        outputs = GENERAL_TRACK_MODEL(embeddings)

    preds = torch.argmax(outputs, dim=1)
    predicted_labels = [LABELS[pred.item()] for pred in preds]

    return predicted_labels
