from acl_anthology import Anthology
# Instantiate the Anthology from the official repository
# This will download the Anthology once at the beginning
anthology = Anthology.from_repo()

import pandas as pd
from pandas import DataFrame
import requests
from tqdm import tqdm
from typing import Optional

API_KEY = "your-api-key" # TODO(mm): remove when making repo public

# Rate limit:
#   1 request per second for the following endpoints:
#       /paper/batch
#       /paper/search
#       /recommendations
#   10 requests / second for all other calls

#########################################################
# Paper utils
#########################################################

class SemanticScholarPaper:
    """A Semantic Scholar paper"""

    paper_id: str = None
    title: str = None
    venue: str = None
    # venue_details: dict = None
    year: int = None
    citation_count: int = None
    influential_citation_count: int = None
    acl_doi: str = None
    embedding: Optional[list[float]] = None

    # additional attributes that will be filled when building a citation graph
    citations: list = []
    cites_result: bool = False
    cites_methodology: bool = False
    cites_background: bool = False
    was_influenced: bool = False
    contexts: list[str] = []
    abstract: Optional[str] = None
    is_influential_citation: bool = False
    doi: Optional[str] = None

    def __init__(self,
                 paper_id: str,
                 title: str,
                 venue: str,
                 year: int,
                 citation_count: int,
                 influential_citation_count: int,
                 embedding: Optional[list[float]]=None,
                 abstract: Optional[str]=None,
                 is_influential_citation: bool = False,
                 cites_result: bool = False,
                 cites_background: bool = False,
                 cites_methodology: bool = False,
                 doi: Optional[str] = None,
                 contexts: list[str] = []
                 ):
        self.paper_id = paper_id
        self.title = title
        self.venue = venue
        self.year = year
        self.citation_count = citation_count
        self.influential_citation_count = influential_citation_count
        self.embedding = embedding
        self.abstract = abstract
        self.is_influential_citation = is_influential_citation
        self.cites_result = cites_result
        self.cites_background = cites_background
        self.cites_methodology = cites_methodology
        self.doi = doi
        self.contexts = contexts

    def add_acl_doi(self, doi: str) -> None:
        self.acl_doi = doi

    def add_citation_types(self, cites_result: bool, cites_methodology: bool, cites_background: bool):
        self.cites_result = cites_result
        self.cites_methodology = cites_methodology
        self.cites_background = cites_background

    def add_was_influenced(self, was_influenced: bool):
        self.was_influenced = was_influenced

    def add_contexts(self, contexts: list[str]):
        self.contexts = contexts

    def retrieve_citations(self, limit: int = 1000) -> None:
        citing_papers = get_citation_details(self.paper_id, limit)
        citing_papers = [merge_into_single_dict(d) for d in citing_papers['data']]

        paper_ids = list(map(lambda x: x["citingPaper_paperId"], citing_papers))
        papers = bulk_get_paper_details(paper_ids)
        indexed_papers = dict([(paper.paper_id, paper) for paper in papers])
        for paper_dict in citing_papers:
            paper = indexed_papers[paper_dict['citingPaper_paperId']]
            paper.add_citation_types(
                cites_result=paper_dict["result"],
                cites_background=paper_dict["background"],
                cites_methodology=paper_dict["methodology"]
            )
            paper.add_was_influenced(paper_dict["isInfluential"])
            paper.add_contexts(paper_dict["contexts"])
            self.citations.append(paper)

    def to_dict(self):
        return vars(self)

    def to_df(self):
        paper_dict = self.to_dict()
        paper_dict = {k: [v] for k, v in paper_dict.items()}
        return pd.DataFrame.from_dict(paper_dict, orient="columns")

    def citations_to_df(self):
        columns = self.citations[0].to_df().columns
        citations_dict = {k: [] for k in columns}

        for paper in self.citations:
            for k, v in paper.to_dict().items():
                citations_dict[k].append(v)

        return pd.DataFrame.from_dict(citations_dict, orient="columns")

    def __str__(self):
        return str(self.to_dict())


#########################################################
# ACL Anthology related utils
#########################################################

def get_acl_anthology_doi(title: str) -> str | None:
    """
    Return the DOI of an ACL Anthology paper.

            Parameters:
                    title (str): The title of the paper

            Returns:
                    doi (str): The DOI of the paper or None of the paper is not found in the ACL Anthology
    """
    for paper in anthology.papers():        
        if title.lower() == str(paper.title).lower():
            return paper.doi

    return None


#########################################################
# Semantic Scholar related utils
#########################################################


def get_papers_by_keywords(keywords: str, limit: int = 100) -> dict:
    # return a dictionary of papers 
    query = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keywords}&offset=0&limit={limit}"
    fields = "paperId,title,year,venue"

    # query Semantic Scholar API
    response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields})
    while response.status_code != 200: # 200 means success
        # try again
        print("Trying again ...")
        response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields})

    # Success! Convert response to dict 
    papers = response.json()

    # TODO(mm): convert to paper class

    return papers

def get_paper_details(paper_id: str) -> SemanticScholarPaper:
    # get details for a single paper
    query = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    fields = "title,venue,year,citationCount,influentialCitationCount"
    
    # query Semantic Scholar API
    response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields})
    while response.status_code != 200: # 200 means success
        # try again
        print("Trying again ...")
        response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields})

    # Success! Convert response to dict 
    paper_dict = response.json()

    paper = SemanticScholarPaper(
        paper_id=paper_id, 
        title=paper_dict["title"],
        venue=paper_dict["venue"],
        year=paper_dict["year"],
        citation_count=paper_dict["citationCount"],
        influential_citation_count=paper_dict["influentialCitationCount"],
    )
    return paper


def chunk_list(lst, chunk_size):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def bulk_get_paper_details(paper_ids: list[str], include_embedding=False, include_citations=False, include_abstracts=False) -> list[SemanticScholarPaper]:
    chunk_size = 500  # API limit
    all_papers = []

    for chunk in tqdm(chunk_list(paper_ids, chunk_size), desc="Fetching papers", total=len(paper_ids)//chunk_size):
        url = 'https://api.semanticscholar.org/graph/v1/paper/batch'
        fields = "title,venue,year,citationCount,influentialCitationCount,externalIds"
        if include_embedding:
            fields += ',embedding'
        if include_citations:
            fields += ',citations'
        if include_abstracts:
            fields += ',abstract'

        response = requests.post(url,
                                 headers={"x-api-key": API_KEY},
                                 params={"fields": fields},
                                 json={"ids": chunk})

        for paper_dict in response.json():
            if paper_dict is None:
                all_papers.append(None)
                continue
            abstract = paper_dict.get('abstract')
            if abstract and len(abstract) < 10:
                abstract = None
            if paper_dict['externalIds'] and 'DOI' in paper_dict['externalIds']:
                doi = paper_dict['externalIds']['DOI']
            else:
                doi = None

            paper = SemanticScholarPaper(
                paper_id=paper_dict['paperId'],
                title=paper_dict["title"],
                venue=paper_dict["venue"],
                year=paper_dict["year"],
                citation_count=paper_dict["citationCount"],
                influential_citation_count=paper_dict["influentialCitationCount"],
                embedding=paper_dict.get('embedding', {}).get('vector'),
                abstract=abstract,
                doi=doi
            )
            all_papers.append(paper)

    return all_papers

def get_citation_details(paper_id: str, limit: int = 1000, max_retries: int = 10, include_abstract=False) -> list:
    # this gets all the citations for a given paper
    # considering also that the results might get split into multiple pages

    all_papers = []
    offset = 0
    query = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    fields = "title,isInfluential,intents,year,venue,contexts,citationCount,influentialCitationCount,externalIds"
    if include_abstract:
        fields += ',abstract'

    while True:
        response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields, "limit": f"{limit}", "offset": offset})

        retries = 1
        while response.status_code != 200:
            print("Trying again ...", paper_id)
            response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields, "limit": f"{limit}"})
            retries += 1
            if retries > max_retries:
                break

        json_response = response.json()
        if "data" in json_response and json_response["data"] is not None:
            for paper in response.json()["data"]:
                if paper['citingPaper']['externalIds'] and 'DOI' in paper['citingPaper']['externalIds']:
                    doi = paper['citingPaper']['externalIds']['DOI']
                else:
                    doi = None
                abstract = paper['citingPaper'].get('abstract')
                if abstract and len(abstract) < 10:
                    abstract = None
                paper = SemanticScholarPaper(
                    paper_id=paper['citingPaper']['paperId'],
                    title=paper['citingPaper']["title"],
                    venue=paper['citingPaper']["venue"],
                    year=paper['citingPaper']["year"],
                    citation_count=paper['citingPaper']["citationCount"],
                    influential_citation_count=paper['citingPaper']["citationCount"],
                    abstract=abstract,
                    is_influential_citation=paper['isInfluential'],
                    cites_result='result' in paper['intents'],
                    cites_methodology='methodology' in paper['intents'],
                    cites_background='background' in paper['intents'],
                    doi=doi,
                    contexts=paper.get('contexts', [])
                )
                all_papers.append(paper)

        if 'next' in json_response:
            # we get the next page of results
            offset = json_response['next']
        else:
            break

    return all_papers

def get_reference_details(paper_id: str, limit: int = 1000, max_retries: int = 10, include_abstract=False) -> list:
    # this gets all the references for a given paper
    # considering also that the results might get split into multiple pages

    all_papers = []
    offset = 0
    query = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
    fields = "title,isInfluential,intents,year,venue,citationCount,influentialCitationCount,externalIds"
    if include_abstract:
        fields += ',abstract'

    while True:
        response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields, "limit": f"{limit}", "offset": offset})

        retries = 1
        while response.status_code != 200:
            print("Trying again ...", paper_id)
            response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields, "limit": f"{limit}"})
            retries += 1
            if retries > max_retries:
                break

        json_response = response.json()
        if "data" in json_response and json_response["data"] is not None:
            for paper in response.json()["data"]:
                if 'extenalIds' in paper['citedPaper'] and 'DOI' in paper['citedPaper']['externalIds']:
                    doi = paper['citedPaper']['externalIds']['DOI']
                else:
                    doi = None
                abstract = paper['citedPaper'].get('abstract')
                if abstract and len(abstract) < 10:
                    abstract = None
                paper = SemanticScholarPaper(
                    paper_id=paper['citedPaper']['paperId'],
                    title=paper['citedPaper']["title"],
                    venue=paper['citedPaper']["venue"],
                    year=paper['citedPaper']["year"],
                    citation_count=paper['citedPaper']["citationCount"],
                    influential_citation_count=paper['citedPaper']["citationCount"],
                    abstract=abstract,
                    is_influential_citation=paper['isInfluential'],
                    cites_result='result' in paper['intents'],
                    cites_methodology='methodology' in paper['intents'],
                    cites_background='background' in paper['intents'],
                    doi=doi,
                    contexts=paper.get('contexts', [])
                )
                all_papers.append(paper)

        if 'next' in json_response:
            # we get the next page of results
            offset = json_response['next']
        else:
            break

    return all_papers

def get_alternative_id(paper_id: str, max_retries=2) -> Optional[str]:
    query = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    fields = "externalIds"
    response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields})
    retries = 0
    while response.status_code != 200:
        print("Trying again ...", paper_id)
        response = requests.get(query, headers={"x-api-key": API_KEY}, params={"fields": fields})
        retries += 1
        if retries > max_retries:
            break
    json_response = response.json()
    if 'externalIds' in json_response and 'DOI' in json_response['externalIds']:
        return json_response['externalIds']['DOI']
    return None


#########################################################
# Abstract sources
#########################################################
import pyalex
import bs4

pyalex.config.email = "tomvergara@uc.cl"
pyalex.config.max_retries = 1
pyalex.config.retry_backoff_factor = 0.1

def get_abstract_with_pyalex(doi):
    paper = pyalex.Works()['https://doi.org/' + doi]
    abstract = paper['abstract']
    if abstract:
        return abstract
    return None

def get_abstract_with_acl_anthology(doi):
    url_path = doi.split('/')[-1]
    url = 'https://aclanthology.org/' + url_path
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    css_selector = "#main > div.row.acl-paper-details > div.col.col-lg-10.order-2 > div > div > span"
    element = soup.select_one(css_selector)
    abstract = element.text
    return abstract

def get_abstract_from_crossref(doi):
    url = 'https://api.crossref.org/works/' + doi
    r = requests.get(url)
    response = r.json()
    soup = bs4.BeautifulSoup(response['message']['abstract'], 'html.parser')
    return soup.find('jats:p').text

def get_abstract(doi):
    try:
        return get_abstract_with_pyalex(doi)
    except Exception as e:
        print(e)
        print("Failed to get abstract from PyAlex")
    try:
        return get_abstract_from_crossref(doi)
    except Exception as e:
        print(e)
        print("Failed to get abstract from Crossref")

    try:
        return get_abstract_with_acl_anthology(doi)
    except Exception as e:
        print(e)
        print("Failed to get abstract from ACL Anthology")

    print("Failed to get abstract from any source", doi)
    return None



#########################################################
# pandas related utils
#########################################################

def convert_to_df(papers: dict) -> DataFrame:
    df = DataFrame.from_dict(papers, orient='columns')
    return df


def filter_df(df, **kwargs):
    for column, value in kwargs.items():
        mask = None
        if column in df.columns:
            if mask is None:
                mask = (df[column] == value).values
            else:
                mask = mask & (df[column] == value).values
        else:
            print(f"Ignoring argument {column} with value {value}")
    
    return df[mask]

#########################################################
# Misc.
#########################################################

def merge_into_single_dict(d: dict) -> dict:
    r = {'result': False, 'methodology': False, 'background': False}
    for k, v in d.items():
        if k == 'citingPaper':
            for kk in v:
                r[f'{k}_{kk}'] = v[kk]
        elif k == 'intents': # convert list intents into single columns
            for intent in v:
                r[intent] = True
        else:
            r[k] = v
    return r


#########################################################
# Constants
#########################################################

TRACK_MAPPING = {
  'Cognitive Modeling and Psycholinguistics': 'Linguistic Theories and Psycholinguistics',
  'Commonsense Reasoning': 'Commonsense Reasoning',
  'Computational Social Science and Cultural Analytics': 'Social Science',
  'Computational Social Science and Social Media': 'Social Science',
  'Dialog and Interactive Systems': 'Dialogue',
  'Dialogue and Interactive Systems': 'Dialogue',
  'Discourse and Pragmatics': 'Discourse and Pragmatics',
  'Discourse and Pragmatics & Ethics in NLP': 'Discourse and Pragmatics',
  'Efficient Methods for NLP': 'Efficient Methods',
  'Ethic Concerns:Dialogue and Interactive Systems': 'Ethics',
  'Ethic Concerns:Ethics': 'Ethics',
  'Ethic Concerns:Linguistic Theories, Cognitive Modeling and Psycholinguistics': 'Ethics',
  'Ethic Concerns:Multilinguality': 'Ethics',
  'Ethic Concerns:NLP Applications': 'Ethics',
  'Ethic Concerns:Resources and Evaluation': 'Ethics',
  'Ethic Concerns:Sentiment Analysis, Stylistic Analysis, and Argument Mining': 'Ethics',
  'Ethic Concerns:Speech, Vision, Robotics, Multimodal Grounding': 'Ethics',
  'Ethic Concerns:Unsupervised and Weakly-Supervised Methods in NLP': 'Ethics',
  'Ethics': 'Ethics',
  'Ethics and NLP': 'Ethics',
  'Ethics in NLP': 'Ethics',
  'Generation': 'Generation',
  'Human-Centered NLP': 'Human-Centered NLP',
  'Industry': 'Industry',
  'Information Extraction': 'Information Extraction/Retrieval',
  'Information Retrieval and Text Mining': 'Information Extraction/Retrieval',
  'Interpretability and Analysis of Models for NLP': 'Interpretability and Analysis',
  'Interpretability, Interactivity and Analysis of Models for NLP': 'Interpretability and Analysis',
  'Interpretability, Interactivity, and Analysis of Models for NLP': 'Interpretability and Analysis',
  'Language Generation': 'Generation',
  'Language Grounding to Vision, Robotics and Beyond': 'Multimodality, Speech and Grounding',
  'Language Grounding to Vision, Robotics, and Beyond': 'Multimodality, Speech and Grounding',
  'Language Groundings, Speech and Multimodality': 'Multimodality, Speech and Grounding',
  'Language Modeling and Analysis of Language Models': 'Interpretability and Analysis',
  'Large Language Models': 'Large Language Models',
  'Linguistic Diversity': 'Machine Translation and Multilinguality',
  'Linguistic Theories, Cognitive Modeling and Psycholinguistics': 'Linguistic Theories and Psycholinguistics',
  'Linguistic Theories, Cognitive Modeling, and Psycholinguistics': 'Linguistic Theories and Psycholinguistics',
  'Machine Learning for NLP': 'Machine Learning',
  'Machine Translation': 'Machine Translation and Multilinguality',
  'Machine Translation and Multilinguality': 'Machine Translation and Multilinguality',
  'Multidisciplinary and Area Chair COI': 'Theme',
  'Multilingualism and Cross-Lingual NLP': 'Machine Translation and Multilinguality',
  'Multilinguality': 'Machine Translation and Multilinguality',
  'Multilinguality and Linguistic Diversity': 'Machine Translation and Multilinguality',
  'NLP Applications': 'Applications',
  'Natural Language Generation': 'Generation',
  'Phonology, Morphology and Word Segmentation': 'Phonology, Morphology and Word Segmentation',
  'Phonology, Morphology, and Word Segmentation': 'Phonology, Morphology and Word Segmentation',
  'Question Answering': 'Question Answering',
  'Resources and Evaluation': 'Resources and Evaluation',
  'Semantics': 'Semantics',
  'Semantics: Lexical': 'Semantics',
  'Semantics: Lexical Semantics': 'Semantics',
  'Semantics: Lexical, Sentence level, Document Level, Textual Inference, etc.': 'Semantics',
  'Semantics: Lexical, Sentence level, Textual Inference and Other areas': 'Semantics',
  'Semantics: Sentence Level': 'Semantics',
  'Semantics: Sentence-level Semantics, Textual Inference and Other areas': 'Semantics',
  'Semantics: Sentence-level Semantics, Textual Inference, and Other Areas': 'Semantics',
  'Semantics: Textual Inference and Other Areas of Semantics': 'Semantics',
  'Sentiment Analysis, Stylistic Analysis, and Argument Mining': 'Sentiment Analysis',
  'Special Theme on Language Diversity: From Low Resource to Endangered': 'Theme',
  'Speech and Multimodality': 'Multimodality, Speech and Grounding',
  'Speech, Vision, Robotics, Multimodal Grounding': 'Multimodality, Speech and Grounding',
  'Summarization': 'Summarization',
  'Syntax, Parsing and their Applications': 'Syntax',
  'Syntax: Tagging, Chunking and Parsing': 'Syntax',
  'Syntax: Tagging, Chunking, and Parsing': 'Syntax',
  'Theme': 'Theme',
  'Theme Track': 'Theme',
  'Theme Track: Large Language Models and the Future of NLP': 'Theme',
  'Theme: Reality Check': 'Theme',
  'Theory and Formalism in NLP (Linguistic and Mathematical)': 'Theory and Formalism in NLP',
  'Unsupervised and Weakly-Supervised Methods in NLP': 'Unsupervised and Weakly-Supervised Methods in NLP',
}
