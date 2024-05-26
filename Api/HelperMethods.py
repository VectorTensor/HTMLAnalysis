import math

import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
def ParseHTMLString(stringData):
    soup = BeautifulSoup(stringData, 'html.parser')

    return soup


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

planes = 10
buckets = 2**planes

class EmbeddingString:
    def __init__(self, text, embedding):
        self.text= text
        self.embedding = embedding
def SplitDocumentsByTags(soup):
    chunks = []

    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6','a']):
        heading_text = heading.get_text()
        sibling = heading.find_next_sibling()
        content = []
        #hd_class = heading.attrs['class'][0].strip("[]").replace("'", "") if heading.has_attr('class') else ''
        # content.append(f"<{heading.name} {hd_class}> {heading_text}</{heading.name}>")
        content.append(f"{heading_text}")
        # content.append("heading.name " "+heading_text+" ")
        while sibling and sibling.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            #x = [f"{keys}: {values}" for keys,values in sibling.attrs.items()]
            #  tag_class = sibling.attrs['class'][0].strip("[]").replace("'", "") if sibling.has_attr('class') else ''

            #  content.append(f"<{sibling.name} {tag_class}>{sibling.get_text()}</{sibling.name}>")
            content.append(f"{sibling}")
            sibling = sibling.find_next_sibling()

        chunks.append((heading_text, " ".join(content)))
    return chunks

def aggregate_embeddings(chunk_embeddings, method='mean'):
    if method == 'mean':
        document_embedding = torch.mean(chunk_embeddings, axis=1)
    elif method == 'sum':
        document_embedding = torch.sum(chunk_embeddings, axis=1)
    else:
        raise ValueError("Unsupported aggregation method")
    return document_embedding

embedding_document_mapping = {}

def GetEmbeddings(chunks):

    embeddings = []
    #print(chunks)
    for heading, text in chunks:
        #print(text)
        if text == "":
            continue
        inputs = tokenizer(text, return_tensors='pt', padding = 'max_length', truncation=True, max_length=170)
        # print(inputs['input_ids'].shape)
        # print(inputs)
        #print(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embedding =  aggregate_embeddings(last_hidden_states,'mean')
            #  print(last_hidden_states.shape)
            embdString  = EmbeddingString(text,embedding.squeeze().numpy())
            embeddings.append(embdString)
            #embedding_document_mapping[text] = embeddings
    return embeddings

def createEmbeddingForSingleDocument(document):
    inputs = tokenizer(document, return_tensors='pt', padding = 'max_length', truncation=True, max_length=170)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = aggregate_embeddings(last_hidden_states,'mean')
        embedding = embedding.squeeze().numpy()

    return EmbeddingString(document, embedding)

def side_of_plane_matrx(P, v):
    # print(P.shape)
    #print(v.shape)
    dotproduct = np.dot(P,v.T)
    sign = np.sign(dotproduct)
    return sign

def hash_table(P_l,v ):
    hash_value = 0
    for i, P in enumerate(P_l):
        sign = side_of_plane_matrx(P, v)
        hash_i = 1 if sign >= 0 else 0
        #print(hash_i)
        hash_value += 2**i * hash_i

    return hash_value


def createEmbeddingSpace(embeddings, random_planes):
    embeddings_space={n : [] for n in range(0,buckets)}


    for e in embeddings:
        embeddings_space[hash_table(random_planes,e.embedding)].append(e)

    return embeddings_space


def cosineSimilarity(a,b):
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)
    #print(a.shape)
    #print(b.shape)
    dot_p = np.dot(a,b)
    cosB = dot_p/(a_mag*b_mag)
    #print(cosB)
    return cosB


def SimmilarText(query,embeddings, random_planes_matrix,thres=0.75):
    queryEmbd = createEmbeddingForSingleDocument(query)
    v = hash_table(random_planes_matrix,queryEmbd.embedding)


    possible_embeddings = embeddings
    if len(possible_embeddings) == 0:
        print("Sorry couldnt find the item")
        return -1
    # neareast_embd = possible_embeddings[0]
    close_embd =[]
    for p in possible_embeddings:
        current = cosineSimilarity(p.embedding,queryEmbd.embedding)
        if current > thres:
            close_embd.append(p.text)
        #prev = cosineSimilarity(neareast_embd.embedding,queryEmbd.embedding)
        # if current < prev:
        #     neareast_embd = p

    return close_embd

def extract_product_info(text):
    # Define regex patterns for product name, price, and description
    soup = BeautifulSoup(text, 'html.parser')

    # Extract the name (text before the first tag)
    name = soup.get_text(strip=True, separator=" ").split("$")[0].strip()

    # Extract the price
    price = soup.find('p', class_='price').get_text(strip=True)

    # Extract the description
    description = soup.find('p', class_='description').get_text(strip=True)

    return {
        'name': name,
        'price': price,
        'description': description
    }