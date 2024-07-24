import json
import torch

from sentence_transformers import SentenceTransformer, util

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)


def load_keywords(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


flat_keywords = load_keywords('keywords/beauty_skincare.json')

keyword_embeddings = {keyword: model.encode(keyword, convert_to_tensor=True) for keyword in flat_keywords}


def get_embedding(text):
    return model.encode(text, convert_to_tensor=True)


def keyword_search(sentence):
    for keyword in flat_keywords:
        if keyword in sentence:
            return keyword
    return None


def find_related_keyword(sentence):
    found_keyword = keyword_search(sentence)
    if found_keyword:
        return found_keyword

    sentence_embedding = get_embedding(sentence)
    max_similarity = -1
    related_keyword = None

    for keyword, keyword_embedding in keyword_embeddings.items():
        similarity = util.pytorch_cos_sim(sentence_embedding, keyword_embedding).item()
        if similarity > max_similarity:
            max_similarity = similarity
            related_keyword = keyword

    return related_keyword
