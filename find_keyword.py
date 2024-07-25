import json
import torch

from sentence_transformers import SentenceTransformer, util

from config import settings
from keywords.prompt import general_prompt

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


import requests
import regex as re

def extract_json_from_string(json_string):
    pattern = r'\{.*\}'
    match = re.search(pattern, json_string, re.DOTALL)

    if match:
        extracted_json = match.group()
        try:
            data = json.loads(extracted_json)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        return None
def general_inference(content_input):
    related_keyword = find_related_keyword(content_input)
    payload = {
        "model": settings.FIREWORKS_MODEL,
        "max_tokens": int(settings.FIREWORKS_API_MAX_TOKEN),
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": float(settings.TEMPERATURE),
        "messages": general_prompt(content=content_input)
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.FIREWORKS_TOKEN}"
    }

    response = requests.post(settings.FIREWORKS_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_data = response.json()
        llm_response = response_data['choices'][0]['message']['content']
        if llm_response:
            core = extract_json_from_string(llm_response)
            core["keywords"] = related_keyword
            result = {
                "id": response.json()['id'],
                "created": response.json()['created'],
                "model": response.json()['model'],
                "core": core,
                "usage": response.json()['usage']
            }
            return result
        else:
            return None
    else:
        response.raise_for_status()
