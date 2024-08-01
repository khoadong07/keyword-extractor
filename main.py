from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
import torch
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from find_keyword import find_related_keyword, general_inference
from generate_keyword_typo import generate_typos_with_llm
from response_template import bad_request, success

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_keywords(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)

flat_keywords = load_keywords('keywords/beauty_skincare.json')

keyword_embeddings = {keyword: model.encode(keyword, convert_to_tensor=True) for keyword in flat_keywords}


def get_embedding(text):
    return model.encode(text, convert_to_tensor=True)


# Hàm kiểm tra từ khóa trong câu nhập vào
def keyword_search(sentence):
    for keyword in flat_keywords:
        if keyword in sentence:
            return keyword
    return None


# Hàm tìm từ khóa liên quan đến câu nhập vào
def find_related_keyword(sentence):
    # Tìm kiếm từ khóa trong câu nhập vào
    found_keyword = keyword_search(sentence)
    if found_keyword:
        return found_keyword

    # Nếu không tìm thấy từ khóa, sử dụng mô hình để tìm từ khóa liên quan
    sentence_embedding = get_embedding(sentence)
    max_similarity = -1
    related_keyword = None

    for keyword, keyword_embedding in keyword_embeddings.items():
        similarity = util.pytorch_cos_sim(sentence_embedding, keyword_embedding).item()
        if similarity > max_similarity:
            max_similarity = similarity
            related_keyword = keyword

    return related_keyword


class Query(BaseModel):
    query: str


def retry_gen_spelling(content):
    retries = 3
    while retries > 0:
        print(f"retry time: {retries}")
        result = generate_typos_with_llm(content)
        if result is not None:
            return result
        retries -= 1
    return False

@app.post("/api/build-filter-keyword")
async def find_keyword(query: Query):
    sentence = query.query
    if not sentence:
        return bad_request(
            message="Query is null or empty",
            data=None
        )
    generate_json = general_inference(sentence)

    if generate_json:
        return success(message="Successfully", data=generate_json)
    else:
        return bad_request(message="Keyword not found", data=None)


@app.post("/api/find-keyword")
async def find_keyword(query: Query):
    sentence = query.query
    if not sentence:
        return bad_request(
            message="Query is null or empty",
            data=None
        )

    related_keyword = find_related_keyword(sentence)

    if related_keyword:
        result = {"related_keyword": related_keyword}
        return success(message="Successfully", data=result)
    else:
        return bad_request(message="Keyword not found", data=None)

@app.post("/api/generate-spelling-err")
async def find_keyword(query: Query):
    sentence = query.query
    if not sentence:
        return bad_request(
            message="Query is null or empty",
            data=None
        )

    generate = retry_gen_spelling(sentence)

    if generate:
        return success(message="Successfully", data=generate)
    else:
        return bad_request(message="Keyword not found", data=None)


tokenizer = GPT2Tokenizer.from_pretrained('NlpHUST/gpt2-vietnamese')
model = GPT2LMHeadModel.from_pretrained('NlpHUST/gpt2-vietnamese')


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        input_text = data.strip()
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        max_length = len(input_ids[0]) + 1

        sample_outputs = model.generate(input_ids,
                                        pad_token_id=tokenizer.eos_token_id,
                                        do_sample=True,
                                        max_length=max_length,
                                        top_k=40,
                                        num_beams=5,
                                        early_stopping=True,
                                        no_repeat_ngram_size=2,
                                        num_return_sequences=3)

        suggestions = [tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in sample_outputs]
        await websocket.send_text("\n---\n".join(suggestions))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
