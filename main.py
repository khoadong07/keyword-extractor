from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
import torch

from find_keyword import find_related_keyword
from response_template import bad_request, success

app = FastAPI()

# Kiểm tra xem có GPU có sẵn không
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tải mô hình Sentence Transformers
model = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)


# Đọc danh sách từ khóa từ tệp JSON
def load_keywords(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


# Tải từ khóa từ tệp
flat_keywords = load_keywords('beauty_keywords.json')

# Tính embedding cho tất cả các từ khóa
keyword_embeddings = {keyword: model.encode(keyword, convert_to_tensor=True) for keyword in flat_keywords}


# Hàm tính embedding của một câu
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
    sentence: str

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

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
