from fastapi import FastAPI
from pydantic import BaseModel

from find_keyword import find_related_keyword
from response_template import bad_request, success

app = FastAPI()

# Root path
@app.get("/")
async def read_version():
    return {"message": "v1.2.beta"}

class Query(BaseModel):
    query: str

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
