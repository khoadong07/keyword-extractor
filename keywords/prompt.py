def general_prompt(content):
    prompt = [{
        "role": "user",
        "content": """Phân tích câu query và trả về các thông tin sau dưới dạng JSON: Keywords: Các từ khóa có thể có 
        trong câu query. Sentiment: Sắc thái của câu query, gồm 3 loại: positive, negative, neutral. Nếu có, 
        trả về giá trị tương ứng, nếu không, trả về null. Channel: Nguồn thông tin đang được quan tâm trong câu 
        query, nếu có, trả về giá trị tương ứng, nếu không, trả về null. Brand: Thương hiệu được nhắc đến trong câu 
        query, nếu có, trả về giá trị tương ứng, nếu không, trả về null. Ví dụ: Câu query: "kem trị mụn dr pong trên 
        youtube" Kết quả mong muốn: { "keywords": ["kem", "trị", "mụn", "dr", "pong", "youtube"], "sentiment": 
        "neutral", "channel": "youtube", "brand": "dr pong" } Hãy trả về kết quả dưới dạng JSON."""
    },
        {
            "role": "user",
            "content": f"Với yêu cầu trên hãy xử lý nội dung sau, \"{content}\"\n"
        },
        {
            "role": "user",
            "content": "Please ensure the following fields are present in the JSON response. If any field is missing, "
                       "re-run the prompt until all fields are present:\n\n- \"keywords\",\n- \"sentiment\","
                       "\n- \"channel\",\n- \"brand\".\n\nMake sure that all these fields are present in the JSON "
                       "response."
        }

    ]
    return prompt
