from openai import OpenAI
import ast
import config

from config import settings
import requests
import regex as re

from find_keyword import extract_json_from_string
import json

def generate_typos_with_llm(text):
    # prompt = (
    #
    # )
    #
    # try:
    #     # Call the API to get the response
    #     response = client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": prompt
    #                     }
    #                 ]
    #             },
    #         ],
    #         temperature=0.7,
    #         max_tokens=1024,
    #         top_p=1,
    #         frequency_penalty=0,
    #         presence_penalty=0
    #     )
    #     content = response.choices[0].message.content
    #
    #     # Validate and convert the content
    #     if content:
    #         try:
    #             result = ast.literal_eval(content)
    #             if isinstance(result, list) and all(isinstance(item, str) for item in result):
    #                 return result
    #             else:
    #                 print("The result is not a valid list of strings.")
    #         except (SyntaxError, ValueError):
    #             print("The content could not be parsed into a list.")
    #     else:
    #         print("The response content is empty.")
    #
    # except Exception as e:
    #     print(f"An error occurred while calling the API: {e}")
    #
    # return []

    prompt = [{
        "role": "user",
        "content": (
            f"Generate typo variations of the following Vietnamese sentence, including the original sentence itself: '{text}'. "
            "Focus on spelling errors in Vietnamese, including handling Vietnamese diacritics. "
            "The output should always be a JSON object with the format {{\"key_gen\": [\"{text}\", \"key1\", \"key2\", ...]}}. "
            "Only generate based on the provided text, do not add any creativity. Pay attention to spaces between words, including potential typos or missing spaces. "
            "Return only the result in the specified JSON format. If the output is not in the exact JSON format as requested, retry until it matches."
        )
    }]

    payload = {
        "model": settings.FIREWORKS_MODEL,
        "max_tokens": int(settings.FIREWORKS_API_MAX_TOKEN),
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": float(settings.TEMPERATURE),
        "messages": prompt
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
            return core
        else:
            return None
    else:
        response.raise_for_status()