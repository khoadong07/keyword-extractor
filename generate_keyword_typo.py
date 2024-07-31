from openai import OpenAI
import ast
import config

client = OpenAI(api_key=config.settings.OPEN_AI_KEY)


def generate_typos_with_llm(text):
    prompt = (
        f"Generate typo variations of the following sentence: '{text}'. Include input errors based on keyboard positions. "
        "The output should be a list of strings. If the result is not in the correct format, retry with the same prompt. "
        "Only generate based on the provided text, do not add any creativity. Pay attention to spaces between words, including potential typos or missing spaces. "
        "Return only the result as a list of strings like ['A', 'B', 'C', 'D']. If the format is incorrect, retry the entire prompt."
    )

    try:
        # Call the API to get the response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        content = response.choices[0].message.content

        # Validate and convert the content
        if content:
            try:
                result = ast.literal_eval(content)
                if isinstance(result, list) and all(isinstance(item, str) for item in result):
                    return result
                else:
                    print("The result is not a valid list of strings.")
            except (SyntaxError, ValueError):
                print("The content could not be parsed into a list.")
        else:
            print("The response content is empty.")

    except Exception as e:
        print(f"An error occurred while calling the API: {e}")

    return []