import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def api_client(prompt: str) -> str:
    try:
        response = client.responses.create(
            model="gpt-4o",
            instructions="You are a Python expert who analyzes and annotates Python code.",
            input=prompt,
            temperature=0.2,  # Lower temperature for more focused and precise responses
            # max_tokens=4000   # Adjust based on the expected response length
        )
        return response.output_text
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"
