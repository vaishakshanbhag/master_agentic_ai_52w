import os
from dotenv import load_dotenv
from openai import OpenAI
 
load_dotenv()
client = OpenAI()
 
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Explain blockchain like I’m 12,"}]
)
 
print(response.choices[0].message.content)