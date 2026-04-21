from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

class mod_skills(list):
    skills: list = Field(description="list of application skills")