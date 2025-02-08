from dotenv import load_dotenv
import os
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ollama
import json
import requests
import uuid
import networkx as nx
import chromadb
import re

app = FastAPI()
class RoadmapRequest(BaseModel):
    skills: list
    career_path: str

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file!")

client = Groq()


model_id = "llama3-8b-8192"  # Example model ID; adjust as needed



def generate_roadmap(skills, career_path):
    """Generate a detailed roadmap using Ollama."""
    prompt = f"""
    Given my skills {skills} and career goal {career_path}, create a **detailed, step-by-step roadmap**.
    Ignore unrelated skills (e.g., if the goal is "Web Developer," exclude "Machine Learning").
    
    **Rules**:
    - Output **ONLY JSON**, no explanations, no markdown.
    - JSON format:
      {{
        "Node1": ["ConnectedNode1", "ConnectedNode2"],
        "Node2": ["ConnectedNode3"],
        ...
      }}
    - Nodes should include:
      - Key topics to learn
      - Certifications or projects
      - Industry tools
      - Learning resources (books, courses, etc.)
      - A project at the end of every node to apply skills
    """

    try:
        messages=[{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=model_id,
            messages=messages
        )

        raw_content = response.choices[0].message.content

        #raw_content = response['message']['content'].strip()

        # Extract only JSON content
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            raw_content = match.group(0)
        else:
            return {"error": "No valid JSON found in LLM response", "raw_output": raw_content}

        return json.loads(raw_content)  # ✅ Return Python dictionary

    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from LLM", "raw_output": raw_content}
    except Exception as e:
        return {"error": str(e)}

# ✅ Correct API endpoint placement
@app.post("/get-roadmap")
async def create_roadmap(request: RoadmapRequest):
    """Main API endpoint for roadmap generation."""
    try:
        roadmap_data = generate_roadmap(request.skills, request.career_path)
        return {"roadmap": roadmap_data}  # ✅ Return proper JSON response

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
