from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ollama
import json
import requests
import uuid
import networkx as nx
import chromadb

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model for roadmap generation
class RoadmapRequest(BaseModel):
    skills: list
    career_path: str

# Remove incorrect decorator
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
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response['message']['content'].strip()

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


class RoadmapFieldRequest(BaseModel):
    field: str  # Field (topic) the user wants details about

def get_field_details(field):
    """Generate a description and learning resources for a specific roadmap field using Ollama."""
    prompt = f"""
    Provide a **detailed description** of the topic: "{field}".

    Also, list at least **5 useful learning resources** (books, online courses, documentation, websites, etc.) for mastering this topic.

    **Rules**:
    - Output **ONLY JSON**, no explanations, no markdown.
    - JSON format:
      {{
          "description": "Detailed explanation of the field...",
          "resources": [
              {{"name": "Resource Name", "url": "https://example.com", "type": "book/course/documentation"}},
              ...
          ]
      }}
    - Provide **at least 5 learning resources**.
    - Do NOT include any extra explanation, introductions, or notes. Only return the JSON response.
    """

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response['message']['content'].strip()

        # 🔹 Extract only JSON content
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            json_content = match.group(0)
        else:
            return {"error": "No valid JSON found in LLM response", "raw_output": raw_content}

        # 🔹 Validate JSON format
        details = json.loads(json_content)

        # 🔹 Ensure minimum 5 resources
        if "resources" in details and isinstance(details["resources"], list):
            while len(details["resources"]) < 5:
                details["resources"].append(
                    {"name": "Additional Resource", "url": "#", "type": "general"}
                )

        return details  # ✅ Return dictionary instead of saving to a file

    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from LLM", "raw_output": raw_content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/get-field-details")
async def get_field_details_api(request: RoadmapFieldRequest):
    """Fetch description & resources for a roadmap field and return as a dictionary."""
    try:
        details = get_field_details(request.field)
        return request.field, details  # ✅ No file creation

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ====================================================
# NEW FEATURE: SKILL EMBEDDINGS + CHROMADB INTEGRATION
# ====================================================

# Initialize ChromaDB


# # chroma_client.delete_collection(name="users_skills")
# collection = chroma_client.get_or_create_collection(name="users_skills")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chromadb_store")

# Get a list of existing collection names
collection_names = chroma_client.list_collections()
hackathons_collection = chroma_client.get_or_create_collection(name="hackathon_opportunities")

# Check if the users_skills collection exists
if "users_skills" not in collection_names:
    print("Collection does not exist. Creating a new one...")
    collection = chroma_client.get_or_create_collection(name="users_skills")
else:
    collection = chroma_client.get_collection(name="users_skills")

import uuid
import re

class UserSkills(BaseModel):
    username: str  # User's display name
    skills: list   # List of skills

MAX_EMBEDDING_DIM = 200  # Set fixed size for embeddings

def get_embedding(text):
    """Generate numerical embeddings and ensure fixed-length format."""
    prompt = f"Convert the following skills into a numeric vector:\n{text}\nOutput only a space-separated list of numbers."

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

    # Extract and convert response to a list of floats
    embedding_text = response['message']['content'].strip()
    embedding = [float(num) for num in re.findall(r"-?\d+\.\d+|-?\d+", embedding_text)]

    # 🔹 Ensure fixed-size embedding
    if len(embedding) > MAX_EMBEDDING_DIM:
        embedding = embedding[:MAX_EMBEDDING_DIM]  # Truncate
    else:
        embedding += [0.0] * (MAX_EMBEDDING_DIM - len(embedding))  # Pad

    return embedding

@app.post("/store-skills")
def store_skills(data: UserSkills):
    """Store user skills in ChromaDB and return a dictionary."""
    try:
        skills_text = " ".join(data.skills)
        embedding = get_embedding(skills_text)

        # Check if username exists
        existing_users = collection.get(where={"username": data.username})

        if existing_users["ids"]:
            user_id = existing_users["ids"][0]  # Use existing user ID
        else:
            user_id = str(uuid.uuid4())  # Generate new UUID

        # Store data in ChromaDB
        collection.add(
            ids=[user_id],
            embeddings=[embedding],
            metadatas=[{
                "username": data.username,
                "skills": ", ".join(data.skills)
            }]
        )

        return {
            "message": "Skills stored successfully",
           
            "username": data.username,
            "dimension": len(embedding),
            "collection_count": collection.count()
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/find-similar")
def find_similar(data: UserSkills):
    """Find up to 5 similar users based on skills and return a dictionary."""
    try:
        skills_text = " ".join(data.skills)
        query_embedding = get_embedding(skills_text)

        # 🔹 Query ChromaDB for similar users
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        similar_users = []
        if results["ids"][0]:  # Check if users were found
            for i, user_id in enumerate(results["ids"][0]):  
                user_metadata = collection.get(ids=[user_id])["metadatas"][0]

                # Handle missing metadata
                username = user_metadata.get("username", "Unknown User")
                skills = user_metadata.get("skills", "No skills found")

                similar_users.append({
                    
                    "username": username,
                    "skills": skills,
                    "score": results["distances"][0][i]  # Similarity score
                })

        return similar_users  # ✅ No JSON file

    except Exception as e:
        return {"error": str(e)}

# =============================
# 🔍 Find Hackathons User Qualifies For
# =============================
# =============================
# 🏆 Hackathon Opportunities Model & Storage
# =============================

class HackathonOpportunity(BaseModel):
    name: str  # Hackathon name
    required_skills: list  # Required skills

@app.post("/add-hackathon/")
def add_hackathon(data: HackathonOpportunity):
    """Store hackathon opportunities in ChromaDB."""
    try:
        skills_text = " ".join(data.required_skills)
        embedding = get_embedding(skills_text)

        hackathon_id = str(uuid.uuid4())  # Unique ID for each hackathon

        # Store in ChromaDB
        hackathons_collection.add(
            ids=[hackathon_id],
            embeddings=[embedding],
            metadatas=[{
                "name": data.name,
                "required_skills": ", ".join(data.required_skills)
            }]
        )

        return {"message": "Hackathon added successfully", "hackathon_id": hackathon_id}

    except Exception as e:
        return {"error": str(e)}

@app.post("/find-hackathons/")
def find_hackathons(data: UserSkills):
    """Find hackathons a user qualifies for based on their skills."""
    try:
        skills_text = " ".join(data.skills)
        query_embedding = get_embedding(skills_text)

        results = hackathons_collection.query(query_embeddings=[query_embedding], n_results=5)

        matched_hackathons = []
        if results["ids"][0]:  # Check if hackathons were found
            for i, hackathon_id in enumerate(results["ids"][0]):  
                hackathon_metadata = hackathons_collection.get(ids=[hackathon_id])["metadatas"][0]

                name = hackathon_metadata.get("name", "Unknown Hackathon")
                required_skills = hackathon_metadata.get("required_skills", "No skills found")

                matched_hackathons.append({
                    "hackathon_id": hackathon_id,
                    "name": name,
                    "required_skills": required_skills,
                    "score": results["distances"][0][i]
                })

        # Save to JSON file
        json_filename = "matched_hackathons.json"
        with open(json_filename, "w") as f:
            json.dump({"matched_hackathons": matched_hackathons}, f, indent=4)

        return {"message": "Results stored in JSON", "json_file": json_filename, "matched_hackathons": matched_hackathons}

    except Exception as e:
        return {"error": str(e)}

# Quiz request model
class QuizRequest(BaseModel):
    topic: str
    difficulty: str  # Can be "easy", "medium", or "hard"
    num_questions: int = 5  # Default to 5 questions

def generate_quiz(topic, difficulty, num_questions):
    """Generate a multiple-choice quiz using AI."""
    
    prompt = f"""
    Generate a **{num_questions}-question multiple-choice quiz** on the topic "{topic}" with a **{difficulty}** difficulty level.
    
    **Rules**:
    - Output **ONLY JSON**, no explanations or markdown.
    - Each question must have exactly **4 choices**, with one correct answer.
    - Format:
      {{
        "questions": [
            {{
                "question": "What is the capital of France?",
                "choices": ["Berlin", "Paris", "Rome", "Madrid"],
                "answer": "Paris"
            }},
            ...
        ]
      }}
    """

    try:
        response = ollama.chat(
            model="mistral",  # Change to the AI model you prefer
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response['message']['content'].strip()

        # Extract only JSON content
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            json_content = match.group(0)
        else:
            return {"error": "No valid JSON found in AI response", "raw_output": raw_content}

        # Parse JSON to dictionary
        quiz_data = json.loads(json_content)

        # Ensure correct number of questions
        if "questions" in quiz_data and isinstance(quiz_data["questions"], list):
            quiz_data["questions"] = quiz_data["questions"][:num_questions]

        return quiz_data  # ✅ Return dictionary

    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from AI", "raw_output": raw_content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate-quiz/")
async def create_quiz(request: QuizRequest):
    """API endpoint to generate a quiz."""
    try:
        quiz = generate_quiz(request.topic, request.difficulty, request.num_questions)
        return {"status": "success", "quiz": quiz}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
