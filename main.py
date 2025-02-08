from dotenv import load_dotenv
import os
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import chromadb
import uuid

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file!")

# Initialize Groq client with API key (all LLM calls will be online)
client = Groq(api_key=GROQ_API_KEY)
model_id = "llama3-8b-8192"  # Example model ID; adjust as needed

# Initialize FastAPI app and enable CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 1. ONLINE ROADMAP GENERATION
# ================================

class RoadmapRequest(BaseModel):
    skills: list
    career_path: str

def generate_roadmap(skills, career_path):
    """Generate a detailed, oriented roadmap using Groq's online LLM."""
  
    prompt = f"""
    Given my skills {skills} and career goal {career_path}, create a **detailed, step-by-step roadmap**.
    Ignore unrelated skills (e.g., if the goal is "Web Developer," exclude "Machine Learning").

    **Rules**:
    - Output **ONLY JSON**, no explanations, no markdown.
    - JSON format:
      {{
        "Phase 1: Fundamentals": ["Key topic 1", "Key topic 2", "Certification", "Relevant Tools", "Book", "Project"],
        "Phase 2: Intermediate": ["Key topic 3", "Key topic 4", "Certification", "Relevant Tools", "Book", "Project"],
        "Phase 3: Advanced": ["Key topic 5", "Key topic 6", "Certification", "Relevant Tools", "Book", "Project"],
        "Phase 4: Expert & Industry Readiness": ["Key topic 7", "Key topic 8", "Certification", "Relevant Tools", "Book", "Capstone Project"]
      }}
    - Each phase should include:
      - Key topics to learn
      - Certifications or projects
      - Industry tools
      - Learning resources (books, courses, etc.)
      - A project at the end of every phase to apply skills and get hands-on experience
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model_id,
            messages=messages
        )
        raw_content = response.choices[0].message.content.strip()
        # Extract only JSON content
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            json_content = match.group(0)
            return json.loads(json_content)
        else:
            return {"error": "No valid JSON found in AI response", "raw_output": raw_content}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from AI", "raw_output": raw_content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/get-roadmap")
async def create_roadmap_endpoint(request: RoadmapRequest):
    """API endpoint for roadmap generation."""
    try:
        roadmap_data = generate_roadmap(request.skills, request.career_path)
        return {"roadmap": roadmap_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# 2. ONLINE FIELD DETAILS GENERATION
# ================================

class RoadmapFieldRequest(BaseModel):
    field: str

def get_field_details(field):
    """Generate a detailed description and learning resources for a field using Groq's online LLM."""
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
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model_id,
            messages=messages
        )
        raw_content = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            json_content = match.group(0)
            details = json.loads(json_content)
            # Ensure at least 5 resources are provided
            if "resources" in details and isinstance(details["resources"], list):
                while len(details["resources"]) < 5:
                    details["resources"].append({"name": "Additional Resource", "url": "#", "type": "general"})
            return details
        return {"error": "No valid JSON found", "raw_output": raw_content}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from AI", "raw_output": raw_content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/get-field-details")
async def get_field_details_api(request: RoadmapFieldRequest):
    """API endpoint for retrieving field details."""
    try:
        details = get_field_details(request.field)
        return {"field": request.field, "details": details}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ================================
# 3. ONLINE SKILL EMBEDDINGS & CHROMADB INTEGRATION
# ================================

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection_names = chroma_client.list_collections()
hackathons_collection = chroma_client.get_or_create_collection(name="hackathon_opportunities")

# Check if the users_skills collection exists
if "users_skills" not in collection_names:
    print("Collection does not exist. Creating a new one...")
    collection = chroma_client.get_or_create_collection(name="users_skills")
else:
    collection = chroma_client.get_collection(name="users_skills")

class UserSkills(BaseModel):
    profileImg: str  # Unique identifier for the user
    username: str  # User's display name
    skills: list   # List of skills

MAX_EMBEDDING_DIM = 200  # Fixed size for embeddings

def get_embedding(text):
    """Generate numerical embeddings using Groq's online LLM."""
    prompt = f"Convert the following skills into a numeric vector:\n{text}\nOutput only a space-separated list of numbers."
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model_id,
            messages=messages
        )
        embedding_text = response.choices[0].message.content.strip()
        embedding = [float(num) for num in re.findall(r"-?\d+\.\d+|-?\d+", embedding_text)]
        if len(embedding) > MAX_EMBEDDING_DIM:
            embedding = embedding[:MAX_EMBEDDING_DIM]
        else:
            embedding += [0.0] * (MAX_EMBEDDING_DIM - len(embedding))
        return embedding
    except Exception as e:
        return {"error": str(e)}

@app.post("/store-skills")
def store_skills(data: UserSkills):
    """Store user skills in ChromaDB."""
    try:
        skills_text = " ".join(data.skills)
        embedding = get_embedding(skills_text)
        if isinstance(embedding, dict) and "error" in embedding:
            return embedding

        existing_users = collection.get(where={"username": data.username})
        if existing_users["ids"]:
            user_id = existing_users["ids"][0]
        else:
            user_id = str(uuid.uuid4())

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
    """Find up to 5 similar users based on skills."""
    try:
        skills_text = " ".join(data.skills)
        query_embedding = get_embedding(skills_text)
        if isinstance(query_embedding, dict) and "error" in query_embedding:
            return query_embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        similar_users = []
        if results["ids"][0]:
            for i, user_id in enumerate(results["ids"][0]):
                user_metadata = collection.get(ids=[user_id])["metadatas"][0]
                username = user_metadata.get("username", "Unknown User")
                profileImg = user_metadata.get("profileImg", "https://example.com/profile.png")
                skills = user_metadata.get("skills", "No skills found")
                similar_users.append({
                    "profileImg": profileImg,
                    "username": username,
                    "skills": skills,
                    "score": results["distances"][0][i]
                })
        return similar_users
    except Exception as e:
        return {"error": str(e)}

# ================================
# 4. ONLINE HACKATHON OPPORTUNITIES
# ================================

class HackathonOpportunity(BaseModel):
    name: str
    required_skills: list

@app.post("/add-hackathon/")
def add_hackathon(data: HackathonOpportunity):
    """Store hackathon opportunities in ChromaDB."""
    try:
        skills_text = " ".join(data.required_skills)
        embedding = get_embedding(skills_text)
        if isinstance(embedding, dict) and "error" in embedding:
            return embedding
        hackathon_id = str(uuid.uuid4())
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

class HUSkills(BaseModel):
    username: str
    skills: list

@app.post("/find-hackathons")
def find_hackathons(data: HUSkills):
    """Find hackathons a user qualifies for based on skills."""
    try:
        skills_text = " ".join(data.skills)
        query_embedding = get_embedding(skills_text)
        if isinstance(query_embedding, dict) and "error" in query_embedding:
            return query_embedding

        results = hackathons_collection.query(query_embeddings=[query_embedding], n_results=5)
        matched_hackathons = []

        if results["ids"][0]:
            for i, hackathon_id in enumerate(results["ids"][0]):
                score = results["distances"][0][i]  # Score from query

                if score > 70:  # ✅ Only include hackathons with score > 70
                    hackathon_metadata = hackathons_collection.get(ids=[hackathon_id])["metadatas"][0]
                    name = hackathon_metadata.get("name", "Unknown Hackathon")
                    required_skills = hackathon_metadata.get("required_skills", "No skills found")

                    matched_hackathons.append({
                        "hackathon_id": hackathon_id,
                        "name": name,
                        "required_skills": required_skills,
                        "score": score
                    })

        return { "matched_hackathons": matched_hackathons}
    except Exception as e:
        return {"error": str(e)}


# ================================
# 5. ONLINE QUIZ GENERATION
# ================================

class QuizRequest(BaseModel):
    topic: str
    difficulty: str  # "easy", "medium", or "hard"
    num_questions: int = 5

def generate_quiz(topic, difficulty, num_questions):
    """Generate a multiple-choice quiz using Groq's online LLM."""
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
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model_id,
            messages=messages
        )
        raw_content = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            json_content = match.group(0)
            quiz_data = json.loads(json_content)
            if "questions" in quiz_data and isinstance(quiz_data["questions"], list):
                quiz_data["questions"] = quiz_data["questions"][:num_questions]
            return quiz_data
        else:
            return {"error": "No valid JSON found in AI response", "raw_output": raw_content}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from AI", "raw_output": raw_content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate-quiz")
async def create_quiz(request: QuizRequest):
    """API endpoint to generate a quiz."""
    try:
        quiz = generate_quiz(request.topic, request.difficulty, request.num_questions)
        return {"status": "success", "quiz": quiz}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ================================
# RUN FASTAPI SERVER
# ================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
