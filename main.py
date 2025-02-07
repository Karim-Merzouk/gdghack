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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ROADMAP_FILE = "roadmap.json"
EXPRESS_API_URL = "http://localhost:5000/api/skills"

# Request model
class RoadmapRequest(BaseModel):
    skills: list  # List of skills
    career_path: str  # Career path or job title
async def generate_roadmap_endpoint(request: RoadmapRequest):
    """Endpoint that matches Express.js implementation"""
    try:
        roadmap_data = generate_roadmap(request.skills, request.career_path)
        
        if "error" in roadmap_data:
            return {"status": "error", "message": roadmap_data["error"]}
            
        return {
            "status": "success",
            "roadmap": roadmap_data
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def roadmap_to_reactflow(roadmap_data):
    """Convert roadmap JSON into React Flow format."""
    try:
        if not roadmap_data or not isinstance(roadmap_data, dict):
            return {"error": "Invalid roadmap format"}

        G = nx.DiGraph(roadmap_data)
        nodes = [
            {
                "id": node,
                "data": {"label": node},
                "position": {"x": i * 250, "y": i * 100}
            } 
            for i, node in enumerate(G.nodes)
        ]
        
        edges = [
            {
                "id": f"{src}-{tgt}",
                "source": src,
                "target": tgt
            }
            for src, tgts in roadmap_data.items() 
            for tgt in tgts
        ]
        
        return {"nodes": nodes, "edges": edges}
    
    except Exception as e:
        return {"error": str(e)}


@app.post("/generate-roadmap/")
# async def create_roadmap(request: RoadmapRequest):
#     """Main endpoint for roadmap generation"""
#     try:
#         express_skills = get_skills_from_express()
#         combined_skills = express_skills + request.skills
        
#         roadmap_data = generate_roadmap(combined_skills, request.career_path)
#         react_flow_data = roadmap_to_reactflow(roadmap_data)
        
#         return {
#             "status": "success",
#             "roadmap": roadmap_data,
#             "react_flow": react_flow_data,
#             "file": ROADMAP_FILE
#         }
        
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

@app.get("/get-roadmap")
async def get_raw_roadmap():
    """Endpoint for Express.js to fetch raw JSON"""
    try:
        with open(ROADMAP_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Roadmap not found"}
    except Exception as e:
        return {"error": str(e)}

class RoadmapFieldRequest(BaseModel):
    field: str  # Field (topic) the user wants details about

def get_field_details(field):
    """Generate a description and learning resources for a specific roadmap field using Ollama."""
    prompt = f"""
    Provide a **detailed description** of the topic: "{field}".

    Also, list **useful learning resources** (books, online courses, documentation, websites, etc.) for mastering this topic.

    The response should be in **pure JSON format ONLY**, without any extra text, like this:
    ```
    {{
        "description": "Detailed explanation of the field...",
        "resources": [
            {{"name": "Resource Name", "url": "https://example.com", "type": "book/course/documentation"}}
        ]
    }}
    ```
    Do NOT include any extra explanation, introductions, or notes. Only return the JSON response.
    """

    try:
        response = ollama.chat(
            model="mistral",  # You can replace this with any Ollama model
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response['message']['content'].strip()

        # 🔹 Extract only JSON content
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            raw_content = match.group(0)

        # 🔹 Remove invalid control characters
        cleaned_content = re.sub(r'[\x00-\x1F]+', '', raw_content)

        # 🔹 Validate JSON format
        details = json.loads(cleaned_content)

        # 🔹 Save JSON to a file
        filename = f"{field.replace(' ', '_').lower()}_details.json"
        with open(filename, "w") as f:
            json.dump(details, f, indent=4)

        return {"file": filename, "details": details}

    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON response from LLM", "raw_output": raw_content}

    except Exception as e:
        return {"error": str(e)}

@app.post("/get-field-details/")
async def get_field_details_api(request: RoadmapFieldRequest):
    """Fetch description & resources for a roadmap field and save as JSON."""
    try:
        result = get_field_details(request.field)
        return {"status": "success", "field": request.field, **result}

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

# Get a list of existing collection names (Chroma v0.6.0+ only returns names)
collection_names = chroma_client.list_collections()
hackathons_collection = chroma_client.get_or_create_collection(name="hackathon_opportunities")

# Check if the collection exists
if "users_skills" not in collection_names:
    print("Collection does not exist. Creating a new one...")
    collection = chroma_client.get_or_create_collection(name="users_skills")
else:
    collection = chroma_client.get_collection(name="users_skills")  # Use get_collection directly


import uuid

class UserSkills(BaseModel):
    username: str  # User's display name
    skills: list  # List of skills

import re
MAX_EMBEDDING_DIM = 200  # Choose an appropriate fixed size

def get_embedding(text):
    """Generate numerical embeddings and ensure fixed-length format."""
    prompt = f"Convert the following skills into a numeric vector:\n{text}\nOutput only a space-separated list of numbers."
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

    # Extract and convert to a list of floats
    embedding_text = response['message']['content'].strip()
    embedding = [float(num) for num in re.findall(r"-?\d+\.\d+|-?\d+", embedding_text)]

    # 🔹 Adjust the embedding size:
    if len(embedding) > MAX_EMBEDDING_DIM:
        embedding = embedding[:MAX_EMBEDDING_DIM]  # Truncate if too long
    else:
        embedding += [0.0] * (MAX_EMBEDDING_DIM - len(embedding))  # Pad if too short

    return embedding


@app.post("/store-skills/")
def store_skills(data: UserSkills):
    """Store user skills in ChromaDB, ensuring unique user IDs."""
    try:
        skills_text = " ".join(data.skills)
        embedding = get_embedding(skills_text)

        # Check if username exists
        existing_users = collection.get(where={"username": data.username})

        if existing_users["ids"]:
            user_id = existing_users["ids"][0]  # Use existing user_id
        else:
            user_id = str(uuid.uuid4())  # Generate new UUID

        # 🔹 Store data in ChromaDB
        collection.add(
            ids=[user_id],
            embeddings=[embedding],
            metadatas=[{
                "username": data.username,  # Make sure username is stored
                "skills": ", ".join(data.skills)
            }]
        )

        return {
            "message": "Skills stored successfully",
            "user_id": user_id,
            "username": data.username,
            "dimension": len(embedding),
            "collection_count": collection.count()
        }

    except Exception as e:
        return {"error": str(e)}







@app.post("/find-similar/")
def find_similar(data: UserSkills):
    """Find up to 5 similar users based on skills and store results in JSON format."""
    try:
        skills_text = " ".join(data.skills)
        query_embedding = get_embedding(skills_text)

        # 🔹 Query ChromaDB for similar users
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5  # Ensure max 5 users
        )

        similar_users = []
        if results["ids"][0]:  # Check if users were found
            for i, user_id in enumerate(results["ids"][0]):  
                user_metadata = collection.get(ids=[user_id])["metadatas"][0]  # Fetch metadata

                # Handle missing metadata
                username = user_metadata.get("username", "Unknown User")
                skills = user_metadata.get("skills", "No skills found")

                similar_users.append({
                    "user_id": user_id,
                    "username": username,
                    "skills": skills,
                    "score": results["distances"][0][i]  # Similarity score
                })

        # 🔹 Save to JSON file
        json_filename = "similar_users.json"
        with open(json_filename, "w") as f:
            json.dump({"similar_users": similar_users}, f, indent=4)

        return {
            "message": "Similar users found and stored in JSON",
            "json_file": json_filename,
            "similar_users": similar_users
        }

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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
