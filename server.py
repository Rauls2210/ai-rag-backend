from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import asyncio
import json
import sys

# Document processing imports
from pypdf import PdfReader
from docx import Document
import io

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Google Gemini integration
from google import genai
from google.genai import types

# Load environment variables (safe for Render)
load_dotenv()

print("🚀 Starting FastAPI server...", file=sys.stderr)

# Validate required env vars
required_vars = ["MONGO_URL", "DB_NAME", "GEMINI_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"❌ ERROR: Missing required environment variable: {var}", file=sys.stderr)

# MongoDB connection
mongo_url = os.environ.get("MONGO_URL", "")
db_name = os.environ.get("DB_NAME", "")

client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

print(f"✅ Connected to MongoDB at {mongo_url}, DB: {db_name}", file=sys.stderr)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory vector database
faiss_index = None
document_chunks = []
chunk_metadata = []

# FastAPI application
app = FastAPI()

# API router
api_router = APIRouter(prefix="/api")

# -------------------
# Pydantic Models
# -------------------

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatMessageCreate(BaseModel):
    conversation_id: str
    message: str

class Conversation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Conversation"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DocumentInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# -------------------
# System Prompt for Gemini
# -------------------

RAG_SYSTEM_PROMPT = """
You are an AI assistant for a Retrieval-Augmented Generation (RAG) chatbot.
Provide accurate, concise responses using retrieved document context.
"""

# -------------------
# Helper Functions
# -------------------

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            content = page.extract_text() or ""
            text += content + "\n"
        return text
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_content))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        logging.error(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_txt(file_content: bytes) -> str:
    try:
        return file_content.decode("utf-8")
    except:
        return ""

def chunk_text(text: str, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query: str, top_k=5):
    global faiss_index, document_chunks, chunk_metadata
    if faiss_index is None:
        return []

    query_embed = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embed, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(document_chunks):
            results.append({
                "chunk": document_chunks[idx],
                "metadata": chunk_metadata[idx],
                "distance": float(dist)
            })

    return results

# -------------------
# API Routes
# -------------------

@api_router.get("/")
async def root():
    return {"message": "RAG Chatbot API running successfully!"}

@api_router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    global faiss_index, document_chunks, chunk_metadata

    content = await file.read()
    filename = file.filename.lower()

    # Determine type
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
        file_type = "pdf"
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(content)
        file_type = "docx"
    elif filename.endswith(".txt"):
        text = extract_text_from_txt(content)
        file_type = "txt"
    else:
        raise HTTPException(400, "Unsupported file type.")

    if not text.strip():
        raise HTTPException(400, "No text extracted.")

    chunks = chunk_text(text)

    # Store
    doc_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        document_chunks.append(chunk)
        chunk_metadata.append({
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_index": i,
            "file_type": file_type
        })

    # Rebuild FAISS
    all_embeddings = embedding_model.encode(document_chunks)
    faiss_index = build_faiss_index(all_embeddings)

    # Save metadata
    info = DocumentInfo(
        id=doc_id,
        filename=file.filename,
        file_type=file_type,
        chunk_count=len(chunks)
    )

    doc_dict = info.model_dump()
    doc_dict["uploaded_at"] = doc_dict["uploaded_at"].isoformat()
    await db.documents.insert_one(doc_dict)

    return {"success": True, "document_id": doc_id, "chunks_created": len(chunks)}

@api_router.post("/chat")
async def chat(req: ChatMessageCreate):
    # Save user message
    user_msg = ChatMessage(
        conversation_id=req.conversation_id,
        role="user",
        content=req.message
    )
    await db.messages.insert_one(user_msg.model_dump())

    # Retrieve context
    chunks = retrieve_relevant_chunks(req.message)

    ctx = ""
    for item in chunks:
        ctx += f"{item['chunk']}\n\n"

    prompt = f"{ctx}\nUSER QUESTION: {req.message}"

    # Gemini
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        async with client.aio as aclient:
            response = await aclient.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_PROMPT,
                    temperature=0.6
                )
            )
            answer = response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        raise HTTPException(500, f"Gemini error: {str(e)}")

    # Save assistant message
    assistant_msg = ChatMessage(
        conversation_id=req.conversation_id,
        role="assistant",
        content=answer
    )
    await db.messages.insert_one(assistant_msg.model_dump())

    return {"success": True, "message": answer}

# Delete conversation
@api_router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    await db.conversations.delete_one({"id": conversation_id})
    await db.messages.delete_many({"conversation_id": conversation_id})
    return {"success": True}

# Add router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Shutdown
@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    print("🔌 MongoDB connection closed")
