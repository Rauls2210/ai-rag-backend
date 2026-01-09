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



ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')


# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]
print(f"Connected to MongoDB at {mongo_url}, using database: {os.environ['GEMINI_API_KEY']}")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Global FAISS index and document chunks storage
faiss_index = None
document_chunks = []
chunk_metadata = []


# Create the main app without a prefix
app = FastAPI()


# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")



# Define Models
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: str  # 'user' or 'assistant'
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



# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are an AI assistant for a Retrieval-Augmented Generation (RAG) chatbot that answers questions using retrieved context documents from a vector database.


Primary Objective:
Provide accurate, concise, and well-structured answers by combining user queries with retrieved context documents. Always prioritize factual accuracy over speculation.


Response Framework:
1. Understand the Query - Read the user's question carefully and identify key information needs
2. Analyze Retrieved Context - Review all provided document chunks and identify relevant passages
3. Generate Response - Start with a direct 1-2 sentence answer, follow with structured explanation


Grounding Rules:
- When Context is Available: Use retrieved documents as your primary source of truth
- When Context is Partial: Combine retrieved information with general knowledge and state confidence level
- When Context is Insufficient: Explicitly state that available documents don't contain the information
- When Sources Conflict: Acknowledge disagreement and choose the most reliable source


Formatting:
- Use markdown headers (##, ###) for section organization
- Use bullet points for lists
- Keep responses focused and scannable
- Professional, neutral tone"""



# Helper functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting PDF: {e}")
        return ""


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from Word document"""
    try:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error extracting DOCX: {e}")
        return ""


def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from plain text file"""
    try:
        return file_content.decode('utf-8')
    except Exception as e:
        logging.error(f"Error extracting TXT: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index from embeddings"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[dict]:
    """Retrieve most relevant chunks for a query"""
    global faiss_index, document_chunks, chunk_metadata

    if faiss_index is None or len(document_chunks) == 0:
        return []

    # Encode query
    query_embedding = embedding_model.encode([query])

    # Search FAISS index
    distances, indices = faiss_index.search(query_embedding, min(top_k, len(document_chunks)))

    # Retrieve chunks with metadata
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(document_chunks):
            results.append({
                'chunk': document_chunks[idx],
                'metadata': chunk_metadata[idx],
                'distance': float(distance)
            })

    return results



# Routes
@api_router.get("/")
async def root():
    return {"message": "RAG Chatbot API"}


@api_router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    global faiss_index, document_chunks, chunk_metadata

    try:
        # Read file content
        file_content = await file.read()

        # Extract text based on file type
        filename = file.filename.lower()
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_content)
            file_type = 'pdf'
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_content)
            file_type = 'docx'
        elif filename.endswith('.txt'):
            text = extract_text_from_txt(file_content)
            file_type = 'txt'
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF, DOCX, and TXT files are supported.")

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        # Chunk the text
        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Document could not be split into chunks.")

        # Generate embeddings
        chunk_embeddings = embedding_model.encode(chunks)

        # Store chunks and metadata
        doc_id = str(uuid.uuid4())
        for i, chunk in enumerate(chunks):
            document_chunks.append(chunk)
            chunk_metadata.append({
                'doc_id': doc_id,
                'filename': file.filename,
                'chunk_index': i,
                'file_type': file_type
            })

        # Rebuild FAISS index
        all_embeddings = embedding_model.encode(document_chunks)
        faiss_index = build_faiss_index(all_embeddings)

        # Save document info to MongoDB
        doc_info = DocumentInfo(
            id=doc_id,
            filename=file.filename,
            file_type=file_type,
            chunk_count=len(chunks)
        )
        doc_dict = doc_info.model_dump()
        doc_dict['uploaded_at'] = doc_dict['uploaded_at'].isoformat()
        await db.documents.insert_one(doc_dict)

        return {
            'success': True,
            'document_id': doc_id,
            'filename': file.filename,
            'chunks_created': len(chunks),
            'total_documents': len(set([m['doc_id'] for m in chunk_metadata]))
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@api_router.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get list of uploaded documents"""
    docs = await db.documents.find({}, {"_id": 0}).to_list(1000)

    for doc in docs:
        if isinstance(doc['uploaded_at'], str):
            doc['uploaded_at'] = datetime.fromisoformat(doc['uploaded_at'])

    return docs


@api_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks"""
    global faiss_index, document_chunks, chunk_metadata

    # Remove chunks from memory
    indices_to_remove = [i for i, m in enumerate(chunk_metadata) if m['doc_id'] == doc_id]

    if not indices_to_remove:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove in reverse order to maintain indices
    for i in sorted(indices_to_remove, reverse=True):
        document_chunks.pop(i)
        chunk_metadata.pop(i)

    # Rebuild FAISS index
    if document_chunks:
        all_embeddings = embedding_model.encode(document_chunks)
        faiss_index = build_faiss_index(all_embeddings)
    else:
        faiss_index = None

    # Remove from MongoDB
    await db.documents.delete_one({"id": doc_id})

    return {'success': True, 'message': 'Document deleted'}


@api_router.post("/conversations", response_model=Conversation)
async def create_conversation():
    """Create a new conversation"""
    conversation = Conversation()
    conv_dict = conversation.model_dump()
    conv_dict['created_at'] = conv_dict['created_at'].isoformat()
    conv_dict['updated_at'] = conv_dict['updated_at'].isoformat()

    await db.conversations.insert_one(conv_dict)
    return conversation


@api_router.get("/conversations", response_model=List[Conversation])
async def get_conversations():
    """Get all conversations"""
    conversations = await db.conversations.find({}, {"_id": 0}).sort("updated_at", -1).to_list(1000)

    for conv in conversations:
        if isinstance(conv['created_at'], str):
            conv['created_at'] = datetime.fromisoformat(conv['created_at'])
        if isinstance(conv['updated_at'], str):
            conv['updated_at'] = datetime.fromisoformat(conv['updated_at'])

    return conversations


@api_router.get("/conversations/{conversation_id}/messages", response_model=List[ChatMessage])
async def get_conversation_messages(conversation_id: str):
    """Get messages for a conversation"""
    messages = await db.messages.find(
        {"conversation_id": conversation_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(1000)

    for msg in messages:
        if isinstance(msg['timestamp'], str):
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])

    return messages


@api_router.post("/chat")
async def chat(request: ChatMessageCreate):
    """Send a message and get AI response"""
    try:
        # Save user message
        user_msg = ChatMessage(
            conversation_id=request.conversation_id,
            role='user',
            content=request.message
        )
        user_dict = user_msg.model_dump()
        user_dict['timestamp'] = user_dict['timestamp'].isoformat()
        await db.messages.insert_one(user_dict)
        
        # Retrieve relevant context
        relevant_chunks = retrieve_relevant_chunks(request.message, top_k=5)
        
        # Build context string
        context = ""
        if relevant_chunks:
            context = "RETRIEVED CONTEXT:\n"
            for i, result in enumerate(relevant_chunks, 1):
                context += f"[Document {i} - {result['metadata']['filename']}]:\n{result['chunk']}\n\n"
        else:
            context = "RETRIEVED CONTEXT: No relevant documents found in the knowledge base.\n\n"
        
        # Build full prompt
        full_prompt = f"{context}\nUSER QUESTION: {request.message}"
        
        # Initialize Gemini client with explicit API key
        gemini_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        
        # Use async context manager for proper resource cleanup
        async with gemini_client.aio as aclient:
            # Generate content using async client
            response = await aclient.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_PROMPT,
                    temperature=0.7,
                ),
            )
            response_text = response.text
        
        # Save assistant message
        assistant_msg = ChatMessage(
            conversation_id=request.conversation_id,
            role='assistant',
            content=response_text
        )
        assistant_dict = assistant_msg.model_dump()
        assistant_dict['timestamp'] = assistant_dict['timestamp'].isoformat()
        await db.messages.insert_one(assistant_dict)
        
        # Update conversation timestamp
        await db.conversations.update_one(
            {"id": request.conversation_id},
            {"$set": {"updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        return {
            'success': True,
            'message': response_text,
            'sources_used': len(relevant_chunks)
        }
    
    except Exception as e:
        logging.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@api_router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    await db.conversations.delete_one({"id": conversation_id})
    await db.messages.delete_many({"conversation_id": conversation_id})

    return {'success': True, 'message': 'Conversation deleted'}



# Include the router in the main app
app.include_router(api_router)


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()