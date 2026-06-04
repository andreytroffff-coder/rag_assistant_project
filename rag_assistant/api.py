from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from rag_assistant.config import Settings
from rag_assistant.ingest import (
    load_documents,
    split_documents,
    upload_to_qdrant,
)
from rag_assistant.retriever import (
    create_query_embedding,
    search_qdrant,
)
from rag_assistant.llm import extract_context, generate_answer, extract_sources


import qdrant_client
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Assistant API",
    description="Simple RAG service with Qdrant and LLM",
    version="1.0.0",
)

settings = Settings()

client = qdrant_client.QdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)

embedding_model = SentenceTransformer(
    settings.embedding_model
)

class QuestionRequest(BaseModel):
    query: str


class QuestionResponse(BaseModel):
    answer: str
    


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "collection": settings.collection_name,
    }

@app.post(
    "/ask",
    response_model=QuestionResponse
)
async def ask(request: QuestionRequest):

    try:
        query_embedding = create_query_embedding(
            embedding_model,
            request.query,
        )

        results = search_qdrant(
            client,
            settings.collection_name,
            query_embedding,
            top_k=settings.top_k
        )

        context = extract_context(results, min_score=settings.min_score)

        answer = generate_answer(
            query=request.query,
            context=context,
            api_key=settings.llm_api_key,
        )

        sources = extract_sources(results)

        return QuestionResponse(
            answer=answer
            
        )

    except Exception as e:
        logger.exception("Error while processing request")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

@app.post("/reindex")
async def reindex():

    try:
        documents = load_documents("./rag_assistant/docs")

        chunks = split_documents(
            documents,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        upload_to_qdrant(
            chunks,
            embedding_model,
            client,
            settings.collection_name,
        )

        return {
            "status": "success",
            "chunks_uploaded": len(chunks),
        }

    except Exception as e:
        logger.exception("Reindex failed")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )