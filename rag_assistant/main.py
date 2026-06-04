import argparse
import logging

import qdrant_client
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from rag_assistant.config import Settings
from rag_assistant.ingest import load_documents, split_documents, upload_to_qdrant
from rag_assistant.llm import extract_context, generate_answer, extract_sources
from rag_assistant.retriever import create_query_embedding, search_qdrant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

DEMO_QUERY = "что такое FastAPI?"


def create_client(settings):
    logger.info("Connecting to Qdrant...")
    return qdrant_client.QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def create_collection_if_missing(client, settings):
    if client.collection_exists(settings.collection_name):
        logger.info("Collection already exists")
        return

    logger.info("Creating collection...")
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(
            size=settings.vector_size,
            distance=Distance.COSINE
        )
    )


def reindex_collection(client, settings):
    if client.collection_exists(settings.collection_name):
        logger.info("Deleting existing collection...")
        client.delete_collection(settings.collection_name)

    create_collection_if_missing(client, settings)


def load_embedding_model(settings):
    logger.info("Loading embedding model...")
    return SentenceTransformer(settings.embedding_model)


def ingest_documents(client, model, settings):
    create_collection_if_missing(client, settings)

    logger.info("Loading documents...")
    documents = load_documents("./rag_assistant/docs")
    logger.info(f"Loaded {len(documents)} documents")

    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap
    )
    logger.info(f"Created {len(chunks)} chunks")

    upload_to_qdrant(
        chunks,
        model,
        client,
        settings.collection_name
    )
    logger.info("Documents uploaded successfully")


def ask_question(client, model, settings, query):
    if not client.collection_exists(settings.collection_name):
        raise RuntimeError("Collection does not exist. Run ingest or reindex first.")

    query_embedding = create_query_embedding(
        model,
        query
    )
    results = search_qdrant(
        client,
        settings.collection_name,
        query_embedding,
        top_k=settings.top_k
    )

    context = extract_context(results, min_score=settings.min_score)
    answer = generate_answer(query, context, settings.llm_api_key)
    sources = extract_sources(results)
    
    print("\n=== ANSWER ===\n")
    print(answer)
    print("\n=== SOURCES ===")
    for s in sources:
        print(f"- {s['source']}, {s['chunk_id']} ")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest", "ask", "reindex"])
    parser.add_argument("query", nargs="?", default=DEMO_QUERY)
    return parser.parse_args()


def main():
    settings = Settings()
    args = parse_args()

    try:
        client = create_client(settings)
        model = load_embedding_model(settings)

        if args.command == "ingest":
            ingest_documents(client, model, settings)
        elif args.command == "reindex":
            reindex_collection(client, settings)
            ingest_documents(client, model, settings)
        elif args.command == "ask":
            ask_question(client, model, settings, args.query)
    except Exception as e:
        logger.exception(f"Application error: {e}")


if __name__ == "__main__":
    main()
