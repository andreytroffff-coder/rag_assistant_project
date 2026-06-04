import pathlib
import re
import logging
from qdrant_client.models import PointStruct

def clean_text(file: str) -> str:
    '''Clean text by removing extra whitespace and trimming.'''
    try:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            logging.info(f"Loaded file: {file}")
    except Exception as e:
        logging.error(f"Error reading file {file}: {e}")
        text = ""
    text = re.sub(r'\s+', ' ', text)   
    text = text.strip()
    return text

def load_documents(directory: str):
    """Load documents from a directory."""
    documents = []
    for file in pathlib.Path(directory).glob("**/*"):
        if file.is_file():
            text = clean_text(file)
            documents.append({
                    "source": file.name,
                    "text": text
                })
    logging.info(f"Loaded {len(documents)} documents from {directory}")
    return documents

def split_documents(documents: list[str], chunk_size: int = 200, overlap: int = 30
) -> list[str]:
    """
    Split documents into overlapping chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    chunk_id = 0
    for doc in documents:
        source = doc["source"]
        text = doc["text"]
        for i in range(0, len(text), step):

            chunk = text[i:i + chunk_size]

            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": source,
                    "chunk_id": chunk_id
                })

                chunk_id += 1

    return chunks

def create_embeddings(model, chunks):
    """Create embeddings only from chunk text, not metadata."""
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts)

def upload_to_qdrant(chunks, model, client, collection_name):
    """Upload chunks to Qdrant."""
    embeddings = create_embeddings(model, chunks)
    points = [PointStruct(id=i, 
                          vector=emb.tolist(), 
                          payload={
                                "text": chunk["text"],
                                "source": chunk["source"],
                                "chunk_id": chunk["chunk_id"]
                            })
                          for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upload_points(collection_name, points)

   
