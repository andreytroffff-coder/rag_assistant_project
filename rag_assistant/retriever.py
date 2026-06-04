def create_query_embedding(model, query):
    """Create embedding for a query."""
    return model.encode(query).tolist()

def search_qdrant(client, collection_name, query_embedding, top_k=5):  
    """Search Qdrant for similar documents."""
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k
    )
    return results


