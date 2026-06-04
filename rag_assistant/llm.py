import logging
from openai import OpenAI
from rag_assistant.config import Settings

logger = logging.getLogger(__name__)
settings = Settings()

MODEL_NAME = settings.model_name

def extract_context(results, min_score: float = 0.5) -> str:
    """
    Извлекает только релевантные чанки.
    """
    chunks = []

    for point in results.points:
        if point.score >= min_score:
            text = point.payload.get("text", "")
            chunks.append(text)

    return "\n\n".join(chunks)

def generate_answer(query: str, context: str, api_key: str) -> str | None:
    """
    Generates an answer using retrieved RAG context.

    Args:
        query: User question
        context: Retrieved document chunks
        api_key: Groq API key

    Returns:
        Generated answer or None if an error occurs
    """
    if not query.strip():
        logger.error("Empty query provided")
        return None
    if not context:
        logger.warning("No context provided")
        return "Не удалось найти релевантную информацию."

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        prompt = f"""
        Context:
        {context}

        Question:
        {query}

        Instructions:
        - Answer only in Russian.
        - Use ONLY the provided context.
        - If the answer is not found in the context, say:
        "Ответ не найден в предоставленном контексте."
        - Give a concise but complete answer.
        """
        logger.info("Sending request to LLM")
        response = client.responses.create(
            model=MODEL_NAME,
            input=query,
            instructions=prompt,
        )
        answer = response.output_text.strip()
        logger.info(
            f"LLM request successful. Response length: {len(answer)} chars"
        )
        return answer

    except Exception as e:
        logger.exception(f"Error during LLM request: {e}")
        return None
    
def extract_sources(results) -> list[dict]:
    sources = []
    for point in results.points:
        source = point.payload.get("source")
        chunk_id = point.payload.get("chunk_id")

        source_data = {
            "source": source,
            "chunk_id": chunk_id,
        }

        if source_data not in sources:
            sources.append(source_data)

    return sources
