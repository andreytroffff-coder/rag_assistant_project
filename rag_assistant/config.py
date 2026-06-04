from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_host: str
    qdrant_port: int
    embedding_model: str
    collection_name: str
    llm_api_key: str
    model_name: str
    vector_size: int = 384
    chunk_size: int = 100
    chunk_overlap: int = 20
    top_k: int = 5
    min_score: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


