# query_processor.py
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack import Document

class QueryProcessor:
    def __init__(self, document_store, embedding_model: str = "./all-MiniLM-L6-v2"):
        self.text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)
        self.retriever = InMemoryEmbeddingRetriever(document_store=document_store)
        self.text_embedder.warm_up()  # Load model into memory

    def process_query(self, query: str, top_k: int = 1) -> list[Document]:
        # Generate embedding for the query
        embedding_result = self.text_embedder.run(query)
        
        # Retrieve documents using the embedding
        retrieval_result = self.retriever.run(
            embedding_result["embedding"],
            top_k=top_k
        )
        
        return retrieval_result["documents"]