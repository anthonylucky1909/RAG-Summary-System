from pathlib import Path
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder


class DocumentProcessor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize document processor with embedding model."""
        self.document_store = InMemoryDocumentStore()
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
        self.doc_embedder.warm_up()

    def load_documents(self, file_path: str) -> None:
        """Load and process a single document."""
        try:
            doc_content = Path(file_path).read_text(encoding='utf-8')
            docs = [Document(content=doc_content, meta={"filename": Path(file_path).name})]
            result = self.doc_embedder.run(docs)
            self.document_store.write_documents(result["documents"])
        except Exception as e:
            raise RuntimeError(f"Error loading document {file_path}: {str(e)}")

    def load_documents_from_folder(self, folder_path: str, pattern: str = "season_*.txt") -> None:
        """Load and process all documents matching pattern in a folder."""
        try:
            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
                
            docs = []
            for file in folder.glob(pattern):
                try:
                    content = file.read_text(encoding='utf-8')
                    docs.append(Document(content=content, meta={"filename": file.name}))
                except Exception as e:
                    print(f"Error processing file {file.name}: {str(e)}")
                    continue

            if not docs:
                print(f"No documents found matching pattern: {pattern}")
                return

            result = self.doc_embedder.run(docs)
            self.document_store.write_documents(result["documents"])
        except Exception as e:
            raise RuntimeError(f"Error loading documents from folder: {str(e)}")


if __name__ == "__main__":
    # For standalone testing
    try:
        processor = DocumentProcessor(embedding_model="all-MiniLM-L6-v2")
        processor.load_documents_from_folder("./data/rick_and_morty_episodes/")

        # Debug print
        all_docs = processor.document_store.filter_documents()
        print(f"Loaded {len(all_docs)} documents")
        if all_docs:
            print("Example file:", all_docs[0].meta["filename"])
            print("Content preview:", all_docs[0].content[:100])
            print("Embedding preview:", all_docs[0].embedding[:5])
    except Exception as e:
        print(f"Error in main execution: {str(e)}")