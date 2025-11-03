from src.document_processor import *
from src.generator import *
from src.query_processor import *
from src.prompt_builder import *
# from document_processor import *
# from generator import *
# from query_processor import *
# from prompt_builder import *


class RAGSystem:
    def __init__(self, data_path: str = "./data"):
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.document_processor.load_documents_from_folder(data_path)

        self.query_processor = QueryProcessor(self.document_processor.document_store)
        self.prompt_builder = RAGPromptBuilder()
        self.code_generator = CodeGenerator()

    def query(self, question: str) -> str:

        relevant_docs = self.query_processor.process_query(question)
        if not relevant_docs:
            return "No relevant documents found."

        prompt = self.prompt_builder.build_prompt(question, relevant_docs)
        if not isinstance(prompt, str):
            raise ValueError(
                f"Prompt must be a string. Got: {type(prompt)}\nPrompt content: {prompt}"
            )
        answer = self.code_generator.generate(prompt)

        return answer


# if __name__ == "__main__":
#     # Initialize the RAG system with data path
#     rag_system = RAGSystem(data_path="./data/rick_and_morty_episodes/")

#     # Example query
#     question = "List all episodes from season 1 with their plot summaries"
#     answer = rag_system.query(question)

#     print(f"Question: {question}")
#     print(f"Answer: {answer}")
