from haystack.components.builders import PromptBuilder
from typing import List
from haystack import Document

class RAGPromptBuilder:
    def __init__(self):
        self.template = """Answer the question based on the Rick and Morty episode information below.
        
        Context:
        {% for document in documents %}
        Episode: {{ document.meta.get('filename', 'Unknown') }}
        {{ document.content }}
        {% endfor %}
        
        Question: {{ question }}
        Answer:"""
        
        self.builder = PromptBuilder(template=self.template)

    def build_prompt(self, question: str, documents: List[Document]) -> str:
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        if not documents or not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("Documents must be a non-empty list of Document objects")
        
        try:
            # Debug print inputs
            print("\nBuilding prompt with:")
            print(f"Question: {question}")
            print(f"Document count: {len(documents)}")
            if documents:
                print(f"First document content preview: {documents[0].content[:100]}...")
                print(f"First document meta: {documents[0].meta}")
            
            # Build the prompt
            result = self.builder.run(
                question=question,
                documents=documents
            )
            
            prompt = result.get("prompt", "")
            print("\nGenerated prompt preview:")
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
            
            return prompt
            
        except Exception as e:
            raise RuntimeError(f"Error building prompt: {str(e)}")