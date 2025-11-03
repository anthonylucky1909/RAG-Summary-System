from src.rag_pipeline import RAGSystem
from src.generator import CodeGenerator

def main(question: str):
    # Initialize RAG system with data
    rag_system = RAGSystem(data_path="./data/rick_and_morty_episodes/")
    
    # Get prompt from RAG
    print("Question :",question)
    prompt = rag_system.query(question)
    print("Part 2 :",prompt)
    
    # Generate answer
    generator = CodeGenerator()
    result = generator.generate(prompt)
    
    # Output results
    print(f"Prompt:\n{prompt}")
    print("\nGenerated:\n" + result)

if __name__ == "__main__":
    question = "who is Ryan Ridley ?"
    main(question)
