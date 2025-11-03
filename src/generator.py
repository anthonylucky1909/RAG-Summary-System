from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional

class CodeGenerator:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if not prompt or not isinstance(prompt, str):
            return "Invalid prompt: must be a non-empty string"
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="max_length"
            ).to(self.device)  # Fixed: Changed self.dev to self.device

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        except Exception as e:
            return f"Error generating response: {str(e)}"