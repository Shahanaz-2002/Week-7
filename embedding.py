# embedding.py

import numpy as np
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel


# 🔹 Singleton BioBERT Model (LOAD ONLY ONCE)
class BioBERTEmbedding:

    _instance = None  # singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BioBERTEmbedding, cls).__new__(cls)

            cls._instance.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            cls._instance.MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

            print("🔄 Loading BioClinicalBERT model... (only once)")

            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                cls._instance.MODEL_NAME
            )
            cls._instance.model = AutoModel.from_pretrained(
                cls._instance.MODEL_NAME
            )

            cls._instance.model.to(cls._instance.device)
            cls._instance.model.eval()

        return cls._instance


    # 🔹 Mean Pooling
    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask


    # 🔹 Get embedding for text
    def get_embedding(self, text: str) -> np.ndarray:

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = self.mean_pooling(
            outputs,
            inputs["attention_mask"]
        ).cpu().numpy()[0]

        # 🔹 Normalize (IMPORTANT for cosine similarity)
        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm
        else:
            embedding = np.zeros_like(embedding)

        return embedding


# 🔹 Embedding Engine
class EmbeddingEngine:

    def __init__(self, embedding_dim: int = 768):

        self.embedding_dim = embedding_dim

        # Singleton model (no repeated loading)
        self.embedding_model = BioBERTEmbedding()

        self.embedding_model_name = "BioClinicalBERT"
        self.embedding_version = "v1"


    # 🔹 Generate embedding (FOR RETRIEVAL)
    def generate_embedding(self, case_data: Dict[str, Any]) -> np.ndarray:

        processed_text = self._preprocess_case(case_data)

        embedding_vector = self.embedding_model.get_embedding(processed_text)

        return embedding_vector


    # 🔹 Generate embedding for DB storage
    def generate_embedding_for_storage(self, case_data: Dict[str, Any]) -> Dict[str, Any]:

        embedding_vector = self.generate_embedding(case_data)

        return {
            "embedding": embedding_vector.tolist(),
            "embedding_model": self.embedding_model_name,
            "embedding_version": self.embedding_version
        }


    # 🔹 Convert case → text
    def _preprocess_case(self, case_data: Dict[str, Any]) -> str:

        symptoms = case_data.get("symptoms", [])
        diagnosis = case_data.get("diagnosis", "")
        notes = case_data.get("doctor_notes", case_data.get("notes", ""))

        combined_text = " ".join(symptoms) + " " + diagnosis + " " + notes

        return combined_text.lower().strip()


# 🔹 TEST BLOCK
if __name__ == "__main__":

    sample_case = {
        "symptoms": ["itching", "red rash"],
        "notes": "Patient has red itchy rash on arm",
        "diagnosis": "eczema"
    }

    engine = EmbeddingEngine()

    emb = engine.generate_embedding(sample_case)

    print("\nEmbedding shape:", emb.shape)
    print("First 5 values:", emb[:5])