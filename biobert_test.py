import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class BioBERTEmbedding:
    def __init__(self):
        print("Loading BioBERT model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use PROVEN working Bio_ClinicalBERT
        MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        
        self.model.to(self.device)
        self.model.eval()
        print("✓ Bio_ClinicalBERT loaded successfully!")
        print(f"Using device: {self.device}")
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = self.mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def get_embeddings(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = self.mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]
    
    def similarity_matrix(self, embeddings):
        return cosine_similarity(embeddings)

if __name__ == "__main__":
    embedding_model = BioBERTEmbedding()
    
    text1 = "Patient presents with erythematous plaques and silvery scales."
    text2 = "Chronic inflammatory skin condition consistent with psoriasis."
    text3 = "Acute bacterial infection with pus formation."
    
    emb1 = embedding_model.get_embedding(text1)
    emb2 = embedding_model.get_embedding(text2)
    emb3 = embedding_model.get_embedding(text3)
    
    print("\nEmbedding dimension:", emb1.shape[0])  # 768
    
    sim_1_2 = embedding_model.compute_similarity(emb1, emb2)
    sim_1_3 = embedding_model.compute_similarity(emb1, emb3)
    
    print("\nSimilarity Results:")
    print(f"Text1 vs Text2 (Expected High): {sim_1_2:.4f}")
    print(f"Text1 vs Text3 (Expected Lower): {sim_1_3:.4f}")
    
    texts = [text1, text2, text3]
    embeddings = embedding_model.get_embeddings(texts)
    sim_matrix = embedding_model.similarity_matrix(embeddings)
    
    print("\nSimilarity Matrix:")
    print(sim_matrix)
