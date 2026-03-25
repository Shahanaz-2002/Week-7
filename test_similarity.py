from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from embedding import EmbeddingEngine


def run_similarity_validation():

    engine = EmbeddingEngine()

    # Similar dermatology cases
    similar_cases = [
        {
            "symptoms": ["itching", "red rash"],
            "notes": "Red itchy rash on the forearm",
            "diagnosis": ""
        },
        {
            "symptoms": ["skin itching", "red patches"],
            "notes": "Itchy red patches on the arm",
            "diagnosis": ""
        },
        {
            "symptoms": ["inflamed skin", "itching"],
            "notes": "Inflamed red rash with itching on forearm",
            "diagnosis": ""
        }
    ]

    # Unrelated cases
    unrelated_cases = [
        {
            "symptoms": ["chest pain", "shortness of breath"],
            "notes": "Possible cardiac issue",
            "diagnosis": ""
        },
        {
            "symptoms": ["headache", "blurred vision"],
            "notes": "Patient reports dizziness and headache",
            "diagnosis": ""
        },
        {
            "symptoms": ["knee pain", "joint swelling"],
            "notes": "Pain after sports injury",
            "diagnosis": ""
        }
    ]

    all_cases = similar_cases + unrelated_cases

    print("\nGenerating embeddings...\n")

    embeddings = []
    for case in all_cases:
        emb = engine.generate_embedding(case)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    similarity_matrix = cosine_similarity(embeddings)

    print("\nSimilarity Matrix:\n")
    print(similarity_matrix)


if __name__ == "__main__":
    run_similarity_validation()