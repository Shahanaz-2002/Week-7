# api_demo_test.py

import time
import numpy as np
from embedding import EmbeddingEngine
from similarity_engine import SimilarityEngine
from database import fetch_case_embeddings, fetch_case_database
from insight_generator import InsightGenerator


def run_demo():

    print("\nStarting CCMS Semantic Retrieval Demo...\n")

    # Load embedding engine
    
    embedding_engine = EmbeddingEngine()

    
    # Load stored embeddings
    
    case_embeddings = fetch_case_embeddings()

    if not case_embeddings:
        print("No embeddings found in database.")
        return

    similarity_engine = SimilarityEngine(case_embeddings)

    
    # Load full case database ->(for insights)
    
    case_database = fetch_case_database()

    insight_generator = InsightGenerator(case_database)

    
    # Example patient case
    
    query_case = {
        "symptoms": ["itching", "red rash"],
        "diagnosis": "",
        "notes": "skin irritation spreading on arms"
    }

    print("Input Patient Case:")
    print(query_case)

    
    # Start timing
    
    start_time = time.time()

    # Generate embedding
    query_embedding = embedding_engine.generate_embedding(query_case)

    # Retrieve similar cases
    results = similarity_engine.retrieve_top_k(query_embedding, top_k=3)

    # Generate insight summary
    insight = insight_generator.generate_insight(results)

    # End timing
    end_time = time.time()

    response_time = end_time - start_time

    
    # Output results
    
    print("\nTop Similar Cases:")
    for case_id, score in results:
        print(f"{case_id}  | Similarity Score: {score:.4f}")

    print("\nInsight Summary:")
    print(insight)

    print(f"\nResponse Time: {response_time:.4f} seconds")


if __name__ == "__main__":
    run_demo()