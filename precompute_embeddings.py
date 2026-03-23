# precompute_embeddings.py

import numpy as np
from database import fetch_case_database, collection
from embedding import EmbeddingEngine


def precompute_embeddings():

    print("Loading cases from MongoDB...")
    case_database = fetch_case_database()

    embedding_engine = EmbeddingEngine()

    print("Generating embeddings...\n")

    for case_id, case_data in case_database.items():

        embedding = embedding_engine.generate_embedding(case_data)

        collection.update_one(
            {"case_id": case_id},
            {
                "$set": {
                    "embedding": embedding.tolist(),
                    "embedding_version": "v1"
                }
            }
        )

        print(f"Stored embedding for case {case_id}")

    print("\nEmbedding precomputation completed.")


if __name__ == "__main__":
    precompute_embeddings()