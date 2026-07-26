"""
test_embedding_generator.py

Validation script for the EmbeddingGenerator module.
"""

from modules.embedding_generator import EmbeddingGenerator
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    print("=" * 60)
    print("TESTING EMBEDDING GENERATOR")
    print("=" * 60)

    try:
        # Initialize
        generator = EmbeddingGenerator()
        print("✅ EmbeddingGenerator initialized successfully.")

        # Retrieve model
        embedding_model = generator.get_embedding_model()

        if isinstance(embedding_model, HuggingFaceEmbeddings):
            print("✅ Correct embedding model type.")
        else:
            raise TypeError("Returned object is not HuggingFaceEmbeddings.")

        # Generate embedding
        sample_text = "Hello Hospital"

        embedding = embedding_model.embed_query(sample_text)

        print("✅ Embedding generated successfully.")
        print(f"Embedding Dimension : {len(embedding)}")

        if len(embedding) == 384:
            print("✅ Embedding dimension is correct (384).")
        else:
            raise ValueError(
                f"Unexpected embedding dimension: {len(embedding)}"
            )

        print("\n🎉 Embedding Generator Validation PASSED!")

    except Exception as e:
        print("\n❌ Validation FAILED")
        print(type(e).__name__)
        print(e)


if __name__ == "__main__":
    main()