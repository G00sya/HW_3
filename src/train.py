from src.utils.shared_embedding import create_pretrained_embedding

if __name__ == "__main__":
    # Initialize SharedEmbedding with glove embedding
    shared_embedding = create_pretrained_embedding(path="./embeddings/glove.6B.300d.txt", padding_idx=0)
