from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load local embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    """Artificial Intelligence (AI) is a branch of computer science that focuses on building systems capable of performing tasks that typically require human intelligence, such as reasoning, learning, problem-solving, perception, and decision-making. AI powers technologies like recommendation systems, autonomous vehicles, virtual assistants, fraud detection, and medical diagnosis.""",

    """Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn patterns from data instead of being explicitly programmed. Common machine learning techniques include supervised learning, unsupervised learning, and reinforcement learning. ML is widely used for spam detection, price prediction, customer segmentation, and recommendation systems.""",

    """Deep Learning is a specialized branch of Machine Learning that uses artificial neural networks with multiple hidden layers to automatically learn complex patterns from large datasets. Deep learning models excel in computer vision, speech recognition, natural language processing, and autonomous driving.""",

    """Large Language Models (LLMs) are deep learning models trained on massive amounts of text data to understand and generate human-like language. Popular LLMs include GPT, Gemini, Claude, Llama, Qwen, and Mistral. They can answer questions, summarize documents, write code, translate languages, and assist with reasoning tasks.""",

    """Embeddings are numerical vector representations of text, images, or other data that capture semantic meaning. Similar pieces of information have similar vector representations. Embeddings are the foundation of semantic search, recommendation systems, clustering, Retrieval-Augmented Generation (RAG), and vector databases.""",

    """Vector Databases are specialized databases designed to efficiently store, index, and search high-dimensional embedding vectors. Popular vector databases include Pinecone, ChromaDB, FAISS, Weaviate, Milvus, and Qdrant. They enable fast similarity search for AI applications such as semantic search and recommendation systems.""",

    """Retrieval-Augmented Generation (RAG) is an AI architecture that combines information retrieval with Large Language Models. Instead of relying only on the model's training knowledge, RAG retrieves relevant documents from a vector database and provides them as context before generating an answer. This improves factual accuracy and reduces hallucinations.""",

    """Transformers are the neural network architecture behind modern Large Language Models. Introduced in the paper 'Attention Is All You Need', Transformers rely on the self-attention mechanism to understand relationships between words regardless of their distance in a sentence. This architecture revolutionized Natural Language Processing.""",

    """Fine-tuning is the process of adapting a pre-trained language model to a specific domain or task using additional labeled data. Organizations fine-tune models for applications such as legal document analysis, healthcare assistants, financial forecasting, customer support automation, and enterprise knowledge systems.""",

    """AI Agents are intelligent systems that use Large Language Models together with tools, memory, reasoning, and planning capabilities to accomplish complex tasks autonomously. Modern AI agents can browse the web, interact with APIs, execute Python code, query databases, and collaborate with other agents to solve real-world problems."""
]

query = "what are LLMs"

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Compute cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the most similar document
index, score = max(enumerate(scores), key=lambda x: x[1])

print("Query:")
print(query)

print("\nMost Similar Document:")
print(documents[index])

print("\nSimilarity Score:")
print(score)