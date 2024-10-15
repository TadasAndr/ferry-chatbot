from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from backend.config import config
from langchain_core.documents import Document


def get_pinecone():
    return Pinecone(api_key=config.PINECONE_API_KEY)


def truncate_string(string, max_bytes):
    encoded = string.encode('utf-8')
    return encoded[:max_bytes].decode('utf-8', errors='ignore')


def create_vector_store(index_name, chunks, model='text-embedding-ada-002'):
    pc = get_pinecone()
    embeddings = OpenAIEmbeddings(model=model)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index, embeddings, "text")

    # Convert chunks to Document objects with truncated content
    max_metadata_size = 40000  # Leave some room for other metadata
    documents = [Document(page_content=truncate_string(chunk, max_metadata_size)) for chunk in chunks]

    vector_store.add_documents(documents)

    return vector_store


def load_vector_store(index_name, model='text-embedding-ada-002', dimension=1536):
    pc = get_pinecone()
    embeddings = OpenAIEmbeddings(model=model)

    print(f"Attempting to load index '{index_name}'")

    try:
        # Try to get the index directly instead of checking if it exists
        index = pc.Index(index_name)
        print(f"Successfully connected to existing index '{index_name}'")
    except PineconeApiException as e:
        if e.status_code == 404:  # Index not found
            print(f"Index '{index_name}' does not exist. Creating it now.")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"Index '{index_name}' created successfully.")
            index = pc.Index(index_name)
        else:
            raise  # Re-raise the exception if it's not a 404 error

    vector_store = PineconeVectorStore(index, embeddings, "text")
    print(f"Vector store loaded successfully. Stats: {index.describe_index_stats()}")
    return vector_store


def delete_vector_store_index(index_name='all'):
    pc = get_pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        for index in indexes:
            pc.delete_index(index)
    else:
        pc.delete_index(index_name)