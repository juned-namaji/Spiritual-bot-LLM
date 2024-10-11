from dotenv import find_dotenv, load_dotenv
import os
import re
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pinecone
from fastapi import FastAPI

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PINECONE_API_KEY = "624cd15e-b2fc-4e1b-99f4-71e0edb92447"
PINECONE_API_ENV = "us-east-1"
INDEX_NAME = "pinecone"

app = FastAPI()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


# Pinecone Initialization
def initialize_pinecone():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENV),
        )
    return PineconeEmbeddings(model="multilingual-e5-large")


initialize_pinecone()


@app.get("/query/{query}")
async def query_pinecone(query: str):

    # Use the Pinecone index for embedding
    embedding = PineconeEmbeddings(model="multilingual-e5-large").embed_query(query)

    # Query the index
    index = pinecone.Index(
        index_name=INDEX_NAME,
        host="https://pinecone-ztbvcbv.svc.aped-4627-b74a.pinecone.io",
        api_key=PINECONE_API_KEY,
    )

    results = index.query(  # Query the index
        vector=embedding, top_k=10, include_values=False, include_metadata=True
    )

    # Extract and return the top 10 chunks
    top_chunks = results.get("matches", [])
    response = []
    for i, chunk in enumerate(top_chunks, start=1):
        response.append(
            {
                "rank": i,
                "date": chunk["metadata"].get("date", "N/A"),
                "title": chunk["metadata"].get("title", "N/A"),
                "text": chunk["metadata"].get("text", "N/A"),
            }
        )

    return response
