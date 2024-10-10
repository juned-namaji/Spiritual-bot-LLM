import os
import re
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import PineconeEmbeddings
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Initialize FastAPI app
app = FastAPI()


# Function to read HTML file and extract paragraphs
def read_html_file(file_path):
    try:
        with open(file_path, "r", encoding="windows-1252") as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return [p.replace("\xa0", " ") for p in paragraphs]
    except Exception as e:
        print(f"Error reading the file: {e}")
        return []


# Clean text by removing unwanted characters
def clean_text(text_list):
    cleaned_list = []
    for text in text_list:
        text = re.sub(r"\s+", " ", text).strip().replace("\n", " ").replace("\t", "")
        cleaned_list.append(text)
    return cleaned_list


# Remove unwanted text patterns
def remove_unwanted_text(text):
    unwanted_patterns = [r"ओम शान्ति", r"अव्यक्त बापदादा", r"मधुबन"]
    for pattern in unwanted_patterns:
        text = re.sub(pattern, "", text)
    return text.strip()


# Extract date from text
def extract_date(text):
    date_patterns = [
        r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b",
        r"\b\d{2}[-/.]\d{2}[-/.]\d{2}\b",
        r"\b\d{4}[-/.]\d{2}[-/.]\d{2}\b",
        r"\b\d{2}\.\d{2}\.\d{4}\b",
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            return date_match.group()
    return "Date not found"


# Extract title from paragraphs
def extract_title(paragraphs):
    common_phrases = [
        "Morning Murli",
        "Om Shanti",
        "BapDada",
        "Madhuban",
        "ओम शान्ति",
        "अव्यक्त बापदादा",
        "मधुबन",
    ]
    meaningful_lines = [
        p.strip()
        for p in paragraphs
        if p.strip() and not any(phrase in p for phrase in common_phrases)
    ]
    return meaningful_lines[0] if meaningful_lines else "Title not found"


# Extract details from the text
def extract_details(text):
    details = {}
    patterns = {
        "Essence": r"Essence\s*:(.*?)\s*Question",
        "Question": r"Question\s*:(.*?)\s*Answer",
        "Answer": r"Answer\s*:(.*?)\s*Essence for dharna",
        "Essence for Dharna": r"Essence for dharna\s*:(.*?)\s*Blessing",
        "Blessing": r"Blessing\s*:(.*?)\s*Slogan",
        "Slogan": r"Slogan\s*:(.+)$",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        details[key] = match.group(1).strip() if match else f"{key} not found"

    return details


# Process the HTML file to extract relevant data
def process_file(file_path):
    text_array = read_html_file(file_path)
    cleaned_text_list = clean_text(text_array)
    date = "Date not found"
    title = "Title not found"

    if cleaned_text_list:
        date_line = remove_unwanted_text(cleaned_text_list[0])
        date = extract_date(date_line)
        title = extract_title(cleaned_text_list)
        content = [line for line in cleaned_text_list[1:] if line != title]

        details = extract_details("\n".join(content))
        details["Date"] = date
        details["Title"] = title
        details["Content"] = "\n".join(content)

        result = (
            f"Date: {details['Date']}\n"
            f"Title: {details['Title']}\n"
            f"Content:\n{details['Content']}\n"
            f"Essence: {details['Essence']}\n"
            f"Question: {details['Question']}\n"
            f"Answer: {details['Answer']}\n"
            f"Essence for Dharna: {details['Essence for Dharna']}\n"
            f"Blessing: {details['Blessing']}\n"
            f"Slogan: {details['Slogan']}\n"
        )

        return result
    return "No data found."


async def setup_pinecone_and_embed_documents(data):
    index_name = "pinecone"
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    wrapped_docs = [Document(page_content=data)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    docs = text_splitter.split_documents(wrapped_docs)

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_API_ENV")),
        )

    docsearch = LangChainPinecone.from_documents(
        docs, embeddings, index_name=index_name
    )
    return docsearch


file_path = os.path.join(os.path.dirname(__file__), "murli.htm")
extracted_data = process_file(file_path)


@app.get("/")
def hello_world():
    return {"message": "Hello, World"}


@app.get("/chunks")
async def get_chunks(query: str = "shiv ratri or shiv jayanti"):
    docsearch = await setup_pinecone_and_embed_documents(extracted_data)
    docs = await docsearch.similarity_search(query)

    if docs:
        chunk_list = [
            f"Top {i + 1} Chunk: {docs[i].page_content}"
            for i in range(min(10, len(docs)))
        ]
        return {"chunks": chunk_list}
    else:
        return {"message": "No results found."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
