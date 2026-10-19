
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_ollama import ChatOllama


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "retail_knowledge"

# Local embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def index_documents():
    """Load, split, embed, and store knowledge documents."""

    files = list(KNOWLEDGE_BASE_DIR.glob("*.md"))

    if not files:
        raise FileNotFoundError(
            f"No Markdown documents found in {KNOWLEDGE_BASE_DIR}"
        )

    documents = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name},
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )

    # Replace the existing collection contents on each re-index.
    existing = vector_store.get()
    existing_ids = existing.get("ids", [])

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_documents(chunks)

    print(f"Indexed {len(files)} documents.")
    print(f"Created {len(chunks)} chunks.")


def retrieve_documents(question: str, k: int = 3):
    """Retrieve the most relevant knowledge chunks."""

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )

    results = vector_store.similarity_search(question, k=k)

    return [
        {"content": document.page_content,
        "source": document.metadata.get("source", "unknown")}
        for document in results
    ]


def run_rag_tool(question: str):
    """Retrieve relevant documents and generate a grounded answer."""

    results = retrieve_documents(question)

    if not results:
        return {
            "question": question,
            "answer": "I could not find relevant information in the knowledge base.",
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['content']}"
        for item in results
    )

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
    )

    prompt = f"""
    You are a retail sales knowledge assistant.

    Answer the user's question using only the context below.
    If the context does not contain the answer, say that the
    available documentation does not provide enough information.
    Do not invent facts.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "sources": sorted({
            item["source"] for item in results
        }),
    }