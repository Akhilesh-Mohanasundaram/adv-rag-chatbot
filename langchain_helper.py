from __future__ import annotations

import io
import os
import tempfile
from typing import Iterable, Sequence

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

BASE_DATASET_PATH = "dataset/unsaidtalks_prompt_response_500.csv"
VECTOR_DB_PATH = "faiss_index"


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def _read_csv_dataframe(source: str | io.BytesIO) -> pd.DataFrame:
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, encoding="utf-8", encoding_errors="ignore")


def _dataframe_to_documents(df: pd.DataFrame, source_name: str) -> list[Document]:
    documents: list[Document] = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        if {"prompt", "response"}.issubset(df.columns):
            content = f"Question: {row['prompt']}\n\nAnswer: {row['response']}"
        else:
            row_text = "\n".join(
                f"{column}: {row[column]}" for column in df.columns if pd.notna(row[column])
            )
            content = f"Row {row_number} from {source_name}\n\n{row_text}"

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "type": "csv",
                    "row": row_number,
                },
            )
        )

    return documents


def _pdf_file_to_documents(file_path: str, source_name: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for index, document in enumerate(documents, start=1):
        document.metadata.update(
            {
                "source": source_name,
                "type": "pdf",
                "page": index,
            }
        )
    return documents


def _build_base_documents() -> list[Document]:
    df = _read_csv_dataframe(BASE_DATASET_PATH)
    return _dataframe_to_documents(df, os.path.basename(BASE_DATASET_PATH))


def _index_exists() -> bool:
    return os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss"))


def load_vector_db() -> FAISS:
    if not _index_exists():
        create_vector_db()

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def create_vector_db() -> None:
    documents = _build_base_documents()
    vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectordb.save_local(VECTOR_DB_PATH)


def add_documents_to_vector_db(documents: Sequence[Document]) -> int:
    if not documents:
        return 0

    vectordb = load_vector_db()
    vectordb.add_documents(list(documents))
    vectordb.save_local(VECTOR_DB_PATH)
    return len(documents)


def load_csv_documents_from_bytes(file_bytes: bytes, source_name: str) -> list[Document]:
    dataframe = _read_csv_dataframe(io.BytesIO(file_bytes))
    return _dataframe_to_documents(dataframe, source_name)


def load_pdf_documents_from_bytes(file_bytes: bytes, source_name: str) -> list[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        return _pdf_file_to_documents(temp_path, source_name)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def format_chat_history(messages: Iterable[dict[str, str]]) -> str:
    formatted_lines: list[str] = []
    for message in messages:
        role = message.get("role", "").strip().capitalize() or "User"
        content = message.get("content", "").strip()
        if content:
            formatted_lines.append(f"{role}: {content}")
    return "\n".join(formatted_lines)


def answer_question(question: str, chat_history: Sequence[dict[str, str]]) -> dict:
    vectordb = load_vector_db()
    source_documents = vectordb.similarity_search(question, k=4)
    context = "\n\n".join(document.page_content for document in source_documents)
    history_text = format_chat_history(chat_history)

    prompt = f"""You are a helpful retrieval-augmented assistant.
Use the conversation history and retrieved knowledge base context below to answer the user's question.
Only answer from the provided context and the conversation history. If the answer is not present, say exactly: I don't know.

Conversation history:
{history_text or 'No previous conversation.'}

Retrieved context:
{context or 'No relevant documents found.'}

User question:
{question}

Answer:"""

    response = llm.invoke(prompt)
    return {
        "result": response.content,
        "source_documents": source_documents,
    }


if __name__ == "__main__":
    create_vector_db()
    print(answer_question("Do you have javascript course?", []))