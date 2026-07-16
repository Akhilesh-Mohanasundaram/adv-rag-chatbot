import streamlit as st
from langchain_helper import (
    add_documents_to_vector_db,
    answer_question,
    create_vector_db,
    load_csv_documents_from_bytes,
    load_pdf_documents_from_bytes,
    load_vector_db,
)
import hashlib
import time
from pathlib import Path



st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.caption("Ask questions from your uploaded knowledge base")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_hashes" not in st.session_state:
    st.session_state.uploaded_file_hashes = set()

# Make sure the base vector database exists before the chat starts.
if "vector_db_ready" not in st.session_state:
    load_vector = None
    try:
        load_vector = load_vector_db()
    except Exception:
        create_vector_db()
        load_vector = load_vector_db()
    st.session_state.vector_db_ready = load_vector is not None

# Sidebar
with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or CSV documents",
        type=["pdf", "csv"],
        accept_multiple_files=True,
    )

    if st.button("Process Uploaded Documents"):
        if not uploaded_files:
            st.warning("Choose at least one PDF or CSV file first.")
        else:
            ingested_documents = []
            skipped_files = []

            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.sha256(file_bytes).hexdigest()

                if file_hash in st.session_state.uploaded_file_hashes:
                    skipped_files.append(uploaded_file.name)
                    continue

                suffix = Path(uploaded_file.name).suffix.lower()
                if suffix == ".pdf":
                    ingested_documents.extend(
                        load_pdf_documents_from_bytes(file_bytes, uploaded_file.name)
                    )
                elif suffix == ".csv":
                    ingested_documents.extend(
                        load_csv_documents_from_bytes(file_bytes, uploaded_file.name)
                    )
                else:
                    skipped_files.append(uploaded_file.name)
                    continue

                st.session_state.uploaded_file_hashes.add(file_hash)

            if ingested_documents:
                with st.spinner("Updating the knowledge base..."):
                    indexed_count = add_documents_to_vector_db(ingested_documents)
                st.success(f"Indexed {indexed_count} new document chunks.")
            else:
                st.info("No new documents were added.")

            if skipped_files:
                st.caption(f"Skipped: {', '.join(skipped_files)}")

    if st.button("Create Knowledge Base"):
        with st.spinner("Creating vector database..."):
            create_vector_db()
            st.session_state.vector_db_ready = True
        st.success("Knowledge base created successfully!")

# Display previous chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question...")

if prompt:
    chat_history = list(st.session_state.messages)

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(prompt, chat_history)
            answer = result["result"]
            placeholder = st.empty()

            response = ""
            for word in answer.split():
                response += word + " "
                placeholder.markdown(response + "▌")
                time.sleep(0.02)

            placeholder.markdown(response)
            if result.get("source_documents"):
                with st.expander("📚 Sources"):
                    for doc in result["source_documents"]:
                        st.write(doc.page_content)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )