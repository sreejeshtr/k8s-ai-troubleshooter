import os
import sys
import time

import ollama
import streamlit as st

# Backend imports
sys.path.append("backend")

from ingest import ingest_documents
from retriever import retrieve_context

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ChatRagGo",
    page_icon="☸️",
    layout="wide"
)

# --------------------------------------------------
# CREATE REQUIRED FOLDERS
# --------------------------------------------------

os.makedirs(
    "data/uploaded_docs",
    exist_ok=True
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("☸️ ChatRagGo")

st.markdown(
    """
An AI-powered assistant to chat with AI, search knowledge bases, and analyze documents.

### Supported Modes

**Ask AI**
- Ask the selected AI model directly

**Search Knowledge Base**
- Search indexed documents using ChromaDB

**Analyze Uploaded Documents**
- Upload documents, generate embeddings and query them immediately
"""
)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

model = st.selectbox(
    "Select Model",
    [
        "qwen3:4b",
        "llama3.2:latest",
        "deepseek-r1:1.5b",
        "granite-code:8b"
    ]
)

st.info(f"Current Model: {model}")

# --------------------------------------------------
# ANALYSIS MODE
# --------------------------------------------------

mode = st.radio(
    "Analysis Mode",
    [
        "Ask AI",
        "Search Knowledge Base",
        "Analyze Uploaded Documents"
    ]
)

# --------------------------------------------------
# USER PROMPT
# --------------------------------------------------

user_prompt = st.text_area(
    "Prompt",
    placeholder="Enter your question...",
    height=150
)

# --------------------------------------------------
# DOCUMENT UPLOAD MODE
# --------------------------------------------------

if mode == "Analyze Uploaded Documents":

    st.subheader("📚 Upload Documents")

    uploaded_docs = st.file_uploader(
        "Choose one or more documents",
        type=["txt"],
        accept_multiple_files=True
    )

    if uploaded_docs:

        for file in uploaded_docs:

            with open(
                f"data/uploaded_docs/{file.name}",
                "wb"
            ) as f:

                f.write(
                    file.getbuffer()
                )

        st.success(
            f"{len(uploaded_docs)} document(s) uploaded"
        )

        if st.button(
            "🚀 Generate Embeddings & Ingest"
        ):

            with st.spinner(
                "Creating embeddings..."
            ):

                count = ingest_documents(
                    "data/uploaded_docs"
                )

            st.success(
                f"{count} document(s) indexed in ChromaDB"
            )

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button("🔍 Analyze"):

    if not user_prompt.strip():

        st.warning(
            "Please enter a prompt."
        )

        st.stop()

    ranking = []
    context = ""

    # ==================================================
    # ASK AI
    # ==================================================

    if mode == "Ask AI":

        prompt = user_prompt

    # ==================================================
    # SEARCH KNOWLEDGE BASE
    # ==================================================

    elif mode == "Search Knowledge Base":

        context, ranking = retrieve_context(
            user_prompt
        )

        prompt = f"""
You are an AI assistant.

Use the following knowledge base content
to answer the user's question.

KNOWLEDGE BASE

{context}

QUESTION

{user_prompt}
"""

    # ==================================================
    # ANALYZE UPLOADED DOCUMENTS
    # ==================================================

    elif mode == "Analyze Uploaded Documents":

        context, ranking = retrieve_context(
            user_prompt
        )

        prompt = f"""
Use the uploaded documents
to answer the user's question.

DOCUMENTS

{context}

QUESTION

{user_prompt}
"""

    # ==================================================
    # CALL MODEL
    # ==================================================

    with st.spinner("Thinking..."):

        start = time.time()

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        end = time.time()

    answer = response["message"]["content"]

    # ==================================================
    # RESULTS
    # ==================================================

    st.success(
        f"Completed in {end - start:.2f} seconds"
    )

    st.subheader("✅ Answer")

    st.markdown(answer)

    # ==================================================
    # SHOW RETRIEVED DOCUMENTS
    # ==================================================

    if mode != "Ask AI" and ranking:

        st.subheader("📚 Retrieved Documents")

        for item in ranking:

            st.write(
                f"#{item['rank']} | "
                f"{item['source']} | "
                f"Distance: {item['distance']:.4f} | "
                f"Confidence: {item['confidence']}%"
            )

            st.progress(
                min(
                    item["confidence"] / 100,
                    1.0
                )
            )

    # ==================================================
    # SHOW RETRIEVED CONTEXT
    # ==================================================

    if mode != "Ask AI" and context:

        with st.expander(
            "🔍 View Retrieved Context"
        ):

            st.code(context)

    # ==================================================
    # DOWNLOAD RESULTS
    # ==================================================

    st.download_button(
        label="📥 Download Answer",
        data=answer,
        file_name="analysis.txt",
        mime="text/plain"
    )
