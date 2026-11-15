import streamlit as st
import time
import ollama
import sys

# Allow importing modules from backend directory
sys.path.append("backend")

from retriever import retrieve_context

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="KubeSage",
    page_icon="☸️",
    layout="wide"
)

# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------
st.title("☸️ KubeSage")

st.markdown(
    "AI-powered Kubernetes and application troubleshooting assistant"
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a log file",
    type=["txt", "log"]
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

st.info(f"Using model: {model}")

# --------------------------------------------------
# PROCESS FILE
# --------------------------------------------------
if uploaded_file:

    # Read uploaded log
    logs = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------
    with col1:

        st.subheader("Log Preview")

        st.text_area(
            "",
            logs[:5000],
            height=400
        )

    # --------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------
    with col2:

        if st.button("Analyze Log"):

            # ------------------------------------------
            # Retrieve relevant documentation
            # ------------------------------------------
            context, ranking = retrieve_context(logs)

            # ------------------------------------------
            # Build RAG Prompt
            # ------------------------------------------
            prompt = f"""
You are a senior Kubernetes and OpenShift support engineer.

Use the documentation below when analyzing the logs.

===================
DOCUMENTATION
===================

{context}

===================
LOGS
===================

{logs}

Provide:

1. Root Cause
2. Evidence From Logs
3. Severity (Low / Medium / High / Critical)
4. Recommended Fix
5. Next Troubleshooting Steps
6. Documentation References
"""

            # ------------------------------------------
            # Call Ollama
            # ------------------------------------------
            with st.spinner("Analyzing logs..."):

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

            analysis = response["message"]["content"]

            # ------------------------------------------
            # Analysis Results
            # ------------------------------------------
            st.success(
                f"Analysis completed in {end - start:.2f} seconds"
            )

            st.subheader("Analysis")

            st.markdown(analysis)

            # ------------------------------------------
            # Retrieved Documents Ranking
            # ------------------------------------------
            st.subheader("Retrieved Documents")

            for item in ranking:

                st.write(
                    f"#{item['rank']} | "
                    f"📄 {item['source']} | "
                    f"Distance: {item['distance']:.4f} | "
                    f"Confidence: {item['confidence']}%"
                )

                st.progress(
                    min(item["confidence"] / 100, 1.0)
                )

            # ------------------------------------------
            # Retrieved Context (Debug Section)
            # ------------------------------------------
            with st.expander("View Retrieved Context"):

                st.code(context)

            # ------------------------------------------
            # Download Analysis
            # ------------------------------------------
            st.download_button(
                label="📥 Download Analysis",
                data=analysis,
                file_name="analysis.txt",
                mime="text/plain"
            )
