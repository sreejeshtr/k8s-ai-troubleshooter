import streamlit as st
import time
import ollama

st.set_page_config(
    page_title="KubeSage",
    page_icon="☸️",
    layout="wide"
)

st.title("☸️ KubeSage")
st.markdown(
    "AI-powered Kubernetes and application troubleshooting assistant"
)

uploaded_file = st.file_uploader(
    "Upload a log file",
    type=["txt", "log"]
)

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

if uploaded_file:

    logs = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Log Preview")
        st.text_area(
            "",
            logs[:5000],
            height=400
        )

    with col2:

        if st.button("Analyze Log"):

            prompt = f"""
You are a senior Kubernetes and OpenShift support engineer.

Analyze the log and provide:

1. Root Cause
2. Evidence from log
3. Severity (Low/Medium/High/Critical)
4. Recommended Fix
5. Next Troubleshooting Steps

Log:

{logs}
"""

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

            st.success(
                f"Analysis completed in {end - start:.2f} seconds"
            )

            st.subheader("Analysis")

            st.markdown(analysis)

            st.download_button(
                "📥 Download Analysis",
                analysis,
                file_name="analysis.txt",
                mime="text/plain"
            )
