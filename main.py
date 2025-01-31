import streamlit as st
import os
from vector_db import process_xlsx
from rag_engine import get_qa_chain

# Set page configuration
st.set_page_config(page_title="Document QA System", page_icon="📚", layout="wide")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# App title and description
st.title("Document QA System 📚")
st.markdown("### Upload XLSX files and ask questions about their content")

# File upload section
uploaded_file = st.file_uploader("Choose an XLSX file", type="xlsx")
if uploaded_file is not None:
    # Save file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Process file
    with st.spinner('Processing file...'):
        process_xlsx(file_path)
    os.remove(file_path)
    st.success("File processed successfully!")

# Question answering section
question = st.text_input("Ask a question about your documents:")
if question:
    with st.spinner("Finding answer..."):
        qa_chain = get_qa_chain()
        result = qa_chain.answer_question(question)
        st.session_state.history.append({
            "question": question,
            "answer": result
        })

# Display conversation history
st.markdown("---")
if st.session_state.history:
    for interaction in reversed(st.session_state.history):
        st.write(f"**Question:** {interaction['question']}")
        st.write(f"**Answer:** {interaction['answer']}")
        st.markdown("---")

# Sidebar information
st.sidebar.header("About")
st.sidebar.write("""
This application allows you to:
1. Upload XLSX files for processing
2. Ask questions about the content
3. Get AI-powered answers using AI21's models
""")
