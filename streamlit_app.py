# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_ai21 import ChatAI21
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.documents import Document
# from uuid import uuid4
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import gc  # Add garbage collection
# import time  # Add time tracking

# # Load environment variables
# load_dotenv()

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Document QA System",
#     page_icon="📚",
#     layout="wide"
# )

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "db_initialized" not in st.session_state:
#     st.session_state.db_initialized = False

# # Constants
# DB_PATH = "chroma_langchain_db"
# COLLECTION_NAME = "document_qa"

# class DocumentQA:
#     def __init__(self):
#         self.api_key = os.getenv("AI21_API_KEY")
#         if not self.api_key:
#             raise ValueError("AI21_API_KEY not found in environment variables")
        
#         # Initialize AI21 chat model
#         self.llm = ChatAI21(
#             model="jamba-1.5-mini",
#             temperature=0.1,
#             api_key=self.api_key
#         )
        
#         # Initialize BGE embeddings
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-small-en-v1.5",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )
        
#         # Initialize vector store
#         self.vectordb = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=self.embeddings,
#             persist_directory=DB_PATH
#         )
        
#         # Add processing settings
#         self.chunk_size = 5  # Process 5 rows at a time
#         self.timeout = 300   # 5 minutes total timeout

#     def clean_column_names(self, df):
#         """Clean and format column names from complex Excel headers"""
#         # Handle multi-level columns if present
#         if isinstance(df.columns, pd.MultiIndex):
#             # Join multi-level column names with underscore
#             df.columns = ['_'.join(str(level) for level in col if str(level) != 'nan').strip() 
#                          for col in df.columns.values]
        
#         # Clean single-level columns
#         df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
#         return df

#     def format_value(self, value):
#         """Format values appropriately based on their type"""
#         if pd.isna(value):
#             return ""
#         elif isinstance(value, (int, float)):
#             if value == 0:
#                 return "0"
#             elif abs(value) < 0.01:
#                 return f"{value:.6f}"
#             elif abs(value) < 1:
#                 return f"{value:.4f}"
#             elif abs(value) >= 1000000:
#                 return f"{value:,.2f}M"
#             elif abs(value) >= 1000:
#                 return f"{value:,.2f}K"
#             else:
#                 return f"{value:.2f}"
#         return str(value)

#     def process_excel(self, file):
#         """Process uploaded Excel file and create vector store"""
#         try:
#             start_time = time.time()
            
#             # Create progress indicators
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             status_text.text("Reading Excel file...")
            
#             # Read file based on extension
#             file_extension = file.name.split('.')[-1].lower()
#             if file_extension == 'xls':
#                 df = pd.read_excel(file, engine='xlrd')
#             else:
#                 df = pd.read_excel(file, engine='openpyxl')
            
#             # Clean column names and convert to numeric where possible
#             df = self.clean_column_names(df)
#             df = df.apply(pd.to_numeric, errors='ignore')
            
#             documents = []
            
#             # Create statistical summary
#             status_text.text("Creating statistical summary...")
#             numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#             stats_summary = ["Numerical Data Statistics:"]
#             for col in numeric_cols:
#                 stats = df[col].describe()
#                 stats_summary.append(f"\n{col}:")
#                 stats_summary.append(f"  Average: {stats['mean']:.2f}")
#                 stats_summary.append(f"  Min: {stats['min']:.2f}")
#                 stats_summary.append(f"  Max: {stats['max']:.2f}")
            
#             # Create summary document
#             summary = [
#                 "Excel File Summary:",
#                 f"Total Rows: {len(df)}",
#                 f"Columns: {', '.join(df.columns)}",
#                 "\nColumn Types:"
#             ]
            
#             for col in df.columns:
#                 dtype = str(df[col].dtype)
#                 summary.append(f"{col}: {dtype}")
            
#             summary.extend(stats_summary)
            
#             summary_doc = Document(
#                 page_content="\n".join(summary),
#                 metadata={"type": "summary", "file_name": file.name}
#             )
#             documents.append(summary_doc)
            
#             # Process rows in chunks with progress tracking
#             total_chunks = len(df) // self.chunk_size + (1 if len(df) % self.chunk_size else 0)
            
#             for i in range(0, len(df), self.chunk_size):
#                 # Check timeout
#                 if time.time() - start_time > self.timeout:
#                     raise TimeoutError("Processing took too long")
                
#                 # Update progress
#                 progress = (i + self.chunk_size) / len(df)
#                 progress_bar.progress(min(progress, 1.0))
#                 status_text.text(f"Processing rows {i+1} to {min(i+self.chunk_size, len(df))}...")
                
#                 chunk = df.iloc[i:i+self.chunk_size]
#                 rows_text = []
                
#                 for idx, row in chunk.iterrows():
#                     row_items = []
#                     for col in df.columns:
#                         value = row[col]
#                         if pd.notna(value):
#                             formatted_value = self.format_value(value)
#                             row_items.append(f"{col}: {formatted_value}")
#                     rows_text.append(" | ".join(row_items))
                
#                 if rows_text:
#                     doc = Document(
#                         page_content="\n".join(rows_text),
#                         metadata={
#                             "type": "data",
#                             "file_name": file.name,
#                             "row_range": f"{i+1}-{min(i+self.chunk_size, len(df))}",
#                             "chunk_id": f"chunk_{i//self.chunk_size}"
#                         }
#                     )
#                     documents.append(doc)
                
#                 # Clean memory after each chunk
#                 gc.collect()
            
#             # Clear progress indicators
#             progress_bar.empty()
#             status_text.empty()
            
#             # Initialize new vector store
#             status_text.text("Creating vector store...")
#             collection_name = f"excel_qa_{uuid4()}"
            
#             # Clear any existing vector store
#             if hasattr(self, 'vectordb'):
#                 del self.vectordb
#                 gc.collect()
            
#             self.vectordb = Chroma(
#                 collection_name=collection_name,
#                 embedding_function=self.embeddings,
#                 persist_directory=DB_PATH
#             )
            
#             # Add documents in batches
#             batch_size = 50
#             for i in range(0, len(documents), batch_size):
#                 batch = documents[i:i+batch_size]
#                 batch_uuids = [str(uuid4()) for _ in range(len(batch))]
#                 self.vectordb.add_documents(documents=batch, ids=batch_uuids)
#                 gc.collect()  # Clean memory after each batch
            
#             status_text.empty()
#             return True
            
#         except TimeoutError as e:
#             st.error(f"Timeout error: {str(e)}")
#             return False
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")
#             return False
#         finally:
#             # Clean up
#             gc.collect()
#             if 'progress_bar' in locals():
#                 progress_bar.empty()
#             if 'status_text' in locals():
#                 status_text.empty()

#     def get_answer(self, question):
#         """Get answer for user question"""
#         try:
#             # Get relevant documents
#             results = self.vectordb.similarity_search(question, k=3)
            
#             # Build context
#             context = "\n---\n".join([doc.page_content for doc in results])
            
#             # Enhanced system prompt to prevent hallucination
#             system_prompt = """You are a data analysis assistant that helps users understand Excel data. 
            
#             IMPORTANT RULES:
#             1. ONLY use information explicitly present in the provided context
#             2. If information is not in the context, say "Based on the provided data, I cannot answer this question" or "This information is not present in the data"
#             3. NEVER make up or infer information that isn't directly in the context
#             4. When listing items from the data, only include items actually present in the context
#             5. For numerical questions, only use numbers that appear in the context
#             6. If you're unsure about any information, express that uncertainty
            
#             Format your responses clearly using:
#             - Bullet points for lists
#             - Clear section headings when appropriate
#             - Direct quotes from the data when relevant"""
            
#             # Query prompt
#             query_prompt = f"""Context from Excel file:
#             {context}

#             Question: {question}

#             Remember: Only provide information that is explicitly present in the context above. Do not make assumptions or infer additional information."""
            
#             # Get response
#             messages = [
#                 SystemMessage(content=system_prompt),
#                 HumanMessage(content=query_prompt)
#             ]
            
#             response = self.llm.invoke(messages)
#             return response.content
            
#         except Exception as e:
#             return f"Error getting answer: {str(e)}"

# # Initialize DocumentQA
# @st.cache_resource
# def get_qa_system():
#     return DocumentQA()

# # Main UI
# st.title("📚 Document QA System")
# st.markdown("### Upload Excel files and ask questions about their content")

# # File upload section
# uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
# if uploaded_file:
#     qa_system = get_qa_system()
#     with st.spinner("Processing document..."):
#         if qa_system.process_excel(uploaded_file):
#             st.success("Document processed successfully!")
#             st.session_state.db_initialized = True
#         else:
#             st.error("Failed to process document.")

# # Question input section
# if st.session_state.db_initialized:
#     question = st.text_input("Ask a question about your document:")
#     if question:
#         qa_system = get_qa_system()
#         with st.spinner("Finding answer..."):
#             answer = qa_system.get_answer(question)
#             st.session_state.chat_history.append({
#                 "question": question,
#                 "answer": answer
#             })

# # Display chat history
# if st.session_state.chat_history:
#     st.markdown("### Chat History")
#     for chat in reversed(st.session_state.chat_history):
#         st.markdown(f"**Q:** {chat['question']}")
#         st.markdown(f"**A:** {chat['answer']}")
#         st.markdown("---")

# # Sidebar
# with st.sidebar:
#     st.header("About")
#     st.markdown("""
#     This application allows you to:
#     - Upload Excel documents
#     - Ask questions about the content
#     - Get AI-powered answers using AI21's models
    
#     The system uses:
#     - AI21 for text generation
#     - ChromaDB for document storage
#     - BGE Small for embeddings
#     - Streamlit for the interface
#     """)


# # Version 2.0 Works! for numerical data too

# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_ai21 import ChatAI21
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.documents import Document
# from uuid import uuid4
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import gc
# import time
# import asyncio

# # Load environment variables
# load_dotenv()

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Document QA System",
#     page_icon="📚",
#     layout="wide"
# )

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "db_initialized" not in st.session_state:
#     st.session_state.db_initialized = False
# if "process_abort" not in st.session_state:
#     st.session_state.process_abort = False

# DB_PATH = "chroma_langchain_db"
# COLLECTION_NAME = "document_qa"

# class DocumentQA:
#     def __init__(self):
#         self.api_key = os.getenv("AI21_API_KEY")
#         if not self.api_key:
#             raise ValueError("AI21_API_KEY not found in environment variables")
        
#         self.llm = ChatAI21(
#             model="jamba-1.5-mini",
#             temperature=0.1,
#             api_key=self.api_key
#         )
        
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-small-en-v1.5",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )
        
#         self.vectordb = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=self.embeddings,
#             persist_directory=DB_PATH
#         )
        
#         # Reduced chunk size for numeric-heavy data and partial updates
#         self.chunk_size = 2
#         # Keep a configurable timeout
#         self.timeout = 180

#     def clean_column_names(self, df):
#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = [
#                 '_'.join(str(level) for level in col if str(level) != 'nan').strip() 
#                 for col in df.columns.values
#             ]
#         df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
#         return df

#     def format_value(self, value):
#         if pd.isna(value):
#             return ""
#         elif isinstance(value, (int, float)):
#             if value == 0:
#                 return "0"
#             elif abs(value) < 0.01:
#                 return f"{value:.6f}"
#             elif abs(value) < 1:
#                 return f"{value:.4f}"
#             elif abs(value) >= 1_000_000:
#                 return f"{value:,.2f}M"
#             elif abs(value) >= 1000:
#                 return f"{value:,.2f}K"
#             else:
#                 return f"{value:.2f}"
#         return str(value)

#     async def process_excel_async(self, df, file_name, start_time, progress_bar, status_text):
#         documents = []
#         numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#         stats_summary = ["Numerical Data Statistics:"]
#         for col in numeric_cols:
#             stats = df[col].describe()
#             stats_summary.append(f"\n{col}:")
#             stats_summary.append(f"  Average: {stats['mean']:.2f}")
#             stats_summary.append(f"  Min: {stats['min']:.2f}")
#             stats_summary.append(f"  Max: {stats['max']:.2f}")

#         summary = [
#             "Excel File Summary:",
#             f"Total Rows: {len(df)}",
#             f"Columns: {', '.join(df.columns)}",
#             "\nColumn Types:"
#         ]
#         for col in df.columns:
#             dtype = str(df[col].dtype)
#             summary.append(f"{col}: {dtype}")

#         summary.extend(stats_summary)
#         if len(numeric_cols) == len(df.columns):
#             summary.append("\nNote: This file contains primarily numeric data. Minimal textual context.")

#         summary_doc = Document(
#             page_content="\n".join(summary),
#             metadata={"type": "summary", "file_name": file_name}
#         )
#         documents.append(summary_doc)

#         total_chunks = len(df) // self.chunk_size + (1 if len(df) % self.chunk_size else 0)

#         for i in range(0, len(df), self.chunk_size):
#             if st.session_state.process_abort:
#                 return []

#             if time.time() - start_time > self.timeout:
#                 raise TimeoutError("Processing took too long")

#             progress = (i + self.chunk_size) / len(df)
#             progress_bar.progress(min(progress, 1.0))
#             status_text.text(f"Processing rows {i+1} to {min(i+self.chunk_size, len(df))}...")

#             chunk = df.iloc[i:i+self.chunk_size]
#             rows_text = []
#             for _, row in chunk.iterrows():
#                 row_items = []
#                 for col in df.columns:
#                     value = row[col]
#                     if pd.notna(value):
#                         formatted_value = self.format_value(value)
#                         row_items.append(f"{col}: {formatted_value}")
#                 rows_text.append(" | ".join(row_items))

#             if rows_text:
#                 doc = Document(
#                     page_content="\n".join(rows_text),
#                     metadata={
#                         "type": "data",
#                         "file_name": file_name,
#                         "row_range": f"{i+1}-{min(i+self.chunk_size, len(df))}",
#                         "chunk_id": f"chunk_{i//self.chunk_size}"
#                     }
#                 )
#                 documents.append(doc)

#             gc.collect()
#             await asyncio.sleep(0.01)  # Minimal await to allow UI updates

#         return documents

#     def process_excel(self, file):
#         try:
#             st.session_state.process_abort = False
#             start_time = time.time()

#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             status_text.text("Reading Excel file...")

#             file_extension = file.name.split('.')[-1].lower()
#             if file_extension == 'xls':
#                 df = pd.read_excel(file, engine='xlrd')
#             else:
#                 df = pd.read_excel(file, engine='openpyxl')

#             df = self.clean_column_names(df)
#             df = df.apply(pd.to_numeric, errors='ignore')

#             async def run_processing():
#                 return await self.process_excel_async(df, file.name, start_time, progress_bar, status_text)

#             documents = asyncio.run(run_processing())

#             if not documents:
#                 status_text.empty()
#                 progress_bar.empty()
#                 st.error("Process aborted.")
#                 return False

#             progress_bar.empty()
#             status_text.empty()

#             status_text.text("Creating vector store...")
#             collection_name = f"excel_qa_{uuid4()}"

#             if hasattr(self, 'vectordb'):
#                 del self.vectordb
#                 gc.collect()

#             self.vectordb = Chroma(
#                 collection_name=collection_name,
#                 embedding_function=self.embeddings,
#                 persist_directory=DB_PATH
#             )

#             batch_size = 50
#             for i in range(0, len(documents), batch_size):
#                 batch = documents[i:i+batch_size]
#                 batch_uuids = [str(uuid4()) for _ in range(len(batch))]
#                 self.vectordb.add_documents(documents=batch, ids=batch_uuids)
#                 gc.collect()

#             status_text.empty()
#             return True
        
#         except TimeoutError as e:
#             st.error(f"Timeout error: {str(e)}")
#             return False
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")
#             return False
#         finally:
#             gc.collect()
#             if 'progress_bar' in locals():
#                 progress_bar.empty()
#             if 'status_text' in locals():
#                 status_text.empty()

#     def get_answer(self, question):
#         try:
#             results = self.vectordb.similarity_search(question, k=3)
#             context = "\n---\n".join([doc.page_content for doc in results])

#             system_prompt = """You are a data analysis assistant that helps users understand Excel data.

# IMPORTANT RULES:
# 1. ONLY use information explicitly present in the provided context
# 2. If information is not in the context, say "Based on the provided data, I cannot answer this question" or "This information is not present in the data"
# 3. NEVER make up or infer information that isn't directly in the context
# 4. When listing items from the data, only include items actually present in the context
# 5. For numerical questions, only use numbers that appear in the context
# 6. If you're unsure about any information, express that uncertainty

# Format your responses clearly using:
# - Bullet points for lists
# - Clear section headings when appropriate
# - Direct quotes from the data when relevant
# """
#             query_prompt = f"""Context from Excel file:
# {context}

# Question: {question}

# Remember: Only provide information that is explicitly present in the context above. Do not make assumptions or infer additional information."""

#             messages = [
#                 SystemMessage(content=system_prompt),
#                 HumanMessage(content=query_prompt)
#             ]
            
#             response = self.llm.invoke(messages)
#             return response.content
#         except Exception as e:
#             return f"Error getting answer: {str(e)}"

# @st.cache_resource
# def get_qa_system():
#     return DocumentQA()

# st.title("📚 Document QA System")
# st.markdown("### Upload Excel files and ask questions about their content")

# uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
# col1, col2 = st.columns(2)
# with col1:
#     if uploaded_file and not st.session_state.db_initialized:
#         if st.button("Start Processing"):
#             qa_system = get_qa_system()
#             with st.spinner("Processing document..."):
#                 success = qa_system.process_excel(uploaded_file)
#                 if success:
#                     st.success("Document processed successfully!")
#                     st.session_state.db_initialized = True
#                 else:
#                     st.error("Failed to process document.")
# with col2:
#     if uploaded_file and not st.session_state.db_initialized:
#         if st.button("Abort Processing"):
#             st.session_state.process_abort = True

# if st.session_state.db_initialized:
#     question = st.text_input("Ask a question about your document:")
#     if question:
#         qa_system = get_qa_system()
#         with st.spinner("Finding answer..."):
#             answer = qa_system.get_answer(question)
#             st.session_state.chat_history.append({
#                 "question": question,
#                 "answer": answer
#             })

# if st.session_state.chat_history:
#     st.markdown("### Chat History")
#     for chat in reversed(st.session_state.chat_history):
#         st.markdown(f"**Q:** {chat['question']}")
#         st.markdown(f"**A:** {chat['answer']}")
#         st.markdown("---")

# with st.sidebar:
#     st.header("About")
#     st.markdown("""
#     This application allows you to:
#     - Upload Excel documents
#     - Ask questions about the content
#     - Get AI-powered answers using AI21's models

#     It uses:
#     - AI21 for text generation
#     - Chroma for document storage
#     - BGE Small for embeddings
#     - Streamlit for the interface
#     """)


#Version 3 - Can find mean, sum, etc of the columns for numerical data.

import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ai21 import ChatAI21
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from uuid import uuid4
from langchain_community.embeddings import HuggingFaceEmbeddings
import gc
import time

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Document QA System",
    page_icon="📚",
    layout="wide"
)

# Keep a single copy of the QA system in session
def init_qa_system():
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = DocumentQA()

# Ensure we only initialize or reinitialize the DB if a new file is uploaded
def process_file_once(file):
    # If we don't have a vectordb in session, or if we are uploading a different file, re-process
    if "latest_file_name" not in st.session_state or st.session_state.latest_file_name != file.name:
        st.session_state.qa_system.process_excel(file)
        st.session_state.latest_file_name = file.name
        st.session_state.db_initialized = True

# Keep track of passing data between reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False
if "latest_file_name" not in st.session_state:
    st.session_state.latest_file_name = None

DB_PATH = "chroma_langchain_db"
COLLECTION_NAME = "document_qa"

class DocumentQA:
    def __init__(self):
        self.api_key = os.getenv("AI21_API_KEY")
        if not self.api_key:
            raise ValueError("AI21_API_KEY not found in environment variables")
        
        # AI21 chat model
        self.llm = ChatAI21(
            model="jamba-1.5-mini",
            temperature=0.1,
            api_key=self.api_key
        )

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Placeholder vector store (will be re-initialized on file upload)
        self.vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=DB_PATH
        )

        self.chunk_size = 5
        self.timeout = 300

    def clean_column_names(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                '_'.join(str(level) for level in col if str(level) != 'nan').strip() 
                for col in df.columns.values
            ]
        df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
        return df

    def format_value(self, value):
        if pd.isna(value):
            return ""
        elif isinstance(value, (int, float)):
            if value == 0:
                return "0"
            elif abs(value) < 0.01:
                return f"{value:.6f}"
            elif abs(value) < 1:
                return f"{value:.4f}"
            elif abs(value) >= 1_000_000:
                return f"{value:,.2f}M"
            elif abs(value) >= 1000:
                return f"{value:,.2f}K"
            else:
                return f"{value:.2f}"
        return str(value)

    def process_excel(self, file):
        try:
            start_time = time.time()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Reading Excel file...")

            file_extension = file.name.split('.')[-1].lower()
            if file_extension == 'xls':
                df = pd.read_excel(file, engine='xlrd')
            else:
                df = pd.read_excel(file, engine='openpyxl')

            df = self.clean_column_names(df)
            df = df.apply(pd.to_numeric, errors='ignore')

            documents = []

            status_text.text("Creating statistical summary...")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            stats_summary = ["Numerical Data Statistics:"]
            for col in numeric_cols:
                stats = df[col].describe()
                col_sum = df[col].sum()
                col_modes = df[col].mode().values
                stats_summary.append(f"\n{col}:")
                stats_summary.append(f"  Count: {int(stats['count'])}")
                stats_summary.append(f"  Sum: {col_sum}")
                stats_summary.append(f"  Mean: {stats['mean']:.2f}")
                stats_summary.append(f"  Min: {stats['min']:.2f}")
                stats_summary.append(f"  Max: {stats['max']:.2f}")
                if len(col_modes) > 0:
                    modes_str = ", ".join([str(m) for m in col_modes])
                    stats_summary.append(f"  Mode(s): {modes_str}")

            summary = [
                "Excel File Summary:",
                f"Total Rows: {len(df)}",
                f"Columns: {', '.join(df.columns)}",
                "\nColumn Types:"
            ]
            for col in df.columns:
                dtype = str(df[col].dtype)
                summary.append(f"{col}: {dtype}")

            summary.extend(stats_summary)

            summary_doc = Document(
                page_content="\n".join(summary),
                metadata={"type": "summary", "file_name": file.name}
            )
            documents.append(summary_doc)
            
            total_chunks = len(df) // self.chunk_size + (1 if len(df) % self.chunk_size else 0)

            for i in range(0, len(df), self.chunk_size):
                if time.time() - start_time > self.timeout:
                    raise TimeoutError("Processing took too long")

                progress = (i + self.chunk_size) / len(df)
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Processing rows {i+1} to {min(i+self.chunk_size, len(df))}...")

                chunk = df.iloc[i:i+self.chunk_size]
                rows_text = []

                for idx, row in chunk.iterrows():
                    row_items = []
                    for col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            formatted_value = self.format_value(value)
                            row_items.append(f"{col}: {formatted_value}")
                    rows_text.append(" | ".join(row_items))

                if rows_text:
                    doc = Document(
                        page_content="\n".join(rows_text),
                        metadata={
                            "type": "data",
                            "file_name": file.name,
                            "row_range": f"{i+1}-{min(i+self.chunk_size, len(df))}",
                            "chunk_id": f"chunk_{i//self.chunk_size}"
                        }
                    )
                    documents.append(doc)
                gc.collect()

            progress_bar.empty()
            status_text.empty()

            status_text.text("Creating vector store...")
            collection_name = f"excel_qa_{uuid4()}"

            # Clear old vector store
            if hasattr(self, 'vectordb'):
                del self.vectordb
                gc.collect()

            self.vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=DB_PATH
            )

            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                batch_uuids = [str(uuid4()) for _ in range(len(batch))]
                self.vectordb.add_documents(documents=batch, ids=batch_uuids)
                gc.collect()

            status_text.empty()
            return True
        except TimeoutError as e:
            st.error(f"Timeout error: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return False
        finally:
            gc.collect()
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()

    def get_answer(self, question):
        try:
            results = self.vectordb.similarity_search(question, k=3)
            context = "\n---\n".join([doc.page_content for doc in results])

            system_prompt = """You are a data analysis assistant that helps users understand Excel data.

IMPORTANT RULES:
1. ONLY use information explicitly present in the provided context
2. If information is not in the context, say "Based on the provided data, I cannot answer this question" or "This information is not present in the data"
3. NEVER make up or infer information that isn't directly in the context
4. When listing items from the data, only include items actually present in the context
5. For numerical questions, only use numbers that appear in the context
6. If you're unsure about any information, express that uncertainty

Format your responses clearly using:
- Bullet points for lists
- Clear section headings when appropriate
- Direct quotes from the data when relevant
"""

            query_prompt = f"""Context from Excel file:
{context}

Question: {question}

Remember: Only provide information that is explicitly present in the context above. Do not make assumptions or infer additional information."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query_prompt)
            ]

            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error getting answer: {str(e)}"

@st.cache_resource
def create_qa_system():
    return DocumentQA()

# Main app

st.title("📚 Document QA System")
st.markdown("### Upload Excel files and ask questions about their content")

init_qa_system()

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
if uploaded_file:
    with st.spinner("Processing document..."):
        process_file_once(uploaded_file)
        if st.session_state.db_initialized:
            st.success("Document processed successfully!")

if st.session_state.get("db_initialized"):
    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Finding answer..."):
            answer = st.session_state.qa_system.get_answer(question)
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })

if st.session_state.chat_history:
    st.markdown("### Chat History")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        st.markdown("---")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This application allows you to:
    - Upload Excel documents
    - Ask questions about the content
    - Get AI-powered answers using AI21's models

    The system uses:
    - AI21 for text generation
    - Chroma for document storage
    - BGE Small for embeddings
    - Streamlit for the interface
    """)









