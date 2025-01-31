import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ai21 import AI21Embeddings

def process_xlsx(file_path, batch_size=50):
    try:
        # Initialize AI21 embeddings
        api_key = os.getenv("AI21_API_KEY")
        if not api_key:
            raise ValueError("AI21_API_KEY environment variable is not set.")
            
        embeddings = AI21Embeddings(
            api_key=api_key,
            model="j2-large"  # or "j2-mid" or "j2-ultra" depending on your needs
        )
        
        # Define database path
        db_folder = 'db'
        os.makedirs(db_folder, exist_ok=True)
        
        # Load and process the Excel file
        print("Loading Excel file...")
        df = pd.read_excel(file_path)
        loader = DataFrameLoader(df, page_content_column=df.columns[0])
        documents = loader.load()
        
        # Split text into chunks
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create or update vector store
        print("Processing vector store...")
        if os.path.exists(os.path.join(db_folder, 'index')):
            vectordb = Chroma(persist_directory=db_folder, embedding_function=embeddings)
            # Add documents in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                vectordb.add_documents(documents=batch)
            print("Updated existing vector database.")
        else:
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_folder
            )
            print("Created new vector database.")
        
        vectordb.persist()
        return vectordb
        
    except Exception as e:
        print(f"Error in process_xlsx: {str(e)}")
        raise
