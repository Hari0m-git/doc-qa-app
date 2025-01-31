from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_ai21 import AI21Embeddings
from langchain.prompts import PromptTemplate
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import os

# Define improved prompt template
PROMPT_TEMPLATE = """
You are an AI assistant specialized in analyzing and explaining document content. Your task is to answer the user's question based on the following context:

{context}

Question: {question}

Please provide a detailed yet concise answer by following these guidelines:
1. Focus on relevant details directly related to the question
2. Use clear and simple language
3. Structure the response in an easy-to-read format
4. Be truthful and mention if the context doesn't fully answer the question

Keep the response accurate and easy to understand.
"""

class QAChain:
    def __init__(self):
        self.api_key = os.getenv("AI21_API_KEY")
        if not self.api_key:
            raise ValueError("AI21_API_KEY environment variable is not set.")
        
        self.client = AI21Client(api_key=self.api_key)
        self.embeddings = AI21Embeddings(
            api_key=self.api_key,
            model="j2-large"  # or "j2-mid" or "j2-ultra"
        )
        self.vectordb = Chroma(
            persist_directory="db",
            embedding_function=self.embeddings
        )
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def get_relevant_context(self, question, k=3):
        results = self.vectordb.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in results])
        sources = "\n".join([f"Source: {str(doc.metadata)}" for doc in results])
        return context, sources

    def answer_question(self, question):
        context, sources = self.get_relevant_context(question)
        full_prompt = self.prompt.format(context=context, question=question)
        
        messages = [
            ChatMessage(
                role="system", 
                content="You are an AI assistant specialized in analyzing documents. Answer questions as best as you can based only on the provided context."
            ),
            ChatMessage(role="user", content=full_prompt)
        ]
        
        response = self.client.chat.completions.create(
            model="jamba-1.5-mini",
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
            timeout=30
        )
        
        final_response = response.choices[0].message.content + "\n\n" + sources
        return final_response

def get_qa_chain():
    return QAChain()
