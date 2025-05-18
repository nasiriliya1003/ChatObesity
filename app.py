import os
import pandas as pd
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load embedding model
embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Load and preprocess data
df = pd.read_csv('./AfrObesity_Data.csv')
context_data = [" ".join([f"{col}: {df.iloc[i][j]}" for j, col in enumerate(df.columns[:4])]) for i in range(len(df))]

# Get API key for LLM
groq_key = os.environ.get('groq_api_key')
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_key)

# Create and populate vector store
vectorstore = Chroma(collection_name="chatObesity_store", embedding_function=embed_model, persist_directory="./")
vectorstore.add_texts(context_data)
retriever = vectorstore.as_retriever()

# Define RAG prompt template
rag_prompt = PromptTemplate.from_template(
    """You are a medical expert.
    Use the provided context to answer the question accurately and concisely.
    If uncertain, say so. Do not discuss the context; provide a direct answer.
    Context: {context}
    Question: {question}
    Answer:"""
)

# Construct RAG chain
rag_chain = ( {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser() )

def rag_memory_stream(message, history):
    partial_text = ""
    for new_text in rag_chain.stream(message):
        partial_text += new_text
        yield partial_text

# Gradio UI setup
demo = gr.ChatInterface(
    fn=rag_memory_stream,
    type="messages",
    title="I am ChatObesity Bot, please try me :)",
    description="Your ultimate guide to better life...",
    fill_height=True,
    examples=[
        "What is Obesity?", 
        "Can you provide an overview of global obesity statistics, including trends and key contributing factors?",
        "30-year-old male, 1.80m, 100kg, drinks alcohol daily, eats fast food 5 times a week, no exercise."
    ],
    theme="glass",
)

if __name__ == "__main__":
    demo.launch(share=True)