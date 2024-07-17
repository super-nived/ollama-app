import bs4
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Function to load, split, and retrieve documents
def load_and_retrieve_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain
def rag_chain(url, question):
    retriever = load_and_retrieve_docs(url)
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Flask endpoint for the RAG chain
@app.route('/rag_chain', methods=['POST'])
def rag_chain_endpoint():
    data = request.get_json()
    url = data.get('url')
    question = data.get('question')
    print(url,question,"*********")
    if not url or not question:
        return jsonify({'error': 'URL and question are required'}), 400
    
    try:
        answer = rag_chain(url, question)
        return jsonify({'message': answer}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
