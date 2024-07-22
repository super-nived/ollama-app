
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
    print('load and retrive docs is starting...')
    loader = WebBaseLoader(url)
    docs = loader.load()
    print("this is loaded data after webscrap",docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("text splitter is done")
    splits = text_splitter.split_documents(docs)
    print("splits")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("embbedings")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("now we are going to return the vector store ok")
    return vectorstore.as_retriever()

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain
def rag_chain(url, question):
    print("i am inside the rag_chain function")
    retriever = load_and_retrieve_docs(url)
    # print(retrieved_docs,"this is after retriever doc function")
    retrieved_docs = retriever.invoke(question)
    print(retrieved_docs,"this is retrived docs")
    formatted_context = format_docs(retrieved_docs)
    print(formatted_context,'this formated docs')
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response

# Flask endpoint for the RAG chain
@app.route('/rag_chain', methods=['POST'])
def rag_chain_endpoint():
    data = request.get_json()
    url = data.get('url')
    print(type(url),'this is type of urls')
    question = data.get('question')
    print(url,question,"*********")
    if not url or not question:
        return jsonify({'error': 'URL and question are required'}), 400  
    try:
        print("entering in to rag chain function")
        answer = rag_chain(url, question)
        return jsonify({'message': answer}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
