import threading
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, vectorstore):
    conversation_chain = get_conversation_chain(vectorstore)
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']
    bot_response = chat_history[-1].content if chat_history else ""
    return bot_response

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    pdf_files = request.files.getlist('pdf_docs')  # Assuming PDF files are sent as multipart/form-data
    text_chunks = get_text_chunks(get_pdf_text(pdf_files))
    vectorstore = get_vectorstore(text_chunks)
    bot_response = handle_userinput(user_input, vectorstore)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
