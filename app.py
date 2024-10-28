from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
import fitz 
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
import cohere
from langchain.chains import RetrievalQA
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here
        return redirect(url_for('plan'))
    return render_template('login.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle signup logic here
        return redirect(url_for('plan'))
    return render_template('signup.html')

# Plan selection route
@app.route('/plan', methods=['GET', 'POST'])
def plan():
    if request.method == 'POST':
        selected_plan = request.form.get('plan')
        if selected_plan:
            session['plan'] = selected_plan
            session['query_count'] = 0  # Reset query count on plan selection
            return redirect(url_for('upload'))
    return render_template('plan.html')

# Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['pdf']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['pdf_path'] = filepath
            session['query_count'] = 0  # Reset query count on file upload
            return redirect(url_for('chat'))
    return render_template('upload.html')

# Chat route
@app.route('/chat')
def chat():
    return render_template('chat.html')

# Route to handle chat queries
@app.route('/query_pdf', methods=['POST'])
def query_pdf():
    answer= 'nothing'
    data = request.get_json()
    query = data.get('query')
    
    # Check for uploaded PDF path in session
    pdf_path = session.get('pdf_path')
    if not pdf_path:
        return jsonify({'answer': 'No PDF uploaded. Please upload a PDF file.'})
    
    # Initialize query count in session
    if 'query_count' not in session:
        session['query_count'] = 0

    # Limit queries if free plan is selected
    if session.get('plan') == 'free' and session['query_count'] >= 5:
        return jsonify({'answer': 'Query limit reached for free plan. Upgrade to continue.'})

    # Extract text from the PDF
    documents = []
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
        # Wrap the extracted text in a Document object
        documents.append(Document(page_content=text))

    # Set up API keys
    os.environ['PINECONE_API_KEY'] = '8fdbabad-b7a9-40ed-833a-bad2d7055df8'
    
    # Initialize Cohere embeddings
    cohere_embeddings = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key="pGdW2vRcRYnh8PKvJoZ5Slm8j0p1ZNHHw8bYVQTo")
    
    # Use Cohere embeddings with Pinecone
    index_name = "pinecone-chatbot"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
    
    # Split the documents
    split_docs = text_splitter.split_documents(documents)
    
    vectorstore = PineconeVectorStore.from_documents(split_docs, cohere_embeddings, index_name=index_name)
    #similar_docs =vectorstore.similarity_search(query)
    cohere_client = cohere.Client('pGdW2vRcRYnh8PKvJoZ5Slm8j0p1ZNHHw8bYVQTo')
    
    if session.get('plan') == 'free':
        similar_docs = vectorstore.similarity_search(query, k=1)
        # If no documents are found, indicate that the query is irrelevant
        if len(similar_docs) == 0:
            print("The question is irrelevant to the content of the PDF.")
        else:
            doc_content = similar_docs[0].page_content
            doc_embedding = cohere_embeddings.embed_query(doc_content)
            query_embedding = cohere_embeddings.embed_query(query)
            similarity_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            relevance_threshold = 0.5
            
            if similarity_score < relevance_threshold:
                answer="The question is irrelevant to the content of the PDF."
            else:
                max_response_tokens = 100
                prompt = f"Based on the following document content, provide a concise answer to the question:\n\n{doc_content}\n\nQuestion: {query}\n\nAnswer (concise):"
                result = cohere_client.generate(
                    model='command-xlarge-nightly',
                    prompt=prompt,
                    max_tokens=max_response_tokens,
                    temperature=0.3 
                )
                answer= result.generations[0].text.strip()
    else:
        result = cohere_client.generate(
            model='command-xlarge-nightly',
            prompt=query,
            max_tokens=100,
            temperature=0.5
        )
        answer = result.generations[0].text

    session['query_count'] += 1
    return jsonify({'answer': answer})

# Start the Flask app
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
