from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import requests
import os
import tempfile
import re
import torch
import gc
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Environment variables (set in Vercel dashboard)
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
AUTH_TOKEN = "4ce8b39610862044f1ef304e9db3b75458b88525c8d21182bd27136a7f956e454"

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class AnswerResponse(BaseModel):
    answers: list[str]

# Initialize models at startup
@app.on_event("startup")
async def load_models():
    global embedding_model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Clear GPU memory if previously allocated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Models loaded successfully")

# --- Helper Functions from your Colab ---
def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def chunk_text(text, max_words=400):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) <= max_words:
            current_chunk.append(sentence)
            word_count += len(words)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            word_count = len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def query_llama(prompt, max_tokens=500, temperature=0.1):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<|eot_id|>", "<|end_of_text|>"],
        "top_p": 0.7
    }
    try:
        response = requests.post("https://api.together.xyz/v1/completions", 
                                headers=headers, 
                                json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['text'].strip()
    except Exception as e:
        print(f"API Error: {str(e)}")
        return f"Error processing request: {str(e)}"

def process_question(question, chunks, embeddings):
    query_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_indices = similarities.topk(min(5, len(similarities))).indices
    
    context = "\n".join([chunks[i] for i in top_indices])
    
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an insurance policy expert. Answer the question concisely based ONLY on the provided context.
Return ONLY the factual answer without additional explanations.
If the information is not in the context, say "Not specified in the document".

Context:
{context}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return query_llama(prompt)

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_query(
    request: QueryRequest,
    authorization: str = Header(None)
):
    # Authentication
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    answers = []
    
    try:
        # Download and process PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
            download_pdf(request.documents, tmp_file.name)
            text = extract_text_from_pdf(tmp_file.name)
            chunks = chunk_text(text)
            
            # Generate embeddings
            embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
            
            # Process each question
            for question in request.questions:
                answer = process_question(question, chunks, embeddings)
                answers.append(answer)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return {"answers": answers}

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "active", "message": "Query processing system online"}