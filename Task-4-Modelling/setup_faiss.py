## by @Prashant 
import faiss
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import pickle
TF_ENABLE_ONEDNN_OPTS=0


# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Process and save FAISS index
document_text = extract_text_from_pdf("data/current-medical-diagnosis-amp-treatment-2021-60nbsped-9781260469875-1260469875-9781260469868-1260469867_compress.pdf")
text_chunks = [document_text[i:i+512] for i in range(0, len(document_text), 512)]
embeddings = np.array([model.encode(chunk) for chunk in text_chunks], dtype=np.float32)

# Initialize FAISS index and store vectors
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index and text chunks
faiss.write_index(index, "faiss_index.idx")
with open("text_chunks.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

print("FAISS index saved successfully!")
