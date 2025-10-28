import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import io
from qdrant_client import QdrantClient, models
import uuid
from dotenv import load_dotenv
from pdf2image import convert_from_bytes  # For PDF to image conversion
from langchain_qdrant import QdrantVectorStore  # Using LangChain for Qdrant integration
from langchain_core.documents import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

load_dotenv()

# --- Initialize FastAPI and Qdrant Client ---
app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set as environment variables.")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# --- Model Loading (Using Transformers for Vision Encoder Extraction, Compatible with vLLM Model) ---
model_path = "deepseek-ai/deepseek-ocr"  # vLLM-compatible model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation='flash_attention_2',
    use_safetensors=True,
).eval().to(device)

# LangChain setup for text embedding (fallback if needed) and vector store
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # 384-dimensional vectors
QDRANT_COLLECTION_NAME = "document_vectors"
VECTOR_DIMENSION = 1024  # DeepSeek-OCR vision encoder output dimension

# Initialize LangChain QdrantVectorStore
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=embedder,  # Used for metadata, but we'll store custom vectors
)

# Ensure the collection exists
try:
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
    )
    print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created/recreated.")
except Exception as e:
    print(f"Could not create Qdrant collection: {e}")

# --- API Endpoint for PDF Processing and Visual Token Embedding Storage ---
@app.post("/process-pdf-and-store")
async def process_pdf_and_store(file: UploadFile = File(...)):
    """
    Receives a PDF, converts pages to images, extracts visual tokens (embeddings) using DeepSeek-OCR's vision encoder (vLLM-compatible), and stores them in Qdrant.
    Each page is stored as a separate vector.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()

    try:
        # Convert PDF to images (one per page)
        images = convert_from_bytes(pdf_bytes, dpi=150)  # Adjust DPI as needed for quality

        stored_pages = []
        for page_num, pil_image in enumerate(images, start=1):
            pil_image = pil_image.convert("RGB")

            # Prepare inputs for vision encoder
            prepare_inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}]}],
                images=[pil_image],
                return_tensors="pt"
            ).to(device)

            # Extract visual tokens using the vision encoder (compatible with vLLM model architecture)
            with torch.no_grad():
                visual_tokens = model.vision_encoder(prepare_inputs['pixel_values'].to(torch.bfloat16))[0]
            document_vector = torch.mean(visual_tokens, dim=1).squeeze().cpu().numpy().tolist()

            # Store in Qdrant (each page as a separate point)
            point_id = str(uuid.uuid4())
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=document_vector,
                        payload={
                            "filename": file.filename,
                            "page_number": page_num,
                            "total_pages": len(images)
                        }
                    )
                ],
                wait=True
            )

            stored_pages.append({
                "page_number": page_num,
                "point_id": point_id
            })

        return {
            "message": f"PDF processed and {len(images)} pages stored as visual token embeddings.",
            "filename": file.filename,
            "stored_pages": stored_pages
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "DeepSeek OCR PDF processing service (vLLM-compatible) is running. POST PDF to /process-pdf-and-store."}
