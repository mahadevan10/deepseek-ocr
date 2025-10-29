import os
import torch
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
from qdrant_client import QdrantClient, models
import uuid
import time

# --- Initialize FastAPI and Qdrant Client ---
app = FastAPI()

# IMPORTANT: Set these as environment variables in your Cloud Run service
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set.")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- Model Loading ---
model_path = "./models/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()

# --- Qdrant Collection Configuration ---
QDRANT_COLLECTION_NAME = "pdf_document_vectors"
VECTOR_DIMENSION = 4096  # This should be confirmed from the model's configuration

# Ensure the collection exists on startup
try:
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
    )
    print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created/recreated.")
except Exception as e:
    print(f"Could not create Qdrant collection: {e}")

# --- API Endpoint ---
@app.post("/process-pdf-and-store")
async def process_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF, converts each page to an image, extracts visual tokens,
    and stores them as a batch in Qdrant.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    start_time = time.time()
    pdf_bytes = await file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    points_to_upload = []
    doc_id = str(uuid.uuid4()) # A unique ID for the entire document

    try:
        # Process each page of the PDF
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Render page to a high-resolution image
            pix = page.get_pixmap(dpi=200)
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # 1. Prepare inputs for the model
            prepare_inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}]}],
                images=[pil_image],
                return_tensors="pt"
            ).to(device)

            # 2. Extract Visual Tokens from the DeepEncoder
            with torch.no_grad():
                visual_tokens = model.vision_encoder(prepare_inputs['pixel_values'].to(torch.bfloat16))[0]
            
            # Use the mean of the tokens as the vector for the page
            page_vector = torch.mean(visual_tokens, dim=1).squeeze().cpu().numpy().tolist()

            # 3. Prepare the point for batch upload
            points_to_upload.append(
                models.PointStruct(
                    id=str(uuid.uuid4()), # Unique ID for each page
                    vector=page_vector,
                    payload={
                        "filename": file.filename,
                        "document_id": doc_id, # Link pages to the same document
                        "page_number": page_num + 1
                    }
                )
            )

        # 4. Upload all points to Qdrant in a single batch operation
        if points_to_upload:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points_to_upload,
                wait=True
            )

        end_time = time.time()
        return {
            "message": "PDF processed and vectorized successfully.",
            "document_id": doc_id,
            "pages_processed": len(points_to_upload),
            "processing_time_seconds": round(end_time - start_time, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        pdf_document.close()


@app.get("/")
def read_root():
    return {"message": "PDF vectorization service is running. POST a PDF to /process-pdf-and-store."}

