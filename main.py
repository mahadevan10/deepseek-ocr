import os
import threading
import uuid
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pdf2image import convert_from_bytes

load_dotenv()

app = FastAPI()

# Globals initialized lazily to let the server bind PORT fast
_model = None
_tokenizer = None
_qdrant = None
_init_lock = threading.Lock()

QDRANT_COLLECTION_NAME = "document_vectors"
VECTOR_DIMENSION = 1024  # expected DeepSeek-OCR vision encoder output
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_initialized():
    global _model, _tokenizer, _qdrant
    if _model is not None and _tokenizer is not None and _qdrant is not None:
        return
    with _init_lock:
        if _model is not None and _tokenizer is not None and _qdrant is not None:
            return

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set as environment variables.")

        # Qdrant client
        _qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=20.0)

        # Ensure collection exists
        try:
            _qdrant.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
            )
        except Exception as e:
            print(f"Qdrant collection setup notice: {e}")

        # Load tokenizer/model lazily
        model_path = "deepseek-ai/deepseek-ocr"
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        _model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            use_safetensors=True,
        ).eval().to(DEVICE)

        print(f"Initialized. Device={DEVICE}")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def read_root():
    return {"message": "DeepSeek OCR PDF processing service is running. POST PDF to /process-pdf-and-store."}


@app.post("/process-pdf-and-store")
async def process_pdf_and_store(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    _ensure_initialized()

    pdf_bytes = await file.read()
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)
        stored_pages = []

        for page_num, pil_image in enumerate(images, start=1):
            pil_image = pil_image.convert("RGB")

            inputs = _tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}]}],
                images=[pil_image],
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                visual_tokens = _model.vision_encoder(inputs["pixel_values"])[0]  # [B, T, D]
            document_vector = torch.mean(visual_tokens, dim=1).squeeze().cpu().numpy().tolist()

            point_id = str(uuid.uuid4())
            _qdrant.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=document_vector,
                        payload={
                            "filename": file.filename,
                            "page_number": page_num,
                            "total_pages": len(images),
                        },
                    )
                ],
                wait=True,
            )

            stored_pages.append({"page_number": page_num, "point_id": point_id})

        return {
            "message": f"PDF processed and {len(images)} pages stored as visual token embeddings.",
            "filename": file.filename,
            "stored_pages": stored_pages,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
