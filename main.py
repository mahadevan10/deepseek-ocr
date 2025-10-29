import os
import threading
import uuid
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer
import open_clip

load_dotenv()
app = FastAPI()

# Lazy singletons
_model = None
_tokenizer = None
_embedder = None
_clip_model = None
_clip_preprocess = None
_qdrant = None
_init_lock = threading.Lock()

QDRANT_COLLECTION_NAME = "document_vectors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_initialized():
    global _model, _tokenizer, _embedder, _clip_model, _clip_preprocess, _qdrant
    if all([_model, _tokenizer, _embedder, _clip_model, _clip_preprocess, _qdrant]):
        return
    with _init_lock:
        if all([_model, _tokenizer, _embedder, _clip_model, _clip_preprocess, _qdrant]):
            return

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set as environment variables.")

        # Qdrant
        _qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30.0)

        # Text embedder (384-dim)
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        text_dim = _embedder.get_sentence_embedding_dimension()

        # Image embedder (CLIP ViT-L/14, 768-dim)
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        _clip_model = _clip_model.to(DEVICE).eval()
        image_dim = 768

        # Create multivector collection if missing (do not wipe existing data)
        try:
            _qdrant.get_collection(QDRANT_COLLECTION_NAME)
        except Exception:
            _qdrant.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config={
                    "image": models.VectorParams(size=image_dim, distance=models.Distance.COSINE),
                    "text": models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
                },
            )

        # DeepSeek-OCR (for text extraction)
        model_id = "deepseek-ai/deepseek-ocr"
        _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
            _tokenizer.pad_token = _tokenizer.eos_token

        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        _model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="eager",
        ).to(dtype=dtype, device=DEVICE).eval()

        print(f"Initialized. Device={DEVICE}, dims: image={image_dim}, text={text_dim}")


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
        if not images:
            raise HTTPException(status_code=400, detail="No pages found in PDF.")

        stored_pages = []
        prompt = "<image>\nFree OCR."

        for page_num, pil_image in enumerate(images, start=1):
            pil_image = pil_image.convert("RGB")

            # OCR: DeepSeek expects a file path
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                pil_image.save(tmp_path, format="PNG")

            try:
                with torch.inference_mode():
                    res = _model.infer(
                        _tokenizer,
                        prompt=prompt,
                        image_file=tmp_path,
                        output_path="/tmp",
                        base_size=768,
                        image_size=512,
                        crop_mode=False,
                        save_results=False,
                        test_compress=False,
                    )
                # Extract text
                ocr_text = ""
                if isinstance(res, dict) and "text" in res:
                    ocr_text = res["text"] or ""
                else:
                    ocr_text = str(res) if res is not None else ""
                text_to_embed = ocr_text.strip()

                # Image embedding (CLIP)
                img_tensor = _clip_preprocess(pil_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    img_feats = _clip_model.encode_image(img_tensor)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                image_vec = img_feats.squeeze(0).detach().cpu().tolist()

                # Text embedding (optional if empty)
                text_vec = None
                if text_to_embed:
                    text_vec = _embedder.encode(text_to_embed, normalize_embeddings=True).tolist()

                # Upsert multimodal vectors
                vectors = {"image": image_vec}
                if text_vec is not None:
                    vectors["text"] = text_vec

                point_id = str(uuid.uuid4())
                _qdrant.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=vectors,
                            payload={
                                "filename": file.filename,
                                "page_number": page_num,
                                "total_pages": len(images),
                                "text": text_to_embed,
                            },
                        )
                    ],
                    wait=True,
                )
                stored_pages.append({"page_number": page_num, "point_id": point_id})
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return {
            "message": f"PDF processed. Stored {len(stored_pages)} page embeddings (image + text).",
            "filename": file.filename,
            "stored_pages": stored_pages,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
