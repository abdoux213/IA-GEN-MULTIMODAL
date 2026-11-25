# config.py
import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv   # pip install python-dotenv

load_dotenv()                    # reads .env at the project root



# ---------------------------------------------------------------- Paths
ROOT = Path(__file__).parent
PDF_DIR = ROOT / "pdfs"
OUTPUT_DIR = ROOT / "resultats_pdf"
CHROMA_DIR = ROOT / "chroma_pdf"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
COLLECTION_NAME = "pdf_clip_vit_b32"

CHUNK_TAILLE = 60          # CLIP tokens
CHUNK_CHEVAUCHEMENT = 10
IMAGE_TAILLE_MIN = 100
TAILLE_LOT = 64
EXTENSIONS_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jp2"}
MAX_STEPS = 10
# ------------------------------------------------------------- Provider
# "local"  -> Ollama on your machine
# "cloud"  -> https://ollama.com with an API key
PROVIDER = "cloud" #os.getenv("LLM_PROVIDER", "local")

PROVIDERS = {
    "local": {
        "type": "ollama",
        "host": "http://localhost:11434",
        "model": "qwen3:1.7b",
    },
    "cloud": {
        "type": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "model": "nvidia/nemotron-3.5-super-vl:free",   # check openrouter.ai/models for exact IDs
    },
}

# ----------------------------------------------------------------- Models
MODELE_GENERATION =  PROVIDERS[PROVIDER]["model"]
API_KEY = PROVIDERS[PROVIDER].get("api_key", None)
BASE_URL = PROVIDERS[PROVIDER].get("base_url", None)

# ---- Retrieval
K = 5
K_IMAGES = 3

# ------------------------------------------------------------- Validation
def verifier():
    if PROVIDER not in PROVIDERS:
        raise ValueError(f"PROVIDER inconnu : {PROVIDER!r}")
    if PROVIDER == "cloud" and not API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY manquante dans .env")

verifier()