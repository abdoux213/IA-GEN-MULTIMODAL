from pathlib import Path
import re
import config
from embedding import encoder_textes, encoder_image
from vectorstore import get_collection
from open_clip.tokenizer import SimpleTokenizer
from dataclasses import dataclass, field
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResultatIndexation:
    document_id: str
    nom_original: str
    fragments_texte: int = 0
    images: int = 0
    avertissements: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _tokenizer():
    return SimpleTokenizer()



def numero_page(chemin):                   # parse page_003 from the filename
    correspondance = re.search(r"page_(\d+)", Path(chemin).stem)
    return int(correspondance.group(1)) if correspondance else -1
          
def decouper_texte(texte, taille=config.CHUNK_TAILLE, chevauchement=config.CHUNK_CHEVAUCHEMENT):
    """Découper le texte en fragments courts adaptés à CLIP."""
    bpe = _tokenizer()
    if not 0 <= chevauchement < taille:
        raise ValueError("Le chevauchement doit être inférieur à la taille.")

    tokens = bpe.encode(texte)

    for debut in range(0, len(tokens), taille - chevauchement):
        fragment = bpe.decode(tokens[debut:debut + taille]).strip()

        if fragment:
            yield fragment

        if debut + taille >= len(tokens):
            break

def indexer_textes(collection, document_id, pdf_path, text_dir): 
    ids = []
    documents = []
    metadatas = []
    avertissements = []

    for fichier in sorted(text_dir.glob("page_*.txt")):
        texte = fichier.read_text(encoding="utf-8")
        page = numero_page(fichier)

        for index, fragment in enumerate(decouper_texte(texte)):
            ids.append(f"{document_id}:texte:{page}:{index}")
            documents.append(fragment)

            metadatas.append({
                "type": "texte",
                "document_id": document_id,
                "pdf": Path(pdf_path).name,
                "page": page,
                "fragment": index,
                "source": str(fichier.resolve())
            })


    for debut in range(0, len(documents), config.TAILLE_LOT):
        fin = debut + config.TAILLE_LOT
        lot = documents[debut:fin]

        collection.upsert(
            ids=ids[debut:fin],
            documents=lot,
            metadatas=metadatas[debut:fin],
            embeddings=encoder_textes(lot)
        )

    logger.info("%s : %d fragments de texte indexés", pdf_path, len(documents))
    return len(documents), avertissements


def indexer_images(collection, document_id, pdf_path, image_dir): 
    nombre_images_indexees = 0
    avertissements = []

    for fichier in sorted(image_dir.iterdir()):
        if not fichier.is_file() or fichier.suffix.lower() not in config.EXTENSIONS_IMAGES:
            continue

        try:
            vecteur = encoder_image(fichier)

            collection.upsert(
                ids=[f"{document_id}:image:{fichier.name}"],
                embeddings=[vecteur],
                documents=[f"Image extraite du PDF : {fichier.name}"],
                metadatas=[{
                    "type": "image",
                    "document_id": document_id,
                    "pdf": Path(pdf_path).name,
                    "page": numero_page(fichier),
                    "source": str(fichier.resolve())
                }]
            )

            nombre_images_indexees += 1

        except Exception as erreur:
            avertissements.append(f"Image ignorée : {fichier.name} — {erreur}")

    logger.info("%s : %d images indexées", pdf_path, nombre_images_indexees)
    return nombre_images_indexees, avertissements



def indexer_document(document_id, out_dir, nom_original):
    """Index one already-extracted document. Safe to call again (replaces old data)."""
    out_dir = Path(out_dir)
    text_dir, image_dir = out_dir / "textes", out_dir / "images"
    if not text_dir.is_dir():
        raise FileNotFoundError(f"Dossier de textes introuvable : {text_dir}")

    collection = get_collection()
    collection.delete(where={"document_id": document_id})   # once, for text AND images

    resultat = ResultatIndexation(document_id, nom_original)

    resultat.fragments_texte, av = indexer_textes(
        collection, document_id, nom_original, text_dir)
    resultat.avertissements += av

    if image_dir.is_dir():
        resultat.images, av = indexer_images(
            collection, document_id, nom_original, image_dir)
        resultat.avertissements += av

    return resultat

def supprimer_document(document_id):
    get_collection().delete(where={"document_id": document_id})