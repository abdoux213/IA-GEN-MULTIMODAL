# extraction.py
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

import config

logger = logging.getLogger(__name__)


@dataclass
class ResultatExtraction:
    pdf: str
    text_dir: Path
    image_dir: Path
    nombre_pages: int = 0
    nombre_images: int = 0
    pages_sans_texte: list[int] = field(default_factory=list)
    avertissements: list[str] = field(default_factory=list)


def _extraire_images_page(document, page, numero, image_dir, avertissements):
    """Save the embedded images of one page. Returns how many were saved."""
    enregistrees = 0
    vues = set()

    for image_info in page.get_images(full=True):
        xref = image_info[0]
        if xref in vues:
            continue
        vues.add(xref)

        try:
            image = document.extract_image(xref)
            if not image:
                avertissements.append(f"Page {numero} : image {xref} non extractible.")
                continue

            if min(image["width"], image["height"]) < config.IMAGE_TAILLE_MIN:
                logger.debug("Page %d, xref %d : image trop petite, ignorée", numero, xref)
                continue

            fichier = image_dir / f"page_{numero:03d}_image_{xref}.{image['ext']}"
            fichier.write_bytes(image["image"])
            enregistrees += 1

        except Exception as erreur:
            avertissements.append(f"Page {numero} : image {xref} — {erreur}")
            logger.warning("Page %d, xref %d : %s", numero, xref, erreur)

    return enregistrees


def extraire_pdf(pdf_path, out_dir):
    """Extract text and images of one PDF into out_dir/textes and out_dir/images."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

    if out_dir.exists():
        shutil.rmtree(out_dir)            # no stale files from a previous extraction

    text_dir = out_dir / "textes"
    image_dir = out_dir / "images"
    text_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    resultat = ResultatExtraction(pdf=pdf_path.name, text_dir=text_dir, image_dir=image_dir)

    with pymupdf.open(pdf_path) as document:
        resultat.nombre_pages = len(document)

        for numero, page in enumerate(document, start=1):
            texte = page.get_text("text", sort=True)
            (text_dir / f"page_{numero:03d}.txt").write_text(texte, encoding="utf-8")

            if not texte.strip():
                resultat.pages_sans_texte.append(numero)

            resultat.nombre_images += _extraire_images_page(
                document, page, numero, image_dir, resultat.avertissements
            )

    if resultat.pages_sans_texte:
        resultat.avertissements.append(
            f"Pages sans texte détectable (scan ?) : {resultat.pages_sans_texte}"
        )

    logger.info("%s : %d pages, %d images", pdf_path.name,
                resultat.nombre_pages, resultat.nombre_images)
    return resultat