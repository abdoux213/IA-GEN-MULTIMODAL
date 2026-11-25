import hashlib
from io import BytesIO
from PIL import Image
from pathlib import Path
import base64
from IPython.display import display

def image_vers_data_url(chemin, taille_max=(1600, 1600)):
    """Préparer une image locale pour l'envoyer au modèle génératif."""
    chemin = Path(chemin)

    if not chemin.is_file():
        raise FileNotFoundError(f"Image introuvable : {chemin}")

    with Image.open(chemin) as image:
        image_rgb = image.convert("RGB")
        image_rgb.thumbnail(taille_max)

        tampon = BytesIO()
        image_rgb.save(tampon, format="PNG")

    image_base64 = base64.b64encode(
        tampon.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{image_base64}"


def afficher_resultats(resultats):
    for rang, (document, metadata, distance) in enumerate(
        zip(
            resultats["documents"][0],
            resultats["metadatas"][0],
            resultats["distances"][0]
        ),
        start=1
    ):
        print(
            f"\n{rang}. {metadata['type'].upper()} "
            f"| PDF : {metadata['pdf']} "
            f"| Page : {metadata['page']} "
            f"| Similarité cosinus : {1 - distance:.3f}"
        )

        if metadata["type"] == "image":
            chemin = Path(metadata["source"])

            if chemin.is_file():
                with Image.open(chemin) as image:
                    apercu = image.convert("RGB")
                    apercu.thumbnail((600, 450))
                    display(apercu)
            else:
                print("Fichier image introuvable :", chemin)
        else:
            print(document)

def calculer_document_id(pdf_path):          # sha256 of the bytes, [:24]
    document_id = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()[:24]
    return document_id