from vectorstore import get_collection
from embedding import encoder_textes, encoder_image



def rechercher(texte=None, image=None, type_resultat=None, k=6):
    """
    Fournir soit un texte, soit un chemin d'image.
    type_resultat : None, "texte" ou "image".
    """
    
    if (texte is None) == (image is None):
        raise ValueError("Fournissez soit un texte, soit une image.")

    if type_resultat not in (None, "texte", "image"):
        raise ValueError("type_resultat doit être 'texte', 'image' ou None.")

    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")
    collection = get_collection()
    total = collection.count()
    if total == 0:
        raise ValueError("La collection est vide.")

    vecteur = (
        encoder_textes([texte])[0]
        if texte is not None
        else encoder_image(image)
    )

    options = {}
    if type_resultat is not None:
        options["where"] = {"type": type_resultat}

    return collection.query(
        query_embeddings=[vecteur],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
        **options
    )


def recuperer_sources(question, k=6, type_resultat=None, image_requete=None):
    """
    Recherche les sources pertinentes.

    Par défaut : recherche à partir de la question.
    Si image_requete est fourni : recherche à partir de cette image.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("La question doit être un texte non vide.")

    if image_requete is None:
        resultats = rechercher(
            texte=question,
            type_resultat=type_resultat,
            k=k
        )
    else:
        resultats = rechercher(
            image=image_requete,
            type_resultat=type_resultat,
            k=k
        )

    sources = []

    for index, (identifiant, document, metadata, distance) in enumerate(
        zip(
            resultats["ids"][0],
            resultats["documents"][0],
            resultats["metadatas"][0],
            resultats["distances"][0]
        ),
        start=1
    ):
        sources.append({
            "reference": f"S{index}",
            "id": identifiant,
            "type": metadata["type"],
            "pdf": metadata["pdf"],
            "page": metadata["page"],
            "source": metadata["source"],
            "contenu": document,
            "similarite": 1 - distance
        })

    return resultats, sources