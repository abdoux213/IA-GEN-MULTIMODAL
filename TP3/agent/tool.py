from langchain_core.tools import tool
import config
from pathlib import Path
from retrieval import recuperer_sources

@tool(response_format="content_and_artifact")
def retriever_tool(query: str, image_path: str | None = None) -> str:
    """Search the knowledge base (PDF text passages AND images/figures/diagrams).
    - query: a short, focused search query (no greetings or filler).
    - image_path: optional path to an image, if the user gave one and wants
      similar images or related passages.
    Returns text passages and image references with their PDF and page."""
    if image_path:
        _, imgs = recuperer_sources(query or "image", k=config.K_IMAGES, type_resultat="image", image_requete=image_path)
        _, txts = recuperer_sources(query or "image", k=config.K, type_resultat="texte", image_requete=image_path)
    else:
        _, txts = recuperer_sources(query, k=config.K, type_resultat="texte")
        _, imgs = recuperer_sources(query, k=config.K_IMAGES, type_resultat="image")

    imgs = [s for s in imgs if s["similarite"] >= 0.25]   # drop weak matches

    lines = [f"[TEXTE] {s['pdf']} p.{s['page']} (sim={s['similarite']:.2f})\n{s['contenu']}" for s in txts]
    lines += [f"[IMAGE {i}] {s['pdf']} p.{s['page']} (sim={s['similarite']:.2f}) {Path(s['source']).name}"
              for i, s in enumerate(imgs, 1)]

    content = "\n\n---\n\n".join(lines) or "No results."
    return content, [s["source"] for s in imgs] 