import torch
from PIL import Image
import open_clip
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model, _, preprocess = open_clip.create_model_and_transforms(
    config.CLIP_MODEL,
    pretrained=config.CLIP_PRETRAINED
)


@torch.inference_mode()
def encoder_textes(textes):
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
    tokens = tokenizer(textes).to(DEVICE)
    vecteurs = model.encode_text(tokens).float()

    vecteurs = vecteurs / vecteurs.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)

    return vecteurs.cpu().tolist()

@torch.inference_mode()
def encoder_image(chemin):
    with Image.open(chemin) as image:
        tenseur = preprocess(image.convert("RGB"))

    tenseur = tenseur.unsqueeze(0).to(DEVICE)
    vecteur = model.encode_image(tenseur).float()

    vecteur = vecteur / vecteur.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)

    return vecteur.cpu().tolist()[0]