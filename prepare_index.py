import os
import json
import faiss
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_CSV = "data/dataset.csv"
INDICES_DIR = "indices"
os.makedirs(INDICES_DIR, exist_ok=True)


def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, processor


def embed_texts(model, processor, texts, batch_size=32):
    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            inputs = processor(
                text=batch,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(DEVICE)

            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_embs.append(text_features.cpu().numpy())

    return np.vstack(all_embs)


def embed_images(model, processor, image_paths, batch_size=32):
    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                images.append(img)

            inputs = processor(
                text=None,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)

            img_features = model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            all_embs.append(img_features.cpu().numpy())

    return np.vstack(all_embs)


def build_faiss_index(embeddings, metric="cosine"):
    d = embeddings.shape[1]
    if metric == "cosine":
        index = faiss.IndexFlatIP(d)   # inner product on normalized vectors = cosine
    else:
        index = faiss.IndexFlatL2(d)

    index.add(embeddings.astype("float32"))
    return index


def main():
    print(f"Using device: {DEVICE}")

    # 1. Load dataset
    df = pd.read_csv(DATA_CSV)

    # Adjust relative paths: "images/..." -> "data/images/..."
    df["image_path"] = df["image_path"].apply(
        lambda p: p if os.path.isabs(p) else os.path.join("data", p)
    )

    texts = df["text"].tolist()
    image_paths = df["image_path"].tolist()

    # 2. Load CLIP
    model, processor = load_clip_model()

    # 3. Compute embeddings
    text_embs = embed_texts(model, processor, texts)
    image_embs = embed_images(model, processor, image_paths)

    # 4. Build FAISS indices
    text_index = build_faiss_index(text_embs, metric="cosine")
    image_index = build_faiss_index(image_embs, metric="cosine")

    # 5. Save indices & embeddings & metadata
    faiss.write_index(text_index, os.path.join(INDICES_DIR, "text_index.faiss"))
    faiss.write_index(image_index, os.path.join(INDICES_DIR, "image_index.faiss"))

    np.save(os.path.join(INDICES_DIR, "text_embeddings.npy"), text_embs)
    np.save(os.path.join(INDICES_DIR, "image_embeddings.npy"), image_embs)

    metadata = {
        "texts": texts,
        "image_paths": image_paths,
    }
    with open(os.path.join(INDICES_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ Embeddings and FAISS indices saved in ./indices/")


if __name__ == "__main__":
    main()
