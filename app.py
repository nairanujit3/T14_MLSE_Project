import os
import json
import faiss
import torch
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor
import ollama  # pip install ollama


# =========================
# Configuration
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDICES_DIR = "indices"

# Must match the model used when you created embeddings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"   # or "clip_finetuned"
OLLAMA_MODEL_NAME = "gemma3:4b"                   # local Ollama model


# =========================
# Model & Index Loading
# =========================

@st.cache_resource
def load_clip_model(model_name: str = CLIP_MODEL_NAME):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, processor


@st.cache_resource
def load_indices_and_metadata():
    text_index = faiss.read_index(os.path.join(INDICES_DIR, "text_index.faiss"))
    image_index = faiss.read_index(os.path.join(INDICES_DIR, "image_index.faiss"))

    # embeddings not strictly needed at inference, but kept for completeness
    text_embeddings = np.load(os.path.join(INDICES_DIR, "text_embeddings.npy"))
    image_embeddings = np.load(os.path.join(INDICES_DIR, "image_embeddings.npy"))

    with open(os.path.join(INDICES_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return text_index, image_index, text_embeddings, image_embeddings, metadata


# =========================
# Embedding Helpers
# =========================

def embed_text(model, processor, text: str):
    with torch.no_grad():
        inputs = processor(
            text=[text],
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype("float32")


def embed_image(model, processor, image: Image.Image):
    with torch.no_grad():
        inputs = processor(
            text=None,
            images=[image.convert("RGB")],
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        img_features = model.get_image_features(**inputs)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features.cpu().numpy().astype("float32")


# =========================
# Ollama LLM Helper
# =========================

def generate_creative_paragraph_with_ollama(base_caption: str) -> str:
    """
    Use local Ollama (gemma3:4b) to turn the retrieved caption into
    a single creative paragraph description.
    """
    caption = " ".join(str(base_caption).split())
    if not caption:
        caption = "a scene that is not clearly described"

    prompt = (
        "You are a creative visual storyteller.\n\n"
        f"Rough caption: \"{caption}\"\n\n"
        "Based on this rough caption, write one vivid, natural-sounding paragraph "
        "that describes the image. Do not mention the caption or that you are an AI. "
        "Do not use bullet points or lists.\n\n"
        "Paragraph:\n"
    )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response["message"]["content"].strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return " ".join(lines)
    except Exception as e:
        return (
            "This image appears to show "
            f"{caption}, but an error occurred while generating a description: {e}"
        )


# =========================
# Streamlit App
# =========================

def main():
    st.set_page_config(page_title="CLIP + Ollama Multimodal RAG", layout="wide")

    st.title("🔎 CLIP-based Multimodal RAG System")
    st.write(
        "Multimodal retrieval using CLIP + FAISS, with creative text from a local Ollama LLM.\n\n"
        "- **Text → Image (Create Image)**: only shows the best-matching image.\n"
        "- **Image → Text (Generate Text)**: only shows the generated paragraph.\n"
        "Similarity scores and intermediate captions stay internal."
    )

    with st.spinner("Loading CLIP model and indices..."):
        model, processor = load_clip_model()
        text_index, image_index, _, _, metadata = load_indices_and_metadata()

    mode = st.sidebar.radio(
        "Select retrieval mode",
        ["Text → Image (Create Image)", "Image → Text (Generate Text)"]
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"Device: **{DEVICE.upper()}**")
    st.sidebar.write(f"CLIP model: {CLIP_MODEL_NAME}")
    st.sidebar.write(f"Ollama model: {OLLAMA_MODEL_NAME} (must be running locally)")

    # always use top-1
    k = 1

    # ---------- TEXT → IMAGE ----------
    if mode.startswith("Text → Image"):
        st.header("📝 Text → 🖼️ Create Image")

        query = st.text_input(
            "Enter your prompt",
            placeholder="e.g., 'a person walking with a dog on a grassy field'"
        )

        if st.button("Create Image"):
            if not query.strip():
                st.warning("Please enter a non-empty text prompt.")
            else:
                try:
                    query_emb = embed_text(model, processor, query)

                    scores, indices = image_index.search(query_emb, k)
                    best_idx = int(indices[0][0])

                    img_path = metadata["image_paths"][best_idx]

                    # Handle both 'data/images/...' and 'images/...'
                    if os.path.exists(img_path):
                        img_fs = img_path
                    else:
                        img_fs = os.path.join("data", img_path)

                    img = Image.open(img_fs).convert("RGB")

                    st.subheader("Generated Image")
                    st.image(img, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during retrieval: {e}")

    # ---------- IMAGE → TEXT ----------
    else:
        st.header("🖼️ Image → 📝 Generate Text")

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except UnidentifiedImageError:
                st.error("Invalid image file. Please upload a valid JPG/PNG.")
                image = None

            if image is not None:
                st.image(image, caption="Query Image", use_container_width=True)

                if st.button("Generate Text"):
                    try:
                        # 1) Retrieve best caption using CLIP + FAISS
                        query_emb = embed_image(model, processor, image)
                        scores, indices = text_index.search(query_emb, k)
                        best_idx = int(indices[0][0])
                        base_caption = metadata["texts"][best_idx]

                        # 2) Generate creative paragraph with Ollama
                        with st.spinner("Generating description with Ollama..."):
                            creative_text = generate_creative_paragraph_with_ollama(base_caption)

                        st.subheader("Generated Description")
                        st.write(creative_text)

                    except Exception as e:
                        st.error(f"Error during retrieval or generation: {e}")

    st.markdown("---")
    st.caption(
        "Only the primary outputs are shown: image for text prompts, text for image uploads. "
        "All retrieval scores and intermediate captions remain internal."
    )


if __name__ == "__main__":
    main()