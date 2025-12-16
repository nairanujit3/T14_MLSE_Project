# T14_MLSE_Project

Anujit Nair - 202418036

Vaibhav Agrawal - 202418059

# 🚀 Multimodal RAG System (CLIP + FAISS + Ollama)

This project implements a **fully local, free, open-source multimodal Retrieval-Augmented Generation (RAG) system** that supports:

### ✅ **Text → Image Retrieval**  
Enter a prompt → CLIP retrieves the most relevant image from your dataset.

### ✅ **Image → Text Generation**  
Upload an image → CLIP retrieves the closest caption → Ollama LLM (Gemma-3 4B) expands it into a **creative paragraph**.

### ⭐ 100% Offline  
- CLIP ViT-B/32 for image & text embeddings  
- FAISS for fast similarity search  
- Ollama (Gemma-3:4B) for creative text generation  
- Works on CPU/GPU

---

## 📁 Project Structure
```
├── data/
│ ├── images/ # Dataset images
│ └── dataset.csv # Image paths + captions (image_path, caption)
│
├── indices/
│ ├── image_index.faiss # FAISS index for image embeddings
│ ├── text_index.faiss # FAISS index for text embeddings
│ ├── image_embeddings.npy
│ ├── text_embeddings.npy
│ └── metadata.json # Stores image paths + captions
│
├── rag_env/ # Python virtual environment
│
├── app.py # Streamlit Web App
├── prepare_index.py # Builds FAISS indices (one-time)
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create & activate virtual environment
```
python -m venv rag_env
.\rag_env\Scripts\activate    # Windows
```

### 3.Install dependencies
```
pip install -r requirements.txt

```
### 4. Install & run Ollama
Download Ollama: https://ollama.com/download

Then pull the model:
```
ollama pull gemma3:4b
ollama serve
```

### 🖥️ Running the Streamlit App
```
streamlit run app.py
```

Features:
- Text → Image (Create Image)
  
    - Enter a prompt
    - CLIP retrieves the closest image
    - Only the final matched image is shown
    (no captions, no similarity scores)

- Image → Text (Generate Text)
  
    - Upload an image
    - CLIP retrieves the closest caption
    - Ollama turns it into a creative, story-like paragraph
    (no dataset images or captions displayed)

### 🧠 Architecture (How It Works)
      Text Query                           Image Input
          │                                     │
          ▼                                     ▼
      CLIP Text Embed                      CLIP Image Embed
          │                                     │
          ▼                                     ▼
      FAISS Search                         FAISS Search
          │                                     │
          ▼                                     ▼
  Best Matching Image                Best Matching Caption
          │                                     │
          ▼                                     ▼
     Final Output                    Ollama LLM (Gemma3 4B)
                                               │
                                               ▼
                             Creative Image Description Paragraph

### 🚀 Future Improvements

- Fine-tuned CLIP model for higher accuracy
- Add stable diffusion or SDXL for generating images
- Multi-image retrieval
- FastAPI backend + React/Flutter front-end
- GPU-accelerated inference


<video src="[https://github.com/user-attachments/assets/ea6a4f6b-8e4f-4022-8bfa-2e0f9b2e701e]" controls title="Video Title"></video>



