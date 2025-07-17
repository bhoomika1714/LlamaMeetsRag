

##  LLaMA Meets RAG: Concept-Aware Summarization of Academic Papers

###  Abstract

This project presents a **concept-aware summarization system** that integrates **Retrieval-Augmented Generation (RAG)** with **LoRA-optimized LLaMA (1B)** and **semantic search (FAISS)** to automatically summarize scientific literature, especially from [arXiv.org](https://arxiv.org). It features a transformer-based concept extractor and a user-friendly Gradio interface to enable quick access to domain-specific summaries.

---

###  Features

* 🔍 Title-based scientific paper retrieval using the arXiv API
* 🧩 Key concept extraction via transformer-based scoring
* 📚 Semantic search using **FAISS**
* 🦙 Summary generation via **LoRA-optimized LLaMA 3.2–1B**
* 🎛️ Real-time interaction through a **Gradio web interface**
* 🧪 Evaluation using **ROUGE**, **BERTScore**, and content preservation

---

### 🛠️ System Architecture

```
flowchart TD
    A[User Query] --> B[arXiv Paper Retrieval]
    B --> C[Text Chunking]
    C --> D[Embedding with SentenceTransformers]
    D --> E[Key Concept Selection via Transformer + Attention]
    E --> F[Summarization with LLaMA (LoRA)]
    F --> G[FAISS Storage and Search]
    G --> H[Display in Gradio UI]
```
<img width="547" height="347" alt="image" src="https://github.com/user-attachments/assets/2a83164b-60e4-49f1-aff0-5231f1c4f16e" />

---

###  Tech Stack

| Component    | Tool                                                                                                      |
| ------------ | --------------------------------------------------------------------------------------------------------- |
| LLM          | [LLaMA 3.2–1B](https://github.com/meta-llama/llama) with [LoRA](https://arxiv.org/abs/2308.03303)         |
| Retrieval    | [arXiv API](https://arxiv.org/help/api/index)                                                             |
| Embedding    | [SentenceTransformers: all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| Search       | [FAISS](https://github.com/facebookresearch/faiss)                                                        |
| Tokenization | NLTK                                                                                                      |
| Interface    | [Gradio](https://gradio.app/)                                                                             |
| Evaluation   | ROUGE, BERTScore, Content Preservation %                                                                  |

---

### Results

| Metric               | Baseline | LoRA Fine-Tuned |
| -------------------- | -------- | --------------- |
| ROUGE-1              | 0.562    | 0.739           |
| ROUGE-2              | 0.558    | 0.736           |
| ROUGE-L              | 0.562    | 0.739           |
| BERTScore F1         | 0.909    | 0.955           |
| Content Preservation | 38.5%    | 57.7%           |

*  **FAISS Latency**: 0.12ms
*  **Avg. arXiv API Response**: <1s
*  Dataset: 2.4M papers, 680K full-text papers indexed

---

###  Web UI Snapshot

> The Gradio UI allows you to:
>
> * Input queries
> * Choose number of papers
> * View responsive cards with title, authors, summary, key concepts, and PDF link

<img width="679" height="558" alt="image" src="https://github.com/user-attachments/assets/da2825ea-9b38-42a0-8aad-9ee033df0cc8" />

---

### 📁 Project Structure

```
├── app/
│   ├── main.py           # Gradio frontend and pipeline runner
│   ├── rag_pipeline.py   # Full RAG logic
│   ├── summarizer.py     # LoRA + LLaMA summarization module
│   └── utils.py          # Tokenization, arXiv API, chunking
├── models/
│   └── llama-lora/       # LoRA fine-tuned weights
├── data/
│   └── index.faiss       # FAISS index
├── requirements.txt
├── README.md
└── LICENSE
```

---

### 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/llama-meets-rag.git
cd llama-meets-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app/main.py
```

---

### ✅ How It Works (Simplified)

1. User enters query → system fetches arXiv papers.
2. Each paper is chunked and embedded.
3. Transformer ranks key chunks (concepts).
4. LoRA+LLaMA generates structured summaries.
5. FAISS indexes results → enables similarity search.
6. Gradio UI displays output.

---

### 📈 Evaluation Metrics

* ROUGE-1, 2, L: For n-gram overlap with reference
* BERTScore: For semantic alignment
* Content Preservation: Human score for factual correctness
* Inference Time & Memory: Compared for LoRA vs. full fine-tuning

---

### 📌 Future Work

* 📖 Full-text integration with arXiv PDFs
* 🌐 Multilingual summarization support
* 🔍 Enhanced semantic search with hybrid models
* 🤖 Integration with academic recommendation engines

---

### 👩‍💻 Contributors

* Shribhakti S Vibhuti
* Snehal V Devasthale
* Shruti Sutar
* **Bhoomika Marigoudar**
* Saakshi Lokhande
* Dr. Uday Kulkarni *(Supervisor)*

---

