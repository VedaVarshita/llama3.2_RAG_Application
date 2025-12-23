# LLaMA 3.2 RAG Research Assistant

A sophisticated retrieval-augmented generation (RAG) research assistant for semantic search and question answering over academic document collections. The system combines dense vector retrieval, local LLM inference, multiple reasoning workflows, and an interactive web interface to support efficient and transparent research workflows.

---

## 🔎 Overview

This project implements a production-grade RAG system that integrates:

* **Dense vector retrieval** using FAISS for semantic search
* **Local LLM inference** with LLaMA 3.2 via Ollama
* **Three operational modes**: Standard RAG, LangGraph workflow, and DSPy-optimized RAG
* **Interactive web interface** built with Gradio

The system is designed for researchers and practitioners who need to rapidly explore, query, and understand large collections of academic papers while maintaining full control over data and models.

---

## 🌟 Features

### Core Capabilities

* **Semantic Search**: Captures query intent beyond keyword matching
* **Multi-Document Support**: Index and query entire research paper collections
* **Three RAG Modes**:

  * **Standard RAG**: Direct retrieval and generation pipeline
  * **LangGraph**: Workflow-based RAG with document grading and conditional routing
  * **DSPy**: Optimized prompting and structured generation for improved quality
* **Interactive Web UI**: Gradio-based interface with real-time streaming responses
* **Source Attribution**: Automatic citation of retrieved documents
* **Performance Analytics**: Query statistics, latency tracking, and mode-level comparisons
* **Example Queries**: Predefined questions for rapid exploration

### Technical Highlights

* Maximal Marginal Relevance (MMR) for diverse and relevant retrieval
* Configurable chunk size and overlap for optimal indexing
* FAISS-based vector indexing for fast similarity search
* Streaming responses for improved UX
* Session-level statistics and performance tracking

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Document Processing                     │
├─────────────────────────────────────────────────────────────┤
│  PDF Loading → Text Splitting → Embedding → Vector Storage  │
│  (PyMuPDF)     (Recursive)      (Nomic)     (FAISS)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Query Processing                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐         │
│  │ Standard    │  │  LangGraph   │  │    DSPy     │         │
│  │    RAG      │  │   Workflow   │  │ Optimized   │         │
│  └─────────────┘  └──────────────┘  └─────────────┘         │
│         ↓                 ↓                  ↓              │
│         └─────────────────┴──────────────────┘              │
│                           ↓                                 │
│              MMR Retrieval (k=3, fetch_k=100)               │
│                           ↓                                 │
│              LLaMA 3.2 Generation (Ollama)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Gradio Web UI                          │
│  Chat Interface | Mode Selection | Statistics | Examples    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* Ollama server running locally
* LLaMA 3.2 model installed in Ollama
* PDF documents for indexing

### Installation

```bash
git clone https://github.com/VedaVarshita/llama3.2_RAG_Application.git
cd llama3.2_RAG_Application
pip install -r requirements.txt
```

### Ollama Setup

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama serve
```

### Environment Configuration

Create a `.env` file in the project root:

```env
EMBEDDING_MODEL=nomic-embed-text
CHAT_MODEL=llama3.2:1b
BASE_URL=http://localhost:11434
CHUNK_SIZE=1024
CHUNK_OVERLAP=128
```

### Prepare Documents

```bash
mkdir rag-data
# Add PDF files to rag-data/
```

### Run the Application

```bash
python main.py
```

The Gradio web interface will launch locally, allowing you to select RAG modes, ask questions, and inspect performance metrics.

---

## ⚙️ Configuration

### Environment Variables

| Variable          | Default                  | Description                       |
| ----------------- | ------------------------ | --------------------------------- |
| `EMBEDDING_MODEL` | `nomic-embed-text`       | Ollama embedding model name       |
| `CHAT_MODEL`      | `llama3.2:1b`            | Ollama chat model name            |
| `BASE_URL`        | `http://localhost:11434` | Ollama API endpoint               |
| `CHUNK_SIZE`      | `1024`                   | Document chunk size in characters |
| `CHUNK_OVERLAP`   | `128`                    | Overlap between chunks            |

These parameters control model selection, document preprocessing, and backend connectivity.

---

## 📊 Performance Metrics

### Overall Results

| Metric              | Standard RAG | LangGraph | DSPy  |
| ------------------- | ------------ | --------- | ----- |
| **Quality Score**   | 74%          | 63%       | 76%   |
| **Average Latency** | 1.79s        | 1.31s     | 1.06s |
| **Success Rate**    | 88%          | 60%       | 94%   |
| **MRR**             | 0.82         | 0.71      | 0.86  |

**Key Observations**:

* DSPy achieves the strongest overall performance
* LangGraph provides lower latency through stricter routing and filtering
* Standard RAG offers a stable and interpretable baseline

---

## 🧩 Key Components

* **Vector Store**: FAISS index with dense embeddings for fast similarity search
* **Retriever**: MMR-based retrieval balancing relevance and diversity
* **RAG Pipelines**:

  * LCEL-based Standard RAG
  * LangGraph conditional workflows
  * DSPy-optimized structured generation
* **UI Layer**: Gradio chat interface with mode selection and analytics

---

## 📚 Dependencies

* **LangChain**: RAG orchestration and LCEL pipelines
* **LangGraph**: Workflow-based reasoning and routing
* **DSPy**: Prompt and generation optimization
* **FAISS**: Vector similarity search
* **Ollama**: Local LLM inference
* **PyMuPDF**: PDF document processing
* **Gradio**: Interactive web interface

---

## 📈 Future Enhancements

* [ ] Advanced document grading strategies
* [ ] Multi-language document support
* [ ] Persistent vector stores and caching
* [ ] Fine-grained citation visualization
* [ ] Cloud-backed deployment options

---

**Built using LangChain, LangGraph, DSPy, FAISS, and LLaMA 3.2 via Ollama**
