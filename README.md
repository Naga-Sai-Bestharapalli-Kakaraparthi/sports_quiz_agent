# sports_quiz_agent
A production-grade, fault-tolerant RAG agent that orchestrates local vector stores (ChromaDB) and live web ingestion (DuckDuckGo) to generate deterministic, zero-hallucination sports quizzes using strict OpenAI Structured Outputs (Pydantic v2).
The **AI-Powered Sports Quiz Agent** is an autonomous, production-grade intelligence pipeline engineered to generate dynamically verified sports trivia evaluations. Built with a strict focus on factual grounding and high structural interface reliability, the agent completely eliminates LLM hallucinations and string-parsing fragility by utilizing advanced RAG (Retrieval-Augmented Generation) patterns.

### Key Architectural Highlights:
* 🗄️ **Factual Grounding Layer (Local RAG):** Implements an embedded, disk-persistent vector repository using **ChromaDB** with automated short-circuit loading logic to prevent redundant re-indexing and document duplication.
* 🌐 **Dynamic Sync Layer (Live Web Scraper):** Features a graceful, fault-tolerant background thread context manager using **DuckDuckGo Search API** to fetch real-time championship updates, tournament brackets, and live standings.
* 🛡️ **Zero-Fragility Interface (Structured Outputs):** Leverages **OpenAI's Structured Outputs API** constrained by **Pydantic v2 data models**. This forces the underlying model to map generation layers directly into rigid JSON schemas, eliminating index errors and parsing breaks.
* 📊 **State-Preserving UX:** Built with a clean **Streamlit** dashboard anchored completely in application session states, guaranteeing that user choices and evaluation tracking matrices persist smoothly across interface re-renders.
