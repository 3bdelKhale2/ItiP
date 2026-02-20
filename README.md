# Smart Contract Assistant (RAG)

A simple RAG web app to upload smart contract documents (PDF/DOCX/TXT), index them into a persistent Chroma vector store, and then:
- Ask grounded questions with citations
- Summarize all documents with structured sections and citations

Tech Stack:
- Python 3.11
- LangChain (runnables)
- Chroma (persistent at ./vectorstore)
- NVIDIAEmbeddings (nvidia/nv-embed-v1) with truncate="END"
- ChatNVIDIA (meta/llama-3.1-8b-instruct)
- Gradio UI

Requirements:
- NVIDIA_API_KEY environment variable set (get one at https://build.nvidia.com)

Setup
1) Create a Python 3.11 environment.
2) Install dependencies:
   pip install -r requirements.txt
3) Set NVIDIA API key:
   On Windows PowerShell:
   $Env:NVIDIA_API_KEY="your_key_here"
   On bash:
   export NVIDIA_API_KEY="your_key_here"

Run
- Start UI:
  python app.py
- Open http://localhost:7860

Usage
1) Upload contract files (.pdf, .docx, .txt).
2) Click "Index Documents" to parse, chunk, and embed into Chroma at ./vectorstore.
3) Ask questions. Answers will be restricted to retrieved context and include citations like [source.pdf p.3 chunk_12]. If unrelated or not found, you'll get a guardrail response.
4) Click "Summarize all documents" for a structured summary with citations.

Notes
- Uploads are stored under ./uploads/ with unique names.
- Chunking uses ~800-1200 chars with 200 overlap; metadata includes source filename, page (if known), and chunk_id.
- Retrieval uses k=4 with LongContextReorder.
- Streaming tokens are shown in the Gradio outputs.
- If NVIDIA_API_KEY is missing, the app shows a clear error message with instructions.

Evaluation
- Run simple automated evaluation:
  python evaluation.py
- Prints:
  - % answered with citations
  - % "I don't know" responses
  - average latency (rough)

Troubleshooting
- If PDFs fail to parse, ensure PyPDF2 is installed (it is in requirements). Some PDFs with scanned images may produce empty text.
- If DOCX fails, ensure python-docx installed.
- If Chroma complains about versions, try upgrading chromadb.

License
- For demo purposes.
