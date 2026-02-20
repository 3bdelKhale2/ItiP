import os
from typing import List, Dict, Any

# Switch to langchain-chroma to address deprecation warnings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Fix imports for LongContextReorder across LangChain versions
try:
    from langchain_community.document_transformers import LongContextReorder
except Exception:
    try:
        from langchain.retrievers.document_compressors import LongContextReorder  # type: ignore
    except Exception:
        # Final fallback; will raise at runtime if used
        LongContextReorder = None  # type: ignore

# Make NVIDIA endpoint imports resilient so the file can be analyzed without the package
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
except Exception:
    NVIDIAEmbeddings = None  # type: ignore
    ChatNVIDIA = None  # type: ignore

# Disable Chroma telemetry to avoid capture() errors
os.environ.setdefault("CHROMADB_TELEMETRY", "False")

VECTORSTORE_DIR = "./vectorstore"


def get_embeddings(model: str = "nvidia/nv-embed-v1"):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Please set it to use NVIDIA endpoints.\n"
            "Get a key at https://build.nvidia.com and set NVIDIA_API_KEY environment variable."
        )
    if NVIDIAEmbeddings is None:
        raise RuntimeError("langchain-nvidia-ai-endpoints is not installed. Please install it from requirements.txt.")
    return NVIDIAEmbeddings(model=model, truncate="END")


def get_llm(model: str = "meta/llama-3.1-8b-instruct"):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Please set it to use NVIDIA endpoints.\n"
            "Get a key at https://build.nvidia.com and set NVIDIA_API_KEY environment variable."
        )
    if ChatNVIDIA is None:
        raise RuntimeError("langchain-nvidia-ai-endpoints is not installed. Please install it from requirements.txt.")
    return ChatNVIDIA(model=model)


def build_vectorstore(chunks: List[Dict[str, Any]]):
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = get_embeddings()
    vs = Chroma(collection_name="contracts",
                embedding_function=embeddings,
                persist_directory=VECTORSTORE_DIR)
    # Add in batches
    if texts:
        vs.add_texts(texts=texts, metadatas=metadatas)
        # Since Chroma 0.4.x, manual persist is not needed
    return vs


def get_retriever() -> Any:
    embeddings = get_embeddings()
    vs = Chroma(collection_name="contracts",
                embedding_function=embeddings,
                persist_directory=VECTORSTORE_DIR)
    # Increase k for broader context during summarization/QA
    return vs.as_retriever(search_kwargs={"k": 20})


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs])


def _citations_from_docs(docs: List[Document]) -> str:
    cites = []
    for d in docs:
        m = d.metadata or {}
        source = m.get("source", "unknown")
        page = m.get("page")
        chunk_id = m.get("chunk_id", "chunk_?")
        if page is not None:
            cites.append(f"[{source} p.{page} {chunk_id}]")
        else:
            cites.append(f"[{source} {chunk_id}]")
    # dedupe
    uniq = []
    seen = set()
    for c in cites:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return " ".join(uniq)


def build_qa_chain():
    retriever = get_retriever()
    if LongContextReorder is None:
        raise RuntimeError("LongContextReorder not available. Update langchain or install langchain-community per requirements.txt.")
    reordering = LongContextReorder()  # DocumentTransformer
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart-contract assistant. Answer only using the retrieved context.\n"
                   "If the answer is not in the context, say you don't know.\n"
                   "Always include citations in the format [source.pdf p.X chunk_Y]."),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations: {citations}")
    ])

    def prepare_docs(inputs: Dict[str, Any]):
        # Support both raw string input and dict with question
        q = inputs if isinstance(inputs, str) else inputs.get("question", "")
        if not q:
            return []
        docs = retriever.invoke(q)  # List[Document]
        return reordering.transform_documents(docs)

    chain = (
        # Normalize input to a dict containing question
        RunnablePassthrough.assign(question=lambda x: x if isinstance(x, str) else x.get("question", ""))
        | {"docs": prepare_docs, "question": lambda x: x["question"]}
        | {
            "context": lambda x: _format_docs(x.get("docs", [])),
            "question": lambda x: x["question"],
            "citations": lambda x: _citations_from_docs(x.get("docs", [])),
        }
        | prompt
        | llm
    )

    return chain


def build_summary_chain():
    retriever = get_retriever()
    if LongContextReorder is None:
        raise RuntimeError("LongContextReorder not available. Update langchain or install langchain-community per requirements.txt.")
    reordering = LongContextReorder()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the uploaded smart contract documents using only the provided context.\n"
                   "Provide a structured summary with sections: purpose, key clauses, risks, missing info, definitions.\n"
                   "Include citations after each point. If information isn't present, say 'I don't know'."),
        ("human", "Context:\n{context}\n\nCitations: {citations}")
    ])

    def gather_all_docs(_: Dict[str, Any]):
        # Fetch top documents using a broad query keyword
        docs = retriever.invoke("contract")
        return reordering.transform_documents(docs)

    def grounded_or_idk(payload: Dict[str, Any]):
        ctx = payload.get("context", "").strip()
        cits = payload.get("citations", "").strip()
        if not ctx or not cits:
            return {"context": "", "citations": "", "question": "Summarize the documents."}
        return payload

    chain = (
        RunnablePassthrough.assign(docs=gather_all_docs)
        | {
            "context": lambda x: _format_docs(x["docs"]) if x.get("docs") else "",
            "citations": lambda x: _citations_from_docs(x["docs"]) if x.get("docs") else "",
        }
        | grounded_or_idk
        | prompt
        | llm
    )
    return chain
