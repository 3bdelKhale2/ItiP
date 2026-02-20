import os
import gradio as gr
from typing import List, Tuple
from pathlib import Path

# Inject NVIDIA API key for local run (prefer environment variables in production)
os.environ["NVIDIA_API_KEY"] = "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"

from utils import ensure_dirs, make_unique_path, guard_unrelated
from ingestion import ingest, SUPPORTED_EXTENSIONS
from rag_pipeline import build_vectorstore, build_qa_chain, build_summary_chain


def _get_file_info(f: gr.File) -> Tuple[str | None, str | None]:
    """Return (temp_path, original_name) for a Gradio File across versions."""
    # Gradio 4: UploadedFile with .path and .orig_name
    temp_path = getattr(f, "path", None)
    orig_name = getattr(f, "orig_name", None)
    # Older Gradio: may provide dict-like or str
    if temp_path is None:
        temp_path = getattr(f, "name", None)  # some versions expose .name
    if isinstance(f, (str, Path)):
        temp_path = str(f)
        orig_name = os.path.basename(temp_path)
    if temp_path and not orig_name:
        orig_name = os.path.basename(temp_path)
    return temp_path, orig_name


def save_uploads(files: List[gr.File]) -> List[str]:
    ensure_dirs()
    saved_paths = []
    for f in files:
        if not f:
            continue
        temp_path, orig_name = _get_file_info(f)
        if not temp_path or not orig_name:
            continue
        ext = os.path.splitext(orig_name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        dest = make_unique_path("./uploads", orig_name)
        with open(dest, "wb") as out, open(temp_path, "rb") as inp:
            out.write(inp.read())
        saved_paths.append(dest)
    return saved_paths


def index_files(uploaded_files: List[gr.File]):
    paths = save_uploads(uploaded_files)
    if not paths:
        return "No supported files uploaded.", None
    try:
        chunks = ingest(paths)
        vs = build_vectorstore(chunks)
        return f"Indexed {len(chunks)} chunks from {len(paths)} files.", vs
    except RuntimeError as e:
        return str(e), None
    except Exception as e:
        return f"Indexing error: {e}", None


qa_chain = None
summary_chain = None


def ensure_chains():
    global qa_chain, summary_chain
    if qa_chain is None:
        qa_chain = build_qa_chain()
    if summary_chain is None:
        summary_chain = build_summary_chain()


def stream_answer(question: str, history: list[dict]):
    """Stream answer returning list of {role, content} messages for Gradio v4, and keep state in sync."""
    history = history or []
    history = history + [{"role": "user", "content": question}]
    
    # Handle greetings and basic conversation
    q_lower = question.lower().strip()
    if any(greeting in q_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
        response = "Hello! I'm your Smart Contract Assistant. I can help you analyze uploaded contract documents. You can:\n\n• Upload documents in the sidebar\n• Ask questions about contract content\n• Generate summaries of your documents\n\nHow can I assist you today?"
        history = history + [{"role": "assistant", "content": response}]
        yield history, history
        return
    
    if any(phrase in q_lower for phrase in ['how are you']):
        response = "I'm doing well, thank you! I'm ready to help you analyze your contract documents. Have you uploaded any documents yet?"
        history = history + [{"role": "assistant", "content": response}]
        yield history, history
        return
    
    if any(phrase in q_lower for phrase in ['what can you do', 'what are you', 'who are you', 'help']):
        response = "I'm a Smart Contract Assistant powered by RAG (Retrieval Augmented Generation). I can:\n\n• **Analyze contract documents** - Upload PDF, DOCX, or TXT files\n• **Answer questions** about contract content with citations\n• **Generate summaries** of uploaded documents\n• **Provide insights** based on your specific contract data\n\nTo get started, upload your documents using the sidebar, then ask me anything about them!"
        history = history + [{"role": "assistant", "content": response}]
        yield history, history
        return
    
    if any(phrase in q_lower for phrase in ['thanks', 'thank you']):
        response = "You're welcome! Is there anything else I can help you with regarding your contract documents?"
        history = history + [{"role": "assistant", "content": response}]
        yield history, history
        return
    
    # Check if the question is off-topic
    if guard_unrelated(question):
        history = history + [{"role": "assistant", "content": "I'm focused on helping with contract document analysis. Please upload your documents and ask questions related to their content, or I can help generate summaries of your uploaded materials."}]
        yield history, history
        return
    
    # Handle document-related queries
    try:
        ensure_chains()
    except RuntimeError as e:
        history = history + [{"role": "assistant", "content": str(e)}]
        yield history, history
        return
    if qa_chain is None:
        history = history + [{"role": "assistant", "content": "I don't have any documents indexed yet. Please upload and index documents first using the sidebar, then I can answer questions about them."}]
        yield history, history
        return

    history = history + [{"role": "assistant", "content": ""}]
    yield history, history

    try:
        for chunk in qa_chain.stream({"question": question}):
            content = getattr(chunk, "content", chunk)
            part = "\n".join(str(x) for x in content) if isinstance(content, list) else str(content)
            history[-1]["content"] += part
            yield history, history
    except Exception as e:
        history[-1]["content"] = f"Answer error: {e}"
        yield history, history


def stream_summary():
    try:
        ensure_chains()
    except RuntimeError as e:
        yield str(e)
        return
    if summary_chain is None:
        yield "Summary chain is not available. Please index documents first."
        return
    # Provide a visible starting message for UX
    yield "Summarizing documents..."
    try:
        for chunk in summary_chain.stream({}):
            content = getattr(chunk, "content", chunk)
            if isinstance(content, list):
                yield "\n".join(str(x) for x in content)
            else:
                yield str(content)
        # Provide a small completion notice so the last chunk doesn't vanish
        yield "\n\nDone."
    except Exception as e:
        # Show any runtime error instead of failing silently
        yield f"Summary error: {e}"


def summarize_sync():
    try:
        ensure_chains()
    except RuntimeError as e:
        return str(e)
    if summary_chain is None:
        return "Summary chain is not available. Please index documents first."
    buf = ["Summarizing documents...\n"]
    try:
        for chunk in summary_chain.stream({}):
            content = getattr(chunk, "content", chunk)
            if isinstance(content, list):
                buf.append("\n".join(str(x) for x in content))
            else:
                buf.append(str(content))
        buf.append("\n\nDone.")
    except Exception as e:
        buf.append(f"\n\nSummary error: {e}")
    return "".join(buf)


def build_ui():
    # ChatGPT-style dark theme CSS
    custom_css = """
    /* Global styles matching ChatGPT */
    .gradio-container { 
        background: #212121 !important; 
        font-family: 'Söhne', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        color: #ececf1 !important;
    }
    
    /* Main layout */
    .app-container { 
        height: 100vh; 
        background: #212121;
        display: flex;
    }
    
    /* Left sidebar - ChatGPT style */
    .sidebar { 
        background: #171717;
        border-right: 1px solid #2f2f2f; 
        padding: 0;
        width: 260px;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }
    
    /* Sidebar header */
    .sidebar-header {
        padding: 16px 12px;
        border-bottom: 1px solid #2f2f2f;
    }
    
    /* Sidebar content */
    .sidebar-content {
        padding: 12px;
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }
    
    /* Summary output area */
    .summary-output {
        max-height: 300px !important;
        overflow-y: auto !important;
        background: #2f2f2f !important;
        border: 1px solid #4d4d4f !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin-top: 8px !important;
        font-size: 13px !important;
        line-height: 1.4 !important;
        color: #ececf1 !important;
    }
    
    .summary-output::-webkit-scrollbar {
        width: 6px;
    }
    
    .summary-output::-webkit-scrollbar-track {
        background: #1a1a1a;
        border-radius: 3px;
    }
    
    .summary-output::-webkit-scrollbar-thumb {
        background: #4d4d4f;
        border-radius: 3px;
    }
    
    .summary-output::-webkit-scrollbar-thumb:hover {
        background: #6d6d6f;
    }
    
    /* Chat header */
    .chat-header {
        padding: 12px 0;
        text-align: center;
        border-bottom: 1px solid #2f2f2f;
    }
    
    /* Chat container */
    .chat-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        max-width: 800px;
        margin: 0 auto;
        width: 100%;
        padding: 0 24px;
    }
    
    /* Input area at bottom */
    .input-container { 
        padding: 24px;
        background: #212121;
        border-top: 1px solid #2f2f2f;
        max-width: 800px;
        margin: 0 auto;
        width: 100%;
    }
    
    /* New chat button */
    .new-chat-btn {
        background: transparent !important;
        border: 1px solid #4d4d4f !important;
        color: #ececf1 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        width: 100% !important;
        text-align: left !important;
        margin-bottom: 12px !important;
        font-weight: normal !important;
    }
    .new-chat-btn:hover {
        background: #2f2f2f !important;
    }
    
    /* Sidebar buttons */
    .sidebar-btn {
        background: transparent !important;
        border: none !important;
        color: #ececf1 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        width: 100% !important;
        text-align: left !important;
        margin: 4px 0 !important;
        font-weight: normal !important;
    }
    .sidebar-btn:hover {
        background: #2f2f2f !important;
    }
    
    /* Primary buttons */
    .primary-btn {
        background: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .primary-btn:hover {
        background: #0d8a6b !important;
    }
    
    /* Text input - ChatGPT style */
    .gradio-textbox textarea {
        background: #2f2f2f !important;
        border: 1px solid #4d4d4f !important;
        border-radius: 12px !important;
        color: #ececf1 !important;
        padding: 16px !important;
        font-size: 16px !important;
        resize: none !important;
        outline: none !important;
    }
    .gradio-textbox textarea:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2) !important;
    }
    .gradio-textbox textarea::placeholder {
        color: #8e8ea0 !important;
    }
    
    /* File upload area */
    .gradio-file {
        background: #2f2f2f !important;
        border: 1px dashed #4d4d4f !important;
        border-radius: 8px !important;
        color: #ececf1 !important;
        padding: 16px !important;
    }
    .gradio-file:hover {
        border-color: #10a37f !important;
        background: #3a3a3a !important;
    }
    
    /* Chatbot styling */
    .gradio-chatbot {
        background: transparent !important;
        border: none !important;
        flex: 1 !important;
    }
    
    /* Message styling */
    .message {
        margin: 16px 0 !important;
        padding: 0 !important;
    }
    
    /* Hide labels */
    .gradio-textbox label,
    .gradio-file label,
    .gradio-chatbot label {
        display: none !important;
    }
    
    /* Markdown content */
    .gradio-markdown {
        background: transparent !important;
        color: #ececf1 !important;
    }
    .gradio-markdown h3 {
        color: #ececf1 !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        font-size: 14px !important;
    }
    .gradio-markdown p {
        color: #8e8ea0 !important;
        font-size: 13px !important;
        line-height: 1.4 !important;
    }
    
    /* Status text */
    .status-text {
        color: #10a37f !important;
        font-size: 13px !important;
        padding: 8px 0 !important;
    }
    
    /* Send button container */
    .send-btn-container {
        position: relative;
    }
    .send-btn {
        position: absolute !important;
        right: 8px !important;
        bottom: 8px !important;
        background: #ececf1 !important;
        color: #2f2f2f !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
    }
    .send-btn:hover {
        background: #d1d5db !important;
    }
    .send-btn:disabled {
        background: #6b7280 !important;
        color: #9ca3af !important;
    }
    
    /* Hide default gradio elements */
    .gradio-container .wrap {
        border: none !important;
        background: transparent !important;
    }
    
    /* Main title */
    .main-title {
        color: #ececf1 !important;
        font-size: 32px !important;
        font-weight: 600 !important;
        text-align: center !important;
        margin: 48px 0 !important;
    }
    
    /* Center welcome message */
    .welcome-container {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }
    """

    with gr.Blocks(title="Smart Contract Assistant", css=custom_css) as demo:
        with gr.Row(elem_classes=["app-container"]):
            # Left Sidebar
            with gr.Column(scale=2, elem_classes=["sidebar"]):
                with gr.Column(elem_classes=["sidebar-header"]):
                    new_chat_btn = gr.Button("+ New chat", elem_classes=["new-chat-btn"])
                
                with gr.Column(elem_classes=["sidebar-content"]):
                    gr.Markdown("### 📁 Upload & Index")
                    upload = gr.File(file_count="multiple")
                    index_btn = gr.Button("Index Documents", elem_classes=["primary-btn"])
                    status = gr.Markdown("Ready to index documents", elem_classes=["status-text"])
                    
                    gr.Markdown("---")
                    gr.Markdown("### 📄 Summary")
                    summarize_btn = gr.Button("Summarize All", elem_classes=["sidebar-btn"])
                    summary_out = gr.Markdown(elem_classes=["summary-output"])
                    
                    gr.Markdown("---")
                    gr.Markdown("### ℹ️ Info")
                    gr.Markdown("""
                    • RAG-powered contract analysis
                    • Upload PDF, DOCX, TXT files  
                    • Ask questions about content
                    • Get document summaries
                    """)

            # Main Chat Area
            with gr.Column(scale=6, elem_classes=["main-chat"]):
                # Chat content
                with gr.Column(elem_classes=["chat-container"]):
                    # Welcome message when no chat
                    with gr.Column(elem_classes=["welcome-container"]):
                        gr.Markdown("# What can I help with?", elem_classes=["main-title"])
                    
                    # Chat history
                    history_state = gr.State([])
                    chat_out = gr.Chatbot(
                        height=400,
                        show_label=False,
                        container=False,
                        avatar_images=(None, None),
                    )
                
                # Input area
                with gr.Column(elem_classes=["input-container"]):
                    with gr.Row():
                        with gr.Column(scale=10, elem_classes=["send-btn-container"]):
                            q_in = gr.Textbox(
                                placeholder="Ask anything about your contracts...",
                                lines=1,
                                max_lines=5,
                                show_label=False,
                                container=False,
                            )
                            ask_btn = gr.Button("↑", elem_classes=["send-btn"], scale=0)

        # Event handlers
        def on_index(files):
            msg, _ = index_files(files)
            return msg

        def clear_chat():
            return [], []

        index_btn.click(on_index, inputs=[upload], outputs=[status])
        ask_btn.click(stream_answer, inputs=[q_in, history_state], outputs=[chat_out, history_state])
        new_chat_btn.click(clear_chat, outputs=[chat_out, history_state])
        summarize_btn.click(summarize_sync, outputs=[summary_out])

    return demo


if __name__ == "__main__":
    ensure_dirs()
    app = build_ui()
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
