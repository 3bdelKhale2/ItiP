import os
import time
from typing import List, Tuple

# Import your NVIDIA API key here
# os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key_here"

from rag_pipeline import build_qa_chain

TEST_QUESTIONS = [
    "What is the purpose of the contract?",
    "List key obligations of the parties.",
    "Are there termination conditions?",
    "What are the risks mentioned?",
    "Define any important terms.",
    "What is the effective date?",
    "Is there an arbitration clause?",
    "What is the governing law?",
    "Payment terms?",
    "Limitations of liability?",
]


def has_citation(text: str) -> bool:
    return "[" in text and "]" in text and "chunk_" in text


def is_idk(text: str) -> bool:
    t = text.lower()
    return "i don't know" in t or "i do not know" in t or "only answer from the uploaded documents" in t


def run_evaluation() -> Tuple[float, float, float]:
    chain = build_qa_chain()
    answered_with_cite = 0
    idk_count = 0
    latencies: List[float] = []

    for q in TEST_QUESTIONS:
        start = time.time()
        buf = []
        for chunk in chain.stream({"question": q}):
            if hasattr(chunk, "content"):
                buf.append(chunk.content)
            else:
                buf.append(str(chunk))
        text = "".join(buf)
        end = time.time()
        latencies.append(end - start)
        if has_citation(text):
            answered_with_cite += 1
        if is_idk(text):
            idk_count += 1

    cite_pct = answered_with_cite / len(TEST_QUESTIONS) * 100.0
    idk_pct = idk_count / len(TEST_QUESTIONS) * 100.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return cite_pct, idk_pct, avg_latency


if __name__ == "__main__":
    try:
        cite_pct, idk_pct, avg_latency = run_evaluation()
        print(f"% answered with citations: {cite_pct:.1f}%")
        print(f"% 'I don't know' responses: {idk_pct:.1f}%")
        print(f"Average latency: {avg_latency:.2f}s")
    except RuntimeError as e:
        print(str(e))
