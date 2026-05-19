"""Lesson 07 · RAG advanced — MultiQuery + contextual compression.

Compares three retrievers on the same question so you can see the
difference qualitatively.

Run:
    uv run python -m lessons.07_rag_advanced.example
"""

from langchain_classic.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared import get_llm, settings
from shared.pretty import console, section


PERSIST_DIR = str(settings.data_dir / "chroma_lesson_07")


def build_store() -> Chroma:
    docs = []
    for name in ("langgraph_intro.md", "company_handbook.md"):
        path = settings.data_dir / "sample_docs" / name
        docs.extend(TextLoader(str(path), encoding="utf-8").load())

    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_documents(docs)
    return Chroma.from_documents(
        chunks,
        embedding=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
        persist_directory=PERSIST_DIR,
        collection_name="lesson07",
    )


def show(label: str, docs) -> None:
    console.print(f"\n[bold cyan]{label}[/] — {len(docs)} chunk(s)")
    for i, d in enumerate(docs, 1):
        preview = d.page_content.strip().replace("\n", " ")[:120]
        console.print(f"  [{i}] {preview}…")


def main() -> None:
    store = build_store()
    llm = get_llm()
    question = "What should I do if a customer asks for a refund larger than $100?"

    section("Baseline retriever (similarity search, k=4)")
    base = store.as_retriever(search_kwargs={"k": 4})
    show("baseline", base.invoke(question))

    section("MultiQueryRetriever — the LLM expands the query into variants")
    multi = MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    show("multi-query", multi.invoke(question))

    section("ContextualCompression — keep only sentences that actually answer Q")
    compressor = LLMChainExtractor.from_llm(llm)
    compressed = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base
    )
    show("compressed", compressed.invoke(question))


if __name__ == "__main__":
    main()
