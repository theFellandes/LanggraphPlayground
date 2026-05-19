"""Lesson 06 · RAG basics — load → split → embed → store → retrieve → generate.

Uses local on-CPU embeddings (FastEmbed) so no extra API key is needed
beyond your chosen LLM provider.

Run:
    uv run python -m lessons.06_rag_basics.example
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared import get_llm, settings
from shared.pretty import console, section


PERSIST_DIR = str(settings.data_dir / "chroma_lesson_06")
SOURCE_DOCS = [
    settings.data_dir / "sample_docs" / "langgraph_intro.md",
    settings.data_dir / "sample_docs" / "company_handbook.md",
]


def build_vectorstore() -> Chroma:
    section("1 · load + split + embed + store")

    docs = []
    for path in SOURCE_DOCS:
        docs.extend(TextLoader(str(path), encoding="utf-8").load())
    console.print(f"Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    console.print(f"Split into {len(chunks)} chunks.")

    embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=PERSIST_DIR,
        collection_name="lesson06",
    )
    console.print(f"Indexed into Chroma at [dim]{PERSIST_DIR}[/]")
    return store


def ask(store: Chroma, question: str) -> None:
    section(f"Q: {question}")

    retriever = store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using ONLY the provided context. "
                       "If the answer isn't in the context, say so explicitly."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    def format_context(docs) -> str:
        return "\n\n---\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    console.print(chain.invoke(question))


def main() -> None:
    store = build_vectorstore()
    ask(store, "How many days of PTO do full-time employees get?")
    ask(store, "What's the difference between LCEL and LangGraph?")
    ask(store, "What is Acme's policy on remote work equipment?")


if __name__ == "__main__":
    main()
