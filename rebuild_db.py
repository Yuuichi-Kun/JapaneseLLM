import os
import re
import shutil
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_PATH = os.path.join(BASE_DIR, "minna_text.txt")
DB_PATH = os.path.join(BASE_DIR, "chroma_db_jepang")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def clean_text(text: str) -> str:
    text = re.sub(r"www[\.,]japandaisuki[\.,]com", "", text)
    text = re.sub(r"\s{3,}", " ", text)
    return text.strip()


def load_pages_from_text(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    page_blocks = raw.split("=== HALAMAN ")
    docs: list[Document] = []

    for block in page_blocks:
        if not block.strip():
            continue

        lines = block.split("\n", 1)
        try:
            page_num = int(lines[0].strip().replace("===", "").strip())
        except ValueError:
            continue

        content = lines[1].strip() if len(lines) > 1 else ""
        content = clean_text(content)

        if content:
            docs.append(Document(page_content=content, metadata={"page": page_num}))

    return docs


def main() -> None:
    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {TEXT_PATH}")

    print("Membaca minna_text.txt ...")
    docs = load_pages_from_text(TEXT_PATH)
    if not docs:
        raise RuntimeError("Tidak ada halaman valid ditemukan di minna_text.txt")
    print(f"Total halaman valid: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    if os.path.exists(DB_PATH):
        print("Menghapus database lama ...")
        shutil.rmtree(DB_PATH)

    print("Memuat model embedding ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Membangun Chroma DB baru ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    print(f"Selesai. Database baru tersimpan di: {DB_PATH}")


if __name__ == "__main__":
    main()
