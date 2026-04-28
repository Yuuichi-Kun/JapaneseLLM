import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os
import re

# ── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="Nihongo Assistant", layout="wide")
st.title("🇯🇵 AI Asisten Belajar Bahasa Jepang")

# ── Constants ─────────────────────────────────────────────
TEXT_PATH = "minna_text.txt"
DB_PATH = "./chroma_db_jepang"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Session State ─────────────────────────────────────────
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ── Helper Functions ──────────────────────────────────────

def clean_text(text):
    """Remove watermarks and excessive whitespace from OCR text."""
    text = re.sub(r'www[\.,]japandaisuki[\.,]com', '', text)
    text = re.sub(r'\s{3,}', ' ', text)
    return text.strip()


def load_pages_from_text(path):
    """Parse the OCR text file into a list of LangChain Documents, one per page."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    page_blocks = raw.split("=== HALAMAN ")
    docs = []

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
            docs.append(Document(
                page_content=content,
                metadata={"page": page_num}
            ))

    return docs


def get_page_content(path, page_num):
    """Directly extract a single page's content from the text file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    page_blocks = raw.split("=== HALAMAN ")

    for block in page_blocks:
        if not block.strip():
            continue

        lines = block.split("\n", 1)
        try:
            num = int(lines[0].strip().replace("===", "").strip())
        except ValueError:
            continue

        if num == page_num:
            content = lines[1].strip() if len(lines) > 1 else ""
            return clean_text(content)

    return None


def get_embeddings():
    """Load the multilingual embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Setup Materi")

    # Status check
    if os.path.exists(TEXT_PATH):
        size_kb = os.path.getsize(TEXT_PATH) / 1024
        st.success(f"✅ File teks ditemukan ({size_kb:.0f} KB)")
    else:
        st.error("❌ minna_text.txt belum ada. Jalankan run_ocr_once.py dulu!")
        st.code("python run_ocr_once.py", language="bash")

    if os.path.exists(DB_PATH):
        st.info("✅ Database vektor ditemukan")

    st.divider()

    # Process text → vector DB
    if os.path.exists(TEXT_PATH):
        if st.button("🔄 Proses ke Database", use_container_width=True):
            with st.spinner("Membaca teks..."):
                docs = load_pages_from_text(TEXT_PATH)

            st.info(f"Total halaman dimuat: {len(docs)}")

            if not docs:
                st.error("File teks kosong atau format salah.")
                st.stop()

            with st.expander("Preview halaman pertama (cek kualitas OCR)"):
                st.write(docs[0].page_content[:500])

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=80,
                separators=["\n\n", "\n", "。", "、", " ", ""]
            )

            with st.spinner("Memotong teks menjadi chunks..."):
                chunks = splitter.split_documents(docs)

            st.info(f"Total chunks: {len(chunks)}")

            with st.spinner("Membuat embedding (pertama kali ~400MB download)..."):
                embeddings = get_embeddings()

            with st.spinner("Menyimpan ke database vektor..."):
                st.session_state.vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=DB_PATH
                )

            st.success(f"✅ Selesai! {len(chunks)} chunks tersimpan.")

    # Load existing DB
    if os.path.exists(DB_PATH):
        if st.button("📂 Muat Database yang Ada", use_container_width=True):
            with st.spinner("Memuat database..."):
                embeddings = get_embeddings()
                st.session_state.vector_db = Chroma(
                    persist_directory=DB_PATH,
                    embedding_function=embeddings
                )
            st.success("✅ Database berhasil dimuat!")

    st.divider()

    # Setup guide
    with st.expander("📋 Panduan Setup"):
        st.markdown("""
        **Langkah pertama kali:**
        1. Install Tesseract:
           - Download: https://github.com/UB-Mannheim/tesseract/wiki
           - Centang **Japanese** & **Indonesian** saat install
        2. Install packages:
        ```
        pip install pytesseract pdf2image pillow
        ```
        3. Jalankan OCR:
        ```
        python run_ocr_once.py
        ```
        4. Klik **Proses ke Database**

        **Selanjutnya:**
        Cukup klik **Muat Database yang Ada**

        **Contoh pertanyaan:**
        - Apa yang dipelajari pada halaman 25?
        - Apa arti partikel は?
        - Cara menggunakan kata kerja bentuk て?
        - Jelaskan pelajaran 1
        """)


# ── Main Chat Area ────────────────────────────────────────
if st.session_state.vector_db:

    query = st.text_input(
        "Tanya apa saja tentang Minna no Nihongo:",
        placeholder="Contoh: Apa yang dipelajari pada halaman 25?"
    )
    show_debug = st.checkbox("Debug: tampilkan konteks yang digunakan")

    if query:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )

        # ── Detect page-specific questions ───────────────
        page_match = re.search(r'halaman\s*(\d+)', query.lower())

        if page_match:
            # Direct page lookup — bypass vector search entirely
            target_page = int(page_match.group(1))

            if not os.path.exists(TEXT_PATH):
                st.error("File minna_text.txt diperlukan untuk pencarian langsung per halaman.")
                st.stop()

            target_content = get_page_content(TEXT_PATH, target_page)

            if target_content:
                context = f"[Hal. {target_page}]\n{target_content}"
            else:
                context = f"Halaman {target_page} tidak ditemukan dalam buku."

            if show_debug:
                with st.expander(f"📄 Konten halaman {target_page} (dari file teks langsung)"):
                    st.write(context)

        else:
            # Semantic search for general questions
            retriever = st.session_state.vector_db.as_retriever(
                search_kwargs={"k": 6}
            )
            retrieved_docs = retriever.invoke(query)

            if show_debug:
                st.markdown("**Chunks yang ditemukan:**")
                for i, doc in enumerate(retrieved_docs):
                    page = doc.metadata.get('page', '?')
                    with st.expander(f"Chunk {i+1} — hal. {page}"):
                        st.write(doc.page_content)

            context = "\n\n---\n\n".join(
                f"[Hal. {d.metadata.get('page', '?')}]\n{d.page_content}"
                for d in retrieved_docs
            )

        # ── Shared prompt & chain ─────────────────────────
        prompt = PromptTemplate.from_template("""Kamu adalah asisten belajar bahasa Jepang yang membantu pengguna memahami isi buku Minna no Nihongo.
Jawab HANYA berdasarkan konteks di bawah ini. Jangan gunakan pengetahuan lain di luar konteks.
Jika jawabannya tidak ada dalam konteks, katakan: "Informasi ini tidak ada dalam buku."
Jawab dalam bahasa Indonesia kecuali diminta lain.

Konteks dari buku:
{context}

Pertanyaan: {question}
Jawaban (berdasarkan konteks di atas):""")

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Mencari jawaban..."):
            response = chain.invoke({"context": context, "question": query})

        st.markdown("### Jawaban:")
        st.markdown(response)

else:
    st.info("👈 Klik **Muat Database yang Ada** atau **Proses ke Database** di sidebar untuk mulai.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🆕 Pertama kali?
        1. Jalankan `python run_ocr_once.py`
        2. Klik **Proses ke Database** di sidebar
        3. Mulai bertanya!
        """)

    with col2:
        st.markdown("""
        #### 🔁 Sudah punya database?
        1. Klik **Muat Database yang Ada**
        2. Langsung mulai bertanya!

        _(Tidak perlu proses ulang)_
        """)