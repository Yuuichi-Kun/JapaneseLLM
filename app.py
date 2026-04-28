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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db_jepang")
OCR_SCRIPT_PATH = os.path.join(BASE_DIR, "run_ocr_once.py")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_ocr_output_path(script_path):
    """Read OUTPUT_PATH value from run_ocr_once.py if available."""
    if not os.path.exists(script_path):
        return None

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            script = f.read()
    except OSError:
        return None

    match = re.search(r'OUTPUT_PATH\s*=\s*r?["\']([^"\']+)["\']', script)
    if match:
        return os.path.normpath(match.group(1))
    return None


def resolve_text_path():
    """Find minna_text.txt from common run locations."""
    ocr_output_path = get_ocr_output_path(OCR_SCRIPT_PATH)
    candidates = [
        ocr_output_path,
        os.path.join(BASE_DIR, "minna_text.txt"),
        os.path.join(os.getcwd(), "minna_text.txt"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    # Default to project file location for creation guidance.
    return ocr_output_path or os.path.join(BASE_DIR, "minna_text.txt")


TEXT_PATH = resolve_text_path()

# ── Session State ─────────────────────────────────────────
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "text_path" not in st.session_state:
    st.session_state.text_path = TEXT_PATH

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


def get_lesson_content(path, lesson_num):
    """Extract chunks related to a specific lesson number from OCR text."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Flexible patterns to handle OCR variants (Pelajaran 9 / Lesson 9 / 第9課)
    lesson_pattern = (
        rf"(pelajaran|lesson)\s*[:\-]?\s*{lesson_num}\b|"
        rf"第\s*{lesson_num}\s*課"
    )
    lines = raw.splitlines()

    matching_indices = []
    for i, line in enumerate(lines):
        if re.search(lesson_pattern, line, flags=re.IGNORECASE):
            matching_indices.append(i)

    if not matching_indices:
        return None

    snippets = []
    for idx in matching_indices[:3]:
        start = max(0, idx - 5)
        end = min(len(lines), idx + 21)
        snippet = "\n".join(lines[start:end]).strip()
        snippet = clean_text(snippet)
        if snippet:
            snippets.append(snippet)

    if not snippets:
        return None

    # De-duplicate repeated OCR blocks
    unique = list(dict.fromkeys(snippets))
    return "\n\n---\n\n".join(unique)


def get_embeddings():
    """Load the multilingual embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Setup Materi")
    st.caption("Atur lokasi file teks Minna no Nihongo")
    st.session_state.text_path = st.text_input(
        "Path `minna_text.txt`",
        value=st.session_state.text_path,
        help="Bisa isi path folder lain, contoh: D:/data/minna_text.txt",
    ).strip().strip('"')
    active_text_path = st.session_state.text_path or TEXT_PATH

    # Status check
    if os.path.exists(active_text_path):
        size_kb = os.path.getsize(active_text_path) / 1024
        st.success(f"✅ File teks ditemukan ({size_kb:.0f} KB)")
        st.caption(f"Path aktif: `{active_text_path}`")
    else:
        st.error("❌ File teks belum ditemukan pada path di atas.")
        st.code("python run_ocr_once.py", language="bash")

    if os.path.exists(DB_PATH):
        st.info("✅ Database vektor ditemukan")

    st.divider()

    # Process text → vector DB
    if os.path.exists(active_text_path):
        if st.button("🔄 Proses ke Database", use_container_width=True):
            with st.spinner("Membaca teks..."):
                docs = load_pages_from_text(active_text_path)

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

        # ── Detect page-specific / lesson-specific questions ───────────────
        page_match = re.search(r'halaman\s*(\d+)', query.lower())
        lesson_match = re.search(r'(pelajaran|lesson)\s*(\d+)', query.lower())

        if page_match:
            # Direct page lookup — bypass vector search entirely
            target_page = int(page_match.group(1))

            active_text_path = st.session_state.text_path or TEXT_PATH
            target_content = get_page_content(active_text_path, target_page) if os.path.exists(active_text_path) else None

            if target_content:
                context = f"[Hal. {target_page}]\n{target_content}"
            else:
                # If text file is unavailable (or page missing), fallback to semantic retrieval.
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": 8}
                )
                focused_query = f"halaman {target_page} Minna no Nihongo"
                retrieved_docs = retriever.invoke(focused_query)
                context = "\n\n---\n\n".join(
                    f"[Hal. {d.metadata.get('page', '?')}]\n{d.page_content}"
                    for d in retrieved_docs
                )

                if not os.path.exists(active_text_path):
                    st.warning("minna_text.txt tidak ditemukan, jadi menggunakan pencarian dari database vektor.")

            if show_debug:
                with st.expander(f"📄 Konten halaman {target_page} (dari file teks langsung)"):
                    st.write(context)

        elif lesson_match:
            # Direct lesson lookup to avoid semantic miss for exact lesson requests
            target_lesson = int(lesson_match.group(2))

            active_text_path = st.session_state.text_path or TEXT_PATH
            lesson_content = get_lesson_content(active_text_path, target_lesson) if os.path.exists(active_text_path) else None

            if lesson_content:
                context = f"[Pelajaran {target_lesson}]\n{lesson_content}"
            else:
                # Fallback to semantic retrieval with focused query if direct scan misses
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": 8}
                )
                focused_query = f"Pelajaran {target_lesson} Minna no Nihongo"
                retrieved_docs = retriever.invoke(focused_query)
                context = "\n\n---\n\n".join(
                    f"[Hal. {d.metadata.get('page', '?')}]\n{d.page_content}"
                    for d in retrieved_docs
                )

                if not os.path.exists(active_text_path):
                    st.warning("minna_text.txt tidak ditemukan, jadi menggunakan pencarian dari database vektor.")

            if show_debug:
                with st.expander(f"📘 Konten pelajaran {target_lesson}"):
                    st.write(context if context else "Tidak ada konteks ditemukan.")

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