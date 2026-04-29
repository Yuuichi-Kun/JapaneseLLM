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
PUBLIC_MODE = True


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
    env_path = os.getenv("MINNA_TEXT_PATH")
    ocr_output_path = get_ocr_output_path(OCR_SCRIPT_PATH)
    candidates = [
        env_path,
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


def lesson_number_appears_in_text(lesson_num: int, text: str) -> bool:
    """Match lesson markers in OCR/Japanese text (ASCII and fullwidth digits)."""
    n = str(lesson_num)
    n_fw = n.translate(str.maketrans("0123456789", "０１２３４５６７８９"))
    patterns = [
        rf"第\s*{re.escape(n)}\s*課",
        rf"第\s*{re.escape(n_fw)}\s*課",
        rf"(?:pelajaran|lesson)\s*[:\-]?\s*{re.escape(n)}(?:\D|$)",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Status")
    if os.path.exists(DB_PATH):
        st.success("✅ Database siap digunakan")
    else:
        st.error("❌ Database belum ditemukan di server.")

    if st.session_state.vector_db:
        st.info("✅ Asisten siap menjawab")
    else:
        st.warning("⏳ Menunggu database dimuat")

    if PUBLIC_MODE:
        st.caption("Mode publik aktif: pengguna tidak perlu upload file OCR.")
        if os.path.exists(TEXT_PATH):
            st.caption("Sumber teks penuh: `minna_text.txt` tersedia di server (lookup halaman/pelajaran lebih akurat).")
        else:
            st.caption("Di hosting: isi buku hanya lewat database vektor (`chroma_db_jepang`), bukan file `minna_text.txt`.")

    with st.expander("🛠️ Panduan Admin"):
        st.markdown("""
        **Untuk update materi:**
        1. Jalankan `python run_ocr_once.py` di lokal (sekali saat update buku)
        2. Bangun ulang folder `chroma_db_jepang`
        3. Commit hasil terbaru ke GitHub
        4. Streamlit akan redeploy otomatis
        """)


# ── Main Chat Area ────────────────────────────────────────
# Auto-load persisted DB for public deployment
if st.session_state.vector_db is None and os.path.exists(DB_PATH):
    with st.spinner("Menyiapkan database..."):
        embeddings = get_embeddings()
        st.session_state.vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

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
            # Page query: prioritize exact page metadata from vector DB.
            target_page = int(page_match.group(1))

            target_content = get_page_content(TEXT_PATH, target_page) if os.path.exists(TEXT_PATH) else None

            if target_content:
                context = f"[Hal. {target_page}]\n{target_content}"
            else:
                # First try exact metadata filter by page.
                retrieved_docs = st.session_state.vector_db.similarity_search(
                    query=query,
                    k=8,
                    filter={"page": target_page}
                )

                # If empty, fallback to broader semantic retrieval.
                if not retrieved_docs:
                    retriever = st.session_state.vector_db.as_retriever(
                        search_kwargs={"k": 10}
                    )
                    fallback_queries = [
                        query,
                        f"halaman {target_page} Minna no Nihongo",
                        f"isi halaman {target_page}",
                    ]
                    seen = set()
                    merged = []
                    for q in fallback_queries:
                        for doc in retriever.invoke(q):
                            key = (doc.metadata.get("page"), doc.page_content[:120])
                            if key not in seen:
                                seen.add(key)
                                merged.append(doc)
                    retrieved_docs = merged[:8]

                context = "\n\n---\n\n".join(
                    f"[Hal. {d.metadata.get('page', '?')}]\n{d.page_content}"
                    for d in retrieved_docs
                )
                if not context:
                    context = f"Tidak ditemukan konteks untuk halaman {target_page} di database."

            if show_debug:
                with st.expander(f"📄 Konten halaman {target_page} (dari file teks langsung)"):
                    st.write(context)

        elif lesson_match:
            # Direct lesson lookup to avoid semantic miss for exact lesson requests
            target_lesson = int(lesson_match.group(2))

            lesson_content = get_lesson_content(TEXT_PATH, target_lesson) if os.path.exists(TEXT_PATH) else None

            if lesson_content:
                context = f"[Pelajaran {target_lesson}]\n{lesson_content}"
            else:
                # Fallback to semantic retrieval with focused query if direct scan misses
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": 16}
                )
                n_fw = str(target_lesson).translate(
                    str.maketrans("0123456789", "０１２３４５６７８９")
                )
                lesson_queries = [
                    query,
                    f"Pelajaran {target_lesson} Minna no Nihongo",
                    f"Bab {target_lesson} Minna no Nihongo",
                    f"Lesson {target_lesson} Minna no Nihongo",
                    f"第{target_lesson}課",
                    f"第{n_fw}課",
                ]
                retrieved_docs = []
                seen = set()
                for q in lesson_queries:
                    for doc in retriever.invoke(q):
                        key = (doc.metadata.get("page"), doc.page_content[:120])
                        if key not in seen:
                            seen.add(key)
                            retrieved_docs.append(doc)

                filtered_docs = [
                    d for d in retrieved_docs
                    if lesson_number_appears_in_text(target_lesson, d.page_content)
                ]
                final_docs = filtered_docs[:10] if filtered_docs else retrieved_docs[:10]
                context = "\n\n---\n\n".join(
                    f"[Hal. {d.metadata.get('page', '?')}]\n{d.page_content}"
                    for d in final_docs
                )

                if not context:
                    context = f"Tidak ditemukan konteks untuk pelajaran {target_lesson} di database."

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
    st.error("Database belum siap di server. Hubungi admin untuk memastikan folder `chroma_db_jepang` ikut ter-deploy.")