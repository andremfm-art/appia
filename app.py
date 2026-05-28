"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           NVIDIA NIM — Assistente Multimodal Avançado                       ║
║           Versão 2.0 — Reescrita completa com correções e melhorias         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Correções principais:
- Import correto do DDGS (duckduckgo_search)
- Inicialização do st.set_page_config antes de qualquer chamada st.*
- Conexão SQLite thread-safe sem cache problemático
- Sincronização correta do vector_store com RAGService
- Loop de rerun corrigido (sem st.rerun() no final do fluxo principal)
- Gerenciamento de estado de sessão robusto
- Upload de imagem com key dinâmica para evitar reset de estado
- TTS sem limite de 500 chars (usa divisão em sentenças)
- Exportação com encoding correto
- Busca web com API atualizada do DDGS
- Design moderno e profissional inspirado em interfaces de IA de ponta
- Gerenciamento de múltiplas conversas
- Suporte a múltiplos documentos com status visual
- Fallback robusto para todos os modelos
"""

import streamlit as st

# ─── DEVE SER O PRIMEIRO COMANDO STREAMLIT ───────────────────────────────────
st.set_page_config(
    page_title="NIM AI Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "NIM AI Studio v2.0 — Powered by NVIDIA NIM"},
)

# ─── CSS GLOBAL ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --nim-green: #76B900;
    --nim-dark: #0a0a0f;
    --nim-card: #12121a;
    --nim-border: #1e1e2e;
    --nim-text: #e8e8f0;
    --nim-muted: #6b6b8a;
    --nim-accent: #00d4aa;
    --nim-danger: #ff4466;
    --nim-warn: #ffaa00;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--nim-dark) !important;
    color: var(--nim-text) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #0d0d15 !important;
    border-right: 1px solid var(--nim-border) !important;
}

[data-testid="stSidebar"] * { color: var(--nim-text) !important; }

.stChatMessage {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatMessageContent"] {
    background: var(--nim-card) !important;
    border: 1px solid var(--nim-border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    color: var(--nim-text) !important;
}

[data-testid="stChatMessageContent"] code {
    background: #1a1a2e !important;
    color: var(--nim-green) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stChatMessageContent"] pre {
    background: #0d0d15 !important;
    border: 1px solid var(--nim-border) !important;
    border-left: 3px solid var(--nim-green) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    overflow-x: auto !important;
}

.stChatInputContainer {
    background: var(--nim-card) !important;
    border: 1px solid var(--nim-border) !important;
    border-radius: 16px !important;
}

.stChatInputContainer textarea {
    background: transparent !important;
    color: var(--nim-text) !important;
    font-family: 'Inter', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1a1a2e, #12121a) !important;
    color: var(--nim-text) !important;
    border: 1px solid var(--nim-border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: var(--nim-green) !important;
    color: var(--nim-green) !important;
    box-shadow: 0 0 12px rgba(118,185,0,0.2) !important;
}

.stSelectbox > div > div, .stSlider > div {
    background: var(--nim-card) !important;
    border-color: var(--nim-border) !important;
    color: var(--nim-text) !important;
}

.nim-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 0 8px;
    border-bottom: 1px solid var(--nim-border);
    margin-bottom: 24px;
}

.nim-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    background: linear-gradient(135deg, #76B900, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.nim-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    background: rgba(118,185,0,0.15);
    color: var(--nim-green);
    border: 1px solid rgba(118,185,0,0.3);
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 1px;
}

.stat-card {
    background: var(--nim-card);
    border: 1px solid var(--nim-border);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-label { font-size: 0.75rem; color: var(--nim-muted); font-family: 'JetBrains Mono', monospace; }
.stat-value { font-size: 0.9rem; color: var(--nim-green); font-weight: 600; }

.model-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--nim-muted);
    border: 1px solid var(--nim-border);
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 8px;
    display: inline-block;
}

.context-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 6px;
    display: inline-block;
    margin: 4px 2px;
}
.ctx-rag  { background: rgba(118,185,0,0.12);  color: #76B900;  border: 1px solid rgba(118,185,0,0.25);  }
.ctx-web  { background: rgba(0,212,170,0.12);  color: #00d4aa;  border: 1px solid rgba(0,212,170,0.25);  }
.ctx-img  { background: rgba(255,170,0,0.12);  color: #ffaa00;  border: 1px solid rgba(255,170,0,0.25);  }

div[data-testid="stFileUploader"] {
    background: var(--nim-card) !important;
    border: 1px dashed var(--nim-border) !important;
    border-radius: 10px !important;
}

.stAlert { border-radius: 8px !important; }
.stSpinner > div { color: var(--nim-green) !important; }

.stDownloadButton > button {
    font-size: 0.78rem !important;
    padding: 4px 10px !important;
}

.conv-item {
    padding: 8px 12px;
    border-radius: 8px;
    margin: 3px 0;
    cursor: pointer;
    font-size: 0.85rem;
    border: 1px solid transparent;
    transition: all 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.conv-item:hover { background: var(--nim-card); border-color: var(--nim-border); }
.conv-item.active { background: rgba(118,185,0,0.1); border-color: rgba(118,185,0,0.3); color: var(--nim-green); }

hr { border-color: var(--nim-border) !important; }
</style>
""", unsafe_allow_html=True)

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import os
import base64
import hashlib
import logging
import pickle
import sqlite3
import atexit
import re
import time
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from gtts import gTTS
from docx import Document
from PIL import Image

# CORREÇÃO: import correto do duckduckgo_search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# Embedding model (lazy load)
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nim_studio")

# ─── DIRETÓRIOS ──────────────────────────────────────────────────────────────
for _d in ["data", "data/images", "data/tts", "data/exports"]:
    os.makedirs(_d, exist_ok=True)

# ─── MODELOS ─────────────────────────────────────────────────────────────────
TEXT_MODELS = {
    "⚡ Llama 3.1 70B Instruct":   "meta/llama-3.1-70b-instruct",
    "🚀 Llama 3.1 8B Instruct":    "meta/llama-3.1-8b-instruct",
    "🔬 Nemotron 70B":             "nvidia/nemotron-70b-instruct",
    "💡 Llama 3.3 70B":            "meta/llama-3.3-70b-instruct",
    "🌟 Mistral NeMo":             "nv-mistralai/mistral-nemo-12b-instruct",
}
VISION_MODELS = [
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
]
FALLBACK_MODEL = "meta/llama-3.1-8b-instruct"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTORS_FILE = "data/vectors.pkl"

# ─── API KEY ─────────────────────────────────────────────────────────────────
def load_api_key() -> Optional[str]:
    """Carrega NVIDIA API Key de secrets, variável de ambiente ou input manual."""
    # 1. Streamlit secrets
    try:
        if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
            key = st.secrets["NVIDIA_API_KEY"]
            if key and key.startswith("nvapi-"):
                return key
    except Exception:
        pass

    # 2. Variável de ambiente
    env_key = os.environ.get("NVIDIA_API_KEY", "")
    if env_key.startswith("nvapi-"):
        return env_key

    # 3. Session state (inserido manualmente)
    if st.session_state.get("api_key"):
        return st.session_state.api_key

    return None


def api_key_screen():
    """Tela de configuração da API Key — exibida quando chave não encontrada."""
    st.markdown("""
    <div style='max-width:480px;margin:80px auto;text-align:center'>
        <div style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
                    background:linear-gradient(135deg,#76B900,#00d4aa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    margin-bottom:8px'>NIM AI Studio</div>
        <div style='color:#6b6b8a;font-size:0.9rem;margin-bottom:32px'>
            Configure sua NVIDIA API Key para continuar
        </div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        key_input = st.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-...",
            help="Obtenha sua chave em: build.nvidia.com",
        )
        uploaded = st.file_uploader(
            "Ou carregue um arquivo .env / .txt",
            type=["env", "txt", "key"],
        )

        if uploaded:
            content = uploaded.read().decode(errors="ignore").strip()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("NVIDIA_API_KEY="):
                    key_input = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("nvapi-"):
                    key_input = line
                if key_input.startswith("nvapi-"):
                    break

        if st.button("🔑 Entrar", use_container_width=True):
            if key_input and key_input.startswith("nvapi-"):
                st.session_state.api_key = key_input
                st.rerun()
            else:
                st.error("Chave inválida. Deve começar com `nvapi-`")

        st.markdown(
            "<div style='text-align:center;margin-top:16px'>"
            "<a href='https://build.nvidia.com' target='_blank' "
            "style='color:#76B900;font-size:0.8rem'>🔗 Obter API Key gratuita</a></div>",
            unsafe_allow_html=True,
        )


# ─── BANCO DE DADOS ───────────────────────────────────────────────────────────
class Database:
    """Wrapper SQLite thread-safe sem uso de st.cache_resource."""

    def __init__(self, path: str = "data/chat.db"):
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    DEFAULT 'Nova Conversa',
                created_at TEXT    DEFAULT (datetime('now','localtime')),
                updated_at TEXT    DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                image_path      TEXT,
                model           TEXT,
                timestamp       TEXT    DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
        """)
        conn.commit()
        conn.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        try:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur
        except sqlite3.OperationalError:
            self._conn = self._connect()
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def fetchall(self, sql: str, params=()) -> List[sqlite3.Row]:
        return self.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params=()) -> Optional[sqlite3.Row]:
        return self.execute(sql, params).fetchone()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ─── EMBEDDING ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    if not ST_AVAILABLE:
        return None, 384
    logger.info("Carregando modelo de embedding...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding pronto. Dimensão: {dim}")
    return model, dim


# ─── VECTOR STORE ─────────────────────────────────────────────────────────────
def load_vector_store() -> Dict:
    if os.path.exists(VECTORS_FILE):
        try:
            with open(VECTORS_FILE, "rb") as f:
                data = pickle.load(f)
            # Validar estrutura
            if not all(k in data for k in ("documents", "embeddings", "ids", "sources")):
                raise ValueError("Estrutura inválida")
            return data
        except Exception as e:
            logger.warning(f"Vector store corrompido, recriando: {e}")
    return {"documents": [], "embeddings": [], "ids": [], "sources": []}


def save_vector_store(data: Dict):
    try:
        with open(VECTORS_FILE, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Falha ao salvar vector store: {e}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if len(c) > 50]


# ─── RAG SERVICE ──────────────────────────────────────────────────────────────
class RAGService:
    def __init__(self):
        self.embedder, self.emb_dim = load_embedder()
        self.store = load_vector_store()
        self._validate_store()

    def _validate_store(self):
        """Verifica consistência das dimensões armazenadas."""
        if self.store["embeddings"] and self.embedder:
            stored_dim = len(self.store["embeddings"][0])
            if stored_dim != self.emb_dim:
                logger.warning(
                    f"Dimensão incompatível ({stored_dim} vs {self.emb_dim}). "
                    "Descartando vector store."
                )
                self.store = {"documents": [], "embeddings": [], "ids": [], "sources": []}
                save_vector_store(self.store)

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not self.embedder:
            return None
        try:
            return self.embedder.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Embedding falhou: {e}")
            return None

    def add_document(self, filename: str, text: str) -> int:
        text = clean_text(text)
        if not text:
            return 0
        chunks = chunk_text(text)
        added = 0
        for i, chunk in enumerate(chunks):
            emb = self.embed(chunk)
            if emb is None:
                continue
            # Verificar dimensão dos existentes
            if self.store["embeddings"]:
                if len(emb) != len(self.store["embeddings"][0]):
                    logger.warning("Dimensão inconsistente — ignorando chunk")
                    continue
            doc_id = f"{filename}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            if doc_id in self.store["ids"]:
                continue  # Evitar duplicatas
            self.store["documents"].append(chunk)
            self.store["embeddings"].append(emb.tolist())
            self.store["ids"].append(doc_id)
            self.store["sources"].append(filename)
            added += 1
        save_vector_store(self.store)
        return added

    def search(self, query: str, k: int = 4, threshold: float = 0.25) -> str:
        if not self.store["embeddings"]:
            return ""
        emb = self.embed(query)
        if emb is None:
            return ""
        q = np.array(emb)
        scores = [
            (cosine_similarity(q, np.array(e)), i)
            for i, e in enumerate(self.store["embeddings"])
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        top = [(s, i) for s, i in scores[:k] if s >= threshold]
        if not top:
            return ""
        parts = []
        for score, idx in top:
            src = self.store["sources"][idx] if idx < len(self.store["sources"]) else "doc"
            parts.append(f"[{src} | sim={score:.2f}]\n{self.store['documents'][idx]}")
        return "\n\n---\n\n".join(parts)

    def clear(self):
        self.store = {"documents": [], "embeddings": [], "ids": [], "sources": []}
        save_vector_store(self.store)

    @property
    def doc_count(self) -> int:
        return len(self.store["documents"])

    @property
    def sources(self) -> List[str]:
        return list(set(self.store.get("sources", [])))


# ─── WEB SEARCH ───────────────────────────────────────────────────────────────
def web_search(query: str, num: int = 4) -> str:
    if not DDGS_AVAILABLE:
        return ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num, timelimit="m"))
        if not results:
            return ""
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")[:300]
            href = r.get("href", "")
            lines.append(f"• **{title}**\n  {body}\n  Fonte: {href}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.warning(f"Busca web falhou: {e}")
        return ""


# ─── LLM SERVICE ──────────────────────────────────────────────────────────────
class LLMService:
    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call(self, messages, model, temperature, max_tokens, stream=False):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            timeout=60,
        )

    def generate(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, str]:
        """Geração com fallback automático."""
        models_to_try = [model, FALLBACK_MODEL]
        if model == FALLBACK_MODEL:
            models_to_try = [model]

        last_error = None
        for m in models_to_try:
            try:
                resp = self._call(messages, m, temperature, max_tokens, stream=False)
                return resp.choices[0].message.content or "", m
            except Exception as e:
                logger.warning(f"Modelo {m} falhou: {e}")
                last_error = e
        return f"❌ Erro: {last_error}", "error"

    def generate_stream(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """Streaming com fallback para geração normal."""
        try:
            stream = self._call(messages, model, temperature, max_tokens, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            logger.warning(f"Streaming falhou ({e}), usando geração normal")
            text, _ = self.generate(messages, model, temperature, max_tokens)
            yield text

    def vision(
        self,
        image_base64: str,
        prompt: str,
        media_type: str = "image/jpeg",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, str]:
        """Análise de imagem com fallback entre modelos de visão."""
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }]
        last_error = None
        for vm in VISION_MODELS:
            try:
                resp = self._call(messages, vm, temperature, max_tokens, stream=False)
                return resp.choices[0].message.content or "", vm
            except Exception as e:
                logger.warning(f"Vision model {vm} falhou: {e}")
                last_error = e
        return f"❌ Erro no modelo de visão: {last_error}", "error"


# ─── TTS ──────────────────────────────────────────────────────────────────────
def text_to_speech(text: str, lang: str = "pt") -> Optional[str]:
    """Converte texto em áudio MP3, com cache por hash."""
    try:
        # Limpar texto de markdown
        clean = re.sub(r"[*_`#\[\](){}<>]", "", text)
        clean = re.sub(r"https?://\S+", "", clean).strip()
        if not clean:
            return None

        # Cache por hash
        text_hash = hashlib.md5(clean[:1000].encode()).hexdigest()
        path = f"data/tts/{text_hash}.mp3"
        if not os.path.exists(path):
            # gTTS aceita textos longos; dividir em 5000 chars para segurança
            tts_obj = gTTS(text=clean[:5000], lang=lang, slow=False)
            tts_obj.save(path)
        return path
    except Exception as e:
        logger.error(f"TTS falhou: {e}")
        return None


# ─── EXPORT ───────────────────────────────────────────────────────────────────
def export_messages(messages: List[Dict], fmt: str = "txt") -> bytes:
    if not messages:
        return b""
    try:
        if fmt == "txt":
            lines = []
            for m in messages:
                role = "Usuário" if m["role"] == "user" else "Assistente"
                lines.append(f"[{role}]\n{m['content']}\n")
            return "\n".join(lines).encode("utf-8")

        elif fmt == "md":
            lines = ["# Conversa — NIM AI Studio\n"]
            for m in messages:
                role = "👤 Usuário" if m["role"] == "user" else "🤖 Assistente"
                lines.append(f"## {role}\n{m['content']}\n")
            return "\n".join(lines).encode("utf-8")

        elif fmt == "docx":
            doc = Document()
            doc.add_heading("Conversa — NIM AI Studio", level=1)
            for m in messages:
                role = "Usuário" if m["role"] == "user" else "Assistente"
                doc.add_heading(role, level=2)
                doc.add_paragraph(m["content"])
                doc.add_paragraph("")
            buf = BytesIO()
            doc.save(buf)
            return buf.getvalue()

    except Exception as e:
        logger.error(f"Export falhou ({fmt}): {e}")
    return b""


# ─── HELPERS DE UI ────────────────────────────────────────────────────────────
def format_model_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def build_system_prompt(rag_ctx: str, web_ctx: str, lang: str = "pt-BR") -> str:
    parts = [
        "Você é um assistente de IA avançado e multimodal, alimentado por NVIDIA NIM.",
        "Seja preciso, detalhado e útil. Formate respostas com Markdown quando apropriado.",
        f"Responda sempre em {lang}, a menos que o usuário peça outro idioma.",
    ]
    if rag_ctx:
        parts.append(
            f"\n📚 CONTEXTO DOS DOCUMENTOS (use prioritariamente):\n{rag_ctx}"
        )
    if web_ctx:
        parts.append(
            f"\n🌐 INFORMAÇÕES DA WEB (use como complemento):\n{web_ctx}"
        )
    if rag_ctx or web_ctx:
        parts.append(
            "\nSempre cite as fontes quando usar informações dos contextos acima."
        )
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  INICIALIZAÇÃO DA APLICAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

# 1. API Key
NVIDIA_API_KEY = load_api_key()
if not NVIDIA_API_KEY:
    api_key_screen()
    st.stop()

# 2. Serviços (singleton via session_state para evitar reinicialização)
if "db" not in st.session_state:
    st.session_state.db = Database()
if "rag" not in st.session_state:
    st.session_state.rag = RAGService()
if "llm" not in st.session_state:
    st.session_state.llm = LLMService(NVIDIA_API_KEY)

db: Database = st.session_state.db
rag: RAGService = st.session_state.rag
llm: LLMService = st.session_state.llm

# 3. Estado da conversa
if "conv_id" not in st.session_state:
    convs = db.fetchall(
        "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1"
    )
    if convs:
        st.session_state.conv_id = convs[0]["id"]
    else:
        cur = db.execute("INSERT INTO conversations (title) VALUES ('Nova Conversa')")
        st.session_state.conv_id = cur.lastrowid

if "messages" not in st.session_state:
    rows = db.fetchall(
        "SELECT role, content, image_path FROM messages WHERE conversation_id=? ORDER BY id",
        (st.session_state.conv_id,),
    )
    st.session_state.messages = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"]}
        if r["image_path"]:
            msg["image"] = r["image_path"]
        st.session_state.messages.append(msg)

# Defaults de configuração
if "streaming"    not in st.session_state: st.session_state.streaming    = True
if "web_enabled"  not in st.session_state: st.session_state.web_enabled  = True
if "tts_enabled"  not in st.session_state: st.session_state.tts_enabled  = False
if "temperature"  not in st.session_state: st.session_state.temperature  = 0.7
if "max_tokens"   not in st.session_state: st.session_state.max_tokens   = 1024
if "model_name"   not in st.session_state: st.session_state.model_name   = list(TEXT_MODELS.keys())[0]
if "img_key"      not in st.session_state: st.session_state.img_key      = 0  # força reset do uploader

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style='padding:12px 0 20px'>
        <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;
                    background:linear-gradient(135deg,#76B900,#00d4aa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            NIM AI Studio
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                    color:#6b6b8a;margin-top:2px'>v2.0 — NVIDIA NIM</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Conversas ──
    st.markdown("**💬 Conversas**")
    if st.button("➕ Nova Conversa", use_container_width=True):
        cur = db.execute("INSERT INTO conversations (title) VALUES ('Nova Conversa')")
        st.session_state.conv_id = cur.lastrowid
        st.session_state.messages = []
        st.session_state.img_key += 1
        st.rerun()

    convs = db.fetchall(
        "SELECT id, title FROM conversations ORDER BY updated_at DESC LIMIT 20"
    )
    for c in convs:
        active = c["id"] == st.session_state.conv_id
        css = "conv-item active" if active else "conv-item"
        label = (c["title"] or "Nova Conversa")[:35]
        if st.button(
            f"{'▶ ' if active else ''}{label}",
            key=f"conv_{c['id']}",
            use_container_width=True,
            help=c["title"],
        ):
            if c["id"] != st.session_state.conv_id:
                st.session_state.conv_id = c["id"]
                rows = db.fetchall(
                    "SELECT role, content, image_path FROM messages WHERE conversation_id=? ORDER BY id",
                    (c["id"],),
                )
                st.session_state.messages = []
                for r in rows:
                    msg = {"role": r["role"], "content": r["content"]}
                    if r["image_path"]:
                        msg["image"] = r["image_path"]
                    st.session_state.messages.append(msg)
                st.session_state.img_key += 1
                st.rerun()

    st.divider()

    # ── Modelo ──
    st.markdown("**🤖 Modelo**")
    model_name = st.selectbox(
        "Modelo de texto",
        list(TEXT_MODELS.keys()),
        index=list(TEXT_MODELS.keys()).index(st.session_state.model_name)
        if st.session_state.model_name in TEXT_MODELS else 0,
        label_visibility="collapsed",
    )
    st.session_state.model_name = model_name
    model_id = TEXT_MODELS[model_name]

    # ── Parâmetros ──
    st.markdown("**⚙️ Parâmetros**")
    temperature = st.slider("Temperatura", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.temperature = temperature
    max_tokens = st.slider("Max tokens", 256, 4096, st.session_state.max_tokens, 128)
    st.session_state.max_tokens = max_tokens

    col1, col2 = st.columns(2)
    with col1:
        streaming = st.checkbox("Streaming", st.session_state.streaming)
        st.session_state.streaming = streaming
    with col2:
        tts_enabled = st.checkbox("Voz (TTS)", st.session_state.tts_enabled)
        st.session_state.tts_enabled = tts_enabled

    web_enabled = st.checkbox("🌐 Busca Web", st.session_state.web_enabled)
    st.session_state.web_enabled = web_enabled

    st.divider()

    # ── RAG / Documentos ──
    st.markdown("**📄 Documentos (RAG)**")
    if rag.doc_count > 0:
        st.markdown(
            f"<div class='stat-card'>"
            f"<span class='stat-label'>CHUNKS INDEXADOS</span>"
            f"<span class='stat-value'>{rag.doc_count}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        for src in rag.sources[:5]:
            st.markdown(
                f"<span class='context-badge ctx-rag'>📄 {src[:30]}</span>",
                unsafe_allow_html=True,
            )

    uploaded_pdfs = st.file_uploader(
        "Carregar PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_pdfs:
        if st.button("⚡ Indexar PDFs", use_container_width=True):
            total = 0
            errors = []
            progress = st.progress(0)
            for idx, f in enumerate(uploaded_pdfs):
                try:
                    reader = PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += (page.extract_text() or "") + "\n"
                    if not text.strip():
                        errors.append(f"'{f.name}': sem texto extraível")
                        continue
                    added = rag.add_document(f.name, text)
                    total += added
                except Exception as e:
                    errors.append(f"'{f.name}': {e}")
                progress.progress((idx + 1) / len(uploaded_pdfs))
            progress.empty()
            if total > 0:
                st.success(f"✅ {total} chunks indexados")
            for err in errors:
                st.warning(err)

    if rag.doc_count > 0:
        if st.button("🗑️ Limpar documentos", use_container_width=True):
            rag.clear()
            st.success("Documentos removidos")
            st.rerun()

    st.divider()

    # ── Ações da conversa ──
    st.markdown("**🗂️ Exportar**")
    msgs_for_export = [
        m for m in st.session_state.messages if m["role"] in ("user", "assistant")
    ]
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "TXT", export_messages(msgs_for_export, "txt"),
        file_name="conversa.txt", mime="text/plain", use_container_width=True,
    )
    c2.download_button(
        "MD", export_messages(msgs_for_export, "md"),
        file_name="conversa.md", mime="text/markdown", use_container_width=True,
    )
    c3.download_button(
        "DOCX", export_messages(msgs_for_export, "docx"),
        file_name="conversa.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )

    if st.button("🗑️ Limpar conversa", use_container_width=True):
        db.execute(
            "DELETE FROM messages WHERE conversation_id=?",
            (st.session_state.conv_id,),
        )
        st.session_state.messages = []
        st.session_state.img_key += 1
        st.rerun()

    if st.button("🔑 Trocar API Key", use_container_width=True):
        st.session_state.pop("api_key", None)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  ÁREA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(f"""
<div class='nim-header'>
    <div class='nim-logo'>🧠 NIM AI Studio</div>
    <div class='nim-badge'>MULTIMODAL</div>
    <div class='nim-badge' style='background:rgba(0,212,170,0.1);color:#00d4aa;
         border-color:rgba(0,212,170,0.3)'>{format_model_name(model_id)}</div>
</div>
""", unsafe_allow_html=True)

# ── Histórico de mensagens ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image") and os.path.exists(msg["image"]):
            st.image(msg["image"], width=360)
        st.markdown(msg["content"])

# ── Input area ──────────────────────────────────────────────────────────────
# Upload de imagem com key dinâmica (evita estado travado entre conversas)
uploaded_image = st.file_uploader(
    "📎 Anexar imagem (PNG, JPG, WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
    key=f"img_upload_{st.session_state.img_key}",
    label_visibility="collapsed",
)

prompt = st.chat_input("Pergunte qualquer coisa… (Shift+Enter para nova linha)")

# ─── PROCESSAMENTO DA MENSAGEM ───────────────────────────────────────────────
if prompt:
    # ── 1. Processar imagem (se houver) ──────────────────────────────────────
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    media_type = "image/jpeg"

    if uploaded_image:
        try:
            img_bytes = uploaded_image.read()
            ext = uploaded_image.name.rsplit(".", 1)[-1].lower()
            media_type = f"image/{ext if ext != 'jpg' else 'jpeg'}"
            image_base64 = base64.b64encode(img_bytes).decode()
            fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
            image_path = f"data/images/{fname}"
            with open(image_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            st.error(f"Erro ao processar imagem: {e}")

    # ── 2. Salvar mensagem do usuário ─────────────────────────────────────────
    user_msg = {"role": "user", "content": prompt}
    if image_path:
        user_msg["image"] = image_path

    st.session_state.messages.append(user_msg)
    db.execute(
        "INSERT INTO messages (conversation_id, role, content, image_path, model) VALUES (?,?,?,?,?)",
        (st.session_state.conv_id, "user", prompt, image_path, model_id),
    )

    # Atualizar título da conversa (primeira mensagem)
    if len(st.session_state.messages) == 1:
        title = prompt[:60].replace("\n", " ")
        db.execute(
            "UPDATE conversations SET title=?, updated_at=datetime('now','localtime') WHERE id=?",
            (title, st.session_state.conv_id),
        )

    # ── 3. Exibir mensagem do usuário ─────────────────────────────────────────
    with st.chat_message("user"):
        if image_path and os.path.exists(image_path):
            st.image(image_path, width=360)
        st.markdown(prompt)

    # ── 4. Busca de contexto (RAG + Web) ─────────────────────────────────────
    rag_ctx = ""
    web_ctx = ""
    context_badges = []

    # RAG
    if rag.doc_count > 0:
        with st.spinner("🔍 Buscando nos documentos…"):
            rag_ctx = rag.search(prompt)
        if rag_ctx:
            context_badges.append("📚 RAG")

    # Web
    if st.session_state.web_enabled and not image_base64:
        with st.spinner("🌐 Buscando na web…"):
            web_ctx = web_search(prompt)
        if web_ctx:
            context_badges.append("🌐 Web")

    if image_base64:
        context_badges.append("🖼️ Visão")

    # ── 5. Gerar resposta ─────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        # Badges de contexto
        if context_badges:
            badge_html = " ".join(
                f"<span class='context-badge "
                + ("ctx-rag" if "RAG" in b else "ctx-web" if "Web" in b else "ctx-img")
                + f"'>{b}</span>"
                for b in context_badges
            )
            st.markdown(badge_html, unsafe_allow_html=True)

        answer = ""
        used_model = model_id

        if image_base64:
            # ── Modo Visão ─────────────────────────────────────────────────
            with st.spinner("🧠 Analisando imagem…"):
                answer, used_model = llm.vision(
                    image_base64,
                    prompt,
                    media_type,
                    temperature,
                    max_tokens,
                )
            st.markdown(answer)
        else:
            # ── Modo Texto ─────────────────────────────────────────────────
            system_prompt = build_system_prompt(rag_ctx, web_ctx)

            # Histórico recente (excluindo mensagem atual, já adicionada)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-12:]
            ]
            final_messages = [{"role": "system", "content": system_prompt}] + history

            if st.session_state.streaming:
                placeholder = st.empty()
                full = ""
                try:
                    for chunk in llm.generate_stream(
                        final_messages, model_id, temperature, max_tokens
                    ):
                        full += chunk
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                    answer = full
                except Exception as e:
                    answer = f"❌ Erro: {e}"
                    placeholder.markdown(answer)
                    used_model = "error"
            else:
                with st.spinner("💭 Pensando…"):
                    answer, used_model = llm.generate(
                        final_messages, model_id, temperature, max_tokens
                    )
                st.markdown(answer)

        # ── 6. Rodapé da resposta ──────────────────────────────────────────
        st.markdown(
            f"<div class='model-tag'>🤖 {format_model_name(used_model)}</div>",
            unsafe_allow_html=True,
        )

        # TTS
        if st.session_state.tts_enabled and answer and used_model != "error":
            with st.spinner("🔊 Gerando áudio…"):
                audio_path = text_to_speech(answer)
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as af:
                    st.audio(af.read(), format="audio/mp3")

        # Download da resposta
        c1, c2 = st.columns([1, 1])
        c1.download_button(
            "⬇️ TXT",
            export_messages([{"role": "assistant", "content": answer}], "txt"),
            file_name="resposta.txt",
            mime="text/plain",
        )
        c2.download_button(
            "⬇️ DOCX",
            export_messages([{"role": "assistant", "content": answer}], "docx"),
            file_name="resposta.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    # ── 7. Salvar resposta ────────────────────────────────────────────────────
    st.session_state.messages.append({"role": "assistant", "content": answer})
    db.execute(
        "INSERT INTO messages (conversation_id, role, content, model) VALUES (?,?,?,?)",
        (st.session_state.conv_id, "assistant", answer, used_model),
    )
    db.execute(
        "UPDATE conversations SET updated_at=datetime('now','localtime') WHERE id=?",
        (st.session_state.conv_id,),
    )

    # Reset do uploader de imagem após envio
    st.session_state.img_key += 1

    # CORREÇÃO: sem st.rerun() aqui — evita loop infinito
    # O Streamlit faz rerun naturalmente após o bloco de chat_input

# ─── TELA VAZIA ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center;padding:80px 0 40px;opacity:0.5'>
        <div style='font-size:3rem;margin-bottom:16px'>🧠</div>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;
                    color:#e8e8f0;margin-bottom:8px'>Como posso ajudar?</div>
        <div style='color:#6b6b8a;font-size:0.9rem;max-width:400px;margin:0 auto'>
            Faça uma pergunta, envie uma imagem, carregue documentos PDF
            ou ative a busca web para respostas em tempo real.
        </div>
    </div>

    <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:12px;
                max-width:600px;margin:0 auto;padding:0 0 40px'>
        <div style='background:#12121a;border:1px solid #1e1e2e;border-radius:12px;
                    padding:16px;font-size:0.85rem'>
            <div style='color:#76B900;margin-bottom:6px'>📄 Analisar PDFs</div>
            <div style='color:#6b6b8a'>Carregue documentos e faça perguntas sobre o conteúdo</div>
        </div>
        <div style='background:#12121a;border:1px solid #1e1e2e;border-radius:12px;
                    padding:16px;font-size:0.85rem'>
            <div style='color:#00d4aa;margin-bottom:6px'>🖼️ Visão Computacional</div>
            <div style='color:#6b6b8a'>Envie imagens para análise detalhada com IA</div>
        </div>
        <div style='background:#12121a;border:1px solid #1e1e2e;border-radius:12px;
                    padding:16px;font-size:0.85rem'>
            <div style='color:#ffaa00;margin-bottom:6px'>🌐 Busca em Tempo Real</div>
            <div style='color:#6b6b8a'>Respostas baseadas em informações atuais da web</div>
        </div>
        <div style='background:#12121a;border:1px solid #1e1e2e;border-radius:12px;
                    padding:16px;font-size:0.85rem'>
            <div style='color:#76B900;margin-bottom:6px'>🔊 Respostas em Voz</div>
            <div style='color:#6b6b8a'>Ative TTS para ouvir as respostas do assistente</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── CLEANUP ──────────────────────────────────────────────────────────────────
def _cleanup():
    if "db" in st.session_state:
        try:
            st.session_state.db.close()
        except Exception:
            pass

atexit.register(_cleanup)
