import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from ddgs import DDGS
from gtts import gTTS
from docx import Document
import sqlite3
import os
import base64
import time
import logging
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import Optional
import atexit
import hashlib
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CARREGAMENTO SEGURO DA CHAVE NVIDIA
# ------------------------------------------------------------
def load_api_key():
    api_key = None
    try:
        if hasattr(st, 'secrets') and 'NVIDIA_API_KEY' in st.secrets:
            api_key = st.secrets['NVIDIA_API_KEY']
    except:
        pass
    if not api_key:
        if 'api_key_value' not in st.session_state:
            st.session_state.api_key_value = None
        if st.session_state.api_key_value:
            api_key = st.session_state.api_key_value
        else:
            st.markdown("## 🔑 Configuração da API Key")
            uploaded = st.file_uploader("Arquivo .env ou .txt com a chave", type=['env','txt'])
            if uploaded:
                content = uploaded.read().decode().strip()
                if '=' in content:
                    for line in content.split('\n'):
                        if line.startswith('NVIDIA_API_KEY='):
                            key = line.split('=',1)[1].strip().strip('"').strip("'")
                            if key.startswith('nvapi-'):
                                st.session_state.api_key_value = key
                                st.success("✅ Chave carregada!")
                                st.rerun()
                elif content.startswith('nvapi-'):
                    st.session_state.api_key_value = content
                    st.success("✅ Chave carregada!")
                    st.rerun()
                else:
                    st.error("Formato inválido")
            manual_key = st.text_input("Ou cole a chave", type="password")
            if manual_key and manual_key.startswith('nvapi-'):
                st.session_state.api_key_value = manual_key
                st.success("✅ Chave aceita!")
                st.rerun()
            st.stop()
    return api_key

NVIDIA_API_KEY = load_api_key()
BASE_URL = "https://integrate.api.nvidia.com/v1"
client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)

# ------------------------------------------------------------
# MODELOS (validados e funcionais)
# ------------------------------------------------------------
TEXT_MODELS = {
    "Llama 3.1 70B": "meta/llama-3.1-70b-instruct",
    "Llama 3.1 8B": "meta/llama-3.1-8b-instruct",
    "Nemotron 70B": "nvidia/nemotron-70b-instruct",
}
VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"

# ------------------------------------------------------------
# BANCO DE DADOS (com cache)
# ------------------------------------------------------------
@st.cache_resource
def get_db_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/chat.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT DEFAULT 'Nova Conversa',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        role TEXT,
        content TEXT,
        image_path TEXT,
        model TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    )""")
    conn.commit()
    return conn

conn = get_db_connection()

# ------------------------------------------------------------
# MODELO DE EMBEDDING LOCAL (substituto da API)
# ------------------------------------------------------------
@st.cache_resource
def load_local_embedder():
    """Carrega modelo de embedding leve e offline."""
    logger.info("Carregando modelo de embedding local...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Modelo de embedding carregado com sucesso.")
    return model

embedder = load_local_embedder()

# ------------------------------------------------------------
# ARMAZENAMENTO VETORIAL LOCAL
# ------------------------------------------------------------
VECTORS_FILE = "data/vectors.pkl"

def load_vectors():
    if os.path.exists(VECTORS_FILE):
        with open(VECTORS_FILE, "rb") as f:
            return pickle.load(f)
    return {"documents": [], "embeddings": [], "ids": []}

def save_vectors(data):
    with open(VECTORS_FILE, "wb") as f:
        pickle.dump(data, f)

vector_store = load_vectors()

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------
def clean_text(text):
    import re
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, size=1000, overlap=200):
    return [text[i:i+size] for i in range(0, len(text), size-overlap)]

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# ------------------------------------------------------------
# RAG SERVICE (agora com embedding LOCAL)
# ------------------------------------------------------------
class RAGService:
    def __init__(self):
        self.store = vector_store
        self.model = embedder

    def embed(self, text):
        """Gera embedding usando modelo SentenceTransformer local."""
        try:
            emb = self.model.encode(text)
            return emb
        except Exception as e:
            logger.error(f"Falha no embedding local: {e}")
            return None

    def search(self, query, k=3):
        if not self.store["embeddings"]:
            logger.warning("Nenhum documento indexado.")
            return ""
        emb = self.embed(query)
        if emb is None:
            return ""
        scores = []
        for i, doc_emb in enumerate(self.store["embeddings"]):
            sim = cosine_similarity(emb, np.array(doc_emb))
            scores.append((sim, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        top_docs = [self.store["documents"][i] for _, i in scores[:k]]
        logger.info(f"RAG retornou {len(top_docs)} documentos.")
        return "\n\n---\n\n".join(top_docs)

    def add_document(self, filename, text):
        text = clean_text(text)
        if not text:
            st.warning(f"'{filename}' não contém texto.")
            return 0
        chunks = chunk_text(text)
        count = 0
        for i, chunk in enumerate(chunks):
            emb = self.embed(chunk)
            if emb is not None:
                self.store["documents"].append(chunk)
                self.store["embeddings"].append(emb.tolist())
                self.store["ids"].append(f"{filename}_{i}")
                count += 1
        save_vectors(self.store)
        logger.info(f"'{filename}' indexado com {count} chunks.")
        return count

rag = RAGService()

# ------------------------------------------------------------
# BUSCA WEB
# ------------------------------------------------------------
def web_search(query, num=3):
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=num):
                results.append(f"- {r['title']}: {r['body']}")
            return "\n".join(results) if results else ""
    except Exception as e:
        logger.error(f"Busca web falhou: {e}")
        return ""

# ------------------------------------------------------------
# LLM SERVICE
# ------------------------------------------------------------
class LLMService:
    def __init__(self):
        self.client = client

    def generate(self, messages, model, temperature=0.7, max_tokens=1024):
        for m in [model, "meta/llama-3.1-8b-instruct"]:
            try:
                resp = self.client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30
                )
                return resp.choices[0].message.content, m
            except Exception as e:
                logger.warning(f"Modelo {m} falhou: {e}")
        return "❌ Nenhum modelo disponível.", "error"

    def generate_stream(self, messages, model, temperature=0.7, max_tokens=1024):
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=30
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.warning(f"Streaming falhou, fallback sem stream: {e}")
            try:
                answer, _ = self.generate(messages, model, temperature, max_tokens)
                yield answer
            except:
                yield f"❌ Erro: {e}"

    def vision(self, image_base64, prompt, temperature=0.7, max_tokens=1024):
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": prompt}
        ]}]
        try:
            resp = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            return resp.choices[0].message.content, VISION_MODEL
        except Exception as e:
            logger.error(f"Visão falhou: {e}")
            raise

llm = LLMService()

# ------------------------------------------------------------
# TTS (sem warnings)
# ------------------------------------------------------------
def tts(text):
    try:
        os.makedirs("data/tts", exist_ok=True)
        text_hash = hashlib.md5(text[:500].encode()).hexdigest()
        path = f"data/tts/{text_hash}.mp3"
        if not os.path.exists(path):
            tts_obj = gTTS(text[:500], lang='pt')
            tts_obj.save(path)
        return path
    except Exception as e:
        logger.error(f"TTS falhou: {e}")
        return None

# ------------------------------------------------------------
# EXPORTAÇÃO
# ------------------------------------------------------------
def export(messages, fmt='txt'):
    if fmt == 'txt':
        return "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]).encode()
    elif fmt == 'docx':
        doc = Document()
        doc.add_heading("Conversa", level=1)
        for m in messages:
            doc.add_heading(m['role'].upper(), level=2)
            doc.add_paragraph(m['content'])
        buf = BytesIO()
        doc.save(buf)
        return buf.getvalue()
    elif fmt == 'md':
        return "\n\n".join([f"## {m['role'].upper()}\n{m['content']}" for m in messages]).encode()
    return b""

# ------------------------------------------------------------
# INTERFACE STREAMLIT
# ------------------------------------------------------------
st.set_page_config(page_title="Assistente NVIDIA", layout="wide")
st.markdown("<h1 style='color:#76B900'>🤖 Assistente NVIDIA NIM</h1>", unsafe_allow_html=True)

if "conv_id" not in st.session_state:
    cur = conn.cursor()
    convs = cur.execute("SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1").fetchall()
    if convs:
        st.session_state.conv_id = convs[0]["id"]
        msgs = cur.execute("SELECT role, content, image_path FROM messages WHERE conversation_id=? ORDER BY id", (st.session_state.conv_id,)).fetchall()
        st.session_state.messages = []
        for m in msgs:
            msg = {"role": m["role"], "content": m["content"]}
            if m["image_path"]:
                msg["image"] = m["image_path"]
            st.session_state.messages.append(msg)
    else:
        cur.execute("INSERT INTO conversations (title) VALUES ('Nova conversa')")
        conn.commit()
        st.session_state.conv_id = cur.lastrowid
        st.session_state.messages = []

if "streaming" not in st.session_state:
    st.session_state.streaming = True

with st.sidebar:
    st.header("⚙️ Configurações")
    model_name = st.selectbox("Modelo de texto", list(TEXT_MODELS.keys()), index=0)
    model_id = TEXT_MODELS[model_name]
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max tokens", 256, 2048, 1024)
    st.session_state.streaming = st.checkbox("Streaming", True)
    web_enabled = st.checkbox("Busca Web (DDGS)", True)
    st.divider()
    st.header("📄 PDFs (RAG)")
    uploaded = st.file_uploader("Carregar PDFs", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Processar PDFs"):
        total = 0
        for f in uploaded:
            path = f"data/{f.name}"
            with open(path, "wb") as fp:
                fp.write(f.getbuffer())
            reader = PdfReader(path)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            if not text.strip():
                st.warning(f"'{f.name}' não possui texto extraível.")
                continue
            added = rag.add_document(f.name, text)
            total += added
        if total > 0:
            st.success(f"✅ {total} chunks indexados com sucesso!")
        else:
            st.error("Nenhum texto foi indexado.")
    st.divider()
    if st.button("🗑️ Limpar conversa"):
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (st.session_state.conv_id,))
        conn.commit()
        st.session_state.messages = []
        st.rerun()

# Exibir histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image" in msg and msg["image"]:
            st.image(msg["image"], width=300)
        st.markdown(msg["content"])

# Input
uploaded_image = st.file_uploader("📎 Imagem (opcional)", type=["png","jpg","jpeg"], key="img")
prompt = st.chat_input("Digite sua mensagem...")

if prompt:
    image_path = None
    image_base64 = None
    if uploaded_image:
        os.makedirs("data/images", exist_ok=True)
        img_bytes = uploaded_image.read()
        image_base64 = base64.b64encode(img_bytes).decode()
        ext = uploaded_image.name.split('.')[-1]
        image_path = f"data/images/{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"
        with open(image_path, "wb") as f:
            f.write(img_bytes)

    user_msg = {"role": "user", "content": prompt}
    if image_path:
        user_msg["image"] = image_path
    st.session_state.messages.append(user_msg)
    conn.execute("INSERT INTO messages (conversation_id, role, content, image_path, model) VALUES (?,?,?,?,?)",
                 (st.session_state.conv_id, "user", prompt, image_path, model_id))
    conn.commit()

    if len(st.session_state.messages) == 1:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (prompt[:50], st.session_state.conv_id))
        conn.commit()

    with st.chat_message("user"):
        if image_path:
            st.image(image_path, width=300)
        st.markdown(prompt)

    # RAG local (sem API)
    rag_text = ""
    if vector_store["embeddings"]:
        with st.spinner("🔍 Buscando nos documentos..."):
            rag_text = rag.search(prompt)
        if rag_text:
            st.info("📚 Contexto RAG carregado.")
        else:
            st.warning("Nenhum trecho relevante encontrado nos PDFs.")
    else:
        st.info("ℹ️ Nenhum PDF foi processado ainda. Use a barra lateral para carregar documentos.")

    web_text = web_search(prompt) if web_enabled else ""

    with st.chat_message("assistant"):
        if image_base64:
            with st.spinner("🧠 Analisando imagem..."):
                try:
                    answer, used_model = llm.vision(image_base64, prompt, temperature, max_tokens)
                    st.markdown(answer)
                except Exception as e:
                    answer = f"❌ Erro no modelo de visão: {e}"
                    st.error(answer)
                    used_model = "error"
        else:
            system = f"""Você é um assistente de IA. Utilize EXCLUSIVAMENTE as informações fornecidas nos documentos PDF abaixo para responder à pergunta. Se os documentos não contiverem a resposta, diga 'Não encontrei essa informação nos documentos fornecidos'. 

📚 DOCUMENTOS PDF (use este conteúdo para responder):
{rag_text if rag_text else "Nenhum documento foi carregado."}

🌐 INFORMAÇÕES DA WEB (use apenas se os documentos não responderem):
{web_text if web_text else "Nenhuma."}

Responda em português, de forma clara e objetiva, citando trechos dos documentos quando possível."""

            recent = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
            final_msgs = [{"role": "system", "content": system}] + recent

            if st.session_state.streaming:
                placeholder = st.empty()
                full = ""
                used_model = model_id
                try:
                    for chunk in llm.generate_stream(final_msgs, model_id, temperature, max_tokens):
                        full += chunk
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                    answer = full
                except Exception as e:
                    answer = f"❌ Erro: {e}"
                    placeholder.markdown(answer)
                    used_model = "error"
            else:
                with st.spinner("Pensando..."):
                    answer, used_model = llm.generate(final_msgs, model_id, temperature, max_tokens)
                st.markdown(answer)

        st.caption(f"🤖 {used_model}")

        audio_path = tts(answer)
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")

        c1, c2, c3 = st.columns(3)
        c1.download_button("📄 TXT", export([{"role":"assistant","content":answer}], 'txt'), file_name="resposta.txt")
        c2.download_button("📑 DOCX", export([{"role":"assistant","content":answer}], 'docx'), file_name="resposta.docx")
        c3.download_button("📝 Conversa", export(st.session_state.messages, 'md'), file_name="conversa.md")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    conn.execute("INSERT INTO messages (conversation_id, role, content, model) VALUES (?,?,?,?)",
                 (st.session_state.conv_id, "assistant", answer, used_model))
    conn.commit()
    st.rerun()

# ------------------------------------------------------------
# LIMPEZA
# ------------------------------------------------------------
def cleanup():
    try:
        conn.close()
    except:
        pass

atexit.register(cleanup)
