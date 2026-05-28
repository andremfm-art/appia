import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from duckduckgo_search import DDGS
from gtts import gTTS
from docx import Document
import tempfile
import sqlite3
import os
import requests
import base64
import time
import logging
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import atexit
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CARREGAMENTO SEGURO DA CHAVE
# ============================================================
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
            uploaded_file = st.file_uploader("Arquivo .env ou .txt com a chave", type=['env','txt'])
            if uploaded_file:
                content = uploaded_file.read().decode().strip()
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

# ============================================================
# MODELOS DISPONÍVEIS (incluindo visão)
# ============================================================
TEXT_MODELS = {
    "Llama 3.1 70B (recomendado)": "meta/llama-3.1-70b-instruct",
    "Llama 3.1 8B (rápido)": "meta/llama-3.1-8b-instruct",
    "DeepSeek-R1": "deepseek-ai/deepseek-r1-distill-llama-8b",
    "Nemotron 70B": "nvidia/nemotron-70b-instruct",
    "Mistral Large 2": "mistralai/mistral-large-2-instruct",
}
VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"  # Modelo multimodal

EMBEDDING_MODELS = {
    "NV-Embed-QA v5": "nvidia/nv-embedqa-e5-v5",
    "NV-EmbedQA 1B v2": "nvidia/llama-3.2-nv-embedqa-1b-v2",
}

# Configurações
MAX_HISTORY = 10
REQUEST_TIMEOUT = 30

# ============================================================
# BANCO DE DADOS E CHROMADB
# ============================================================
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

chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")

# ============================================================
# SERVIÇOS
# ============================================================
def clean_text(text):
    import re
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, size=1000, overlap=200):
    return [text[i:i+size] for i in range(0, len(text), size-overlap)]

class LLMService:
    def __init__(self):
        self.client = client
        self.fallback_chain = list(TEXT_MODELS.values())
    
    def text_generate(self, messages, model, temperature=0.7, max_tokens=1024):
        # Fallback entre modelos de texto
        for m in [model] + [x for x in self.fallback_chain if x != model]:
            try:
                resp = self.client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=REQUEST_TIMEOUT
                )
                return resp.choices[0].message.content, m
            except Exception as e:
                logger.warning(f"Falha com {m}: {e}")
                time.sleep(0.5)
        raise Exception("Nenhum modelo de texto disponível")
    
    def vision_generate(self, image_base64: str, prompt: str, temperature=0.7, max_tokens=1024):
        """Usa o modelo de visão para analisar imagem + texto."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        try:
            resp = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=REQUEST_TIMEOUT
            )
            return resp.choices[0].message.content, VISION_MODEL
        except Exception as e:
            # Fallback: tenta descrever a imagem com outro modelo? Ou retorna erro.
            raise Exception(f"Modelo de visão falhou: {e}")
    
    def generate_stream(self, messages, model, temperature=0.7, max_tokens=1024):
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=REQUEST_TIMEOUT
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

llm = LLMService()

class RAGService:
    def __init__(self):
        self.client = client
        self.collection = collection
    
    def embed(self, text, model="nvidia/nv-embedqa-e5-v5"):
        resp = self.client.embeddings.create(input=[text], model=model, encoding_format="float", timeout=REQUEST_TIMEOUT)
        return resp.data[0].embedding
    
    def search(self, query, model="nvidia/nv-embedqa-e5-v5", k=3):
        if not query:
            return ""
        emb = self.embed(query, model)
        results = self.collection.query(query_embeddings=[emb], n_results=k)
        if results['documents']:
            return "\n\n".join(results['documents'][0])
        return ""
    
    def add_document(self, filename, text, model="nvidia/nv-embedqa-e5-v5"):
        text = clean_text(text)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            emb = self.embed(chunk, model)
            self.collection.add(documents=[chunk], embeddings=[emb], ids=[f"{filename}_{i}"])
        return len(chunks)

rag = RAGService()

def web_search(query, num=3):
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=num):
                results.append(f"- {r['title']}: {r['body']}")
            return "\n".join(results) if results else ""
    except:
        return ""

def tts(text, lang='pt'):
    try:
        tts_obj = gTTS(text[:500], lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts_obj.save(tmp.name)
        return tmp.name
    except:
        return None

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

# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.set_page_config(page_title="Assistente Multimodal NVIDIA", layout="wide")
st.markdown("<h1 style='color:#76B900'>🤖 Assistente Multimodal NVIDIA</h1>", unsafe_allow_html=True)
st.caption("Envie imagens e texto, como no ChatGPT, Claude ou DeepSeek")

# Inicialização de estado
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

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    model_name = st.selectbox("Modelo de texto", list(TEXT_MODELS.keys()), index=0)
    model_id = TEXT_MODELS[model_name]
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max tokens", 256, 2048, 1024)
    st.session_state.streaming = st.checkbox("Streaming", True)
    web_enabled = st.checkbox("Busca Web", True)
    st.divider()
    st.header("📄 PDFs (RAG)")
    uploaded = st.file_uploader("Carregar PDFs", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Processar"):
        total = 0
        for f in uploaded:
            path = f"data/{f.name}"
            with open(path, "wb") as fp:
                fp.write(f.getbuffer())
            reader = PdfReader(path)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            total += rag.add_document(f.name, text)
        st.success(f"{total} chunks processados")
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

# Input multimodal
uploaded_image = st.file_uploader("📎 Anexar imagem (opcional)", type=["png", "jpg", "jpeg"], key="image_upload")
prompt = st.chat_input("Digite sua mensagem...")

if prompt:
    # Salva a imagem, se houver
    image_path = None
    image_base64 = None
    if uploaded_image:
        os.makedirs("data/images", exist_ok=True)
        image_bytes = uploaded_image.read()
        # Converte para base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        # Salva arquivo para referência
        ext = uploaded_image.name.split('.')[-1]
        image_path = f"data/images/{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
    
    # Adiciona mensagem do usuário
    user_msg = {"role": "user", "content": prompt}
    if image_path:
        user_msg["image"] = image_path
    st.session_state.messages.append(user_msg)
    
    # Salva no banco
    conn.execute("INSERT INTO messages (conversation_id, role, content, image_path, model) VALUES (?,?,?,?,?)",
                 (st.session_state.conv_id, "user", prompt, image_path, model_id))
    conn.commit()
    
    # Atualiza título
    if len(st.session_state.messages) == 1:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (prompt[:50], st.session_state.conv_id))
        conn.commit()
    
    # Exibe mensagem do usuário
    with st.chat_message("user"):
        if image_path:
            st.image(image_path, width=300)
        st.markdown(prompt)
    
    # Contexto RAG e web
    rag_text = rag.search(prompt)
    web_text = web_search(prompt) if web_enabled else ""
    
    # Prepara mensagens para o modelo
    if image_base64:
        # Modo multimodal: usa modelo de visão
        system_vision = f"""Você é um assistente multimodal. Analise a imagem fornecida e responda à pergunta do usuário.
Contexto adicional (documentos): {rag_text if rag_text else 'Nenhum'}
Informações da web: {web_text if web_text else 'Nenhuma'}
Responda em português."""
        # O prompt do usuário já contém a imagem e o texto
        user_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt}]
        vision_messages = [{"role": "system", "content": system_vision},
                           {"role": "user", "content": user_content}]
        try:
            with st.chat_message("assistant"):
                with st.spinner("🧠 Analisando imagem..."):
                    answer, used_model = llm.vision_generate(image_base64, prompt, temperature, max_tokens)
                    st.markdown(answer)
                st.caption(f"🤖 Modelo de visão: {used_model}")
                # Áudio e downloads
                audio_file = tts(answer)
                if audio_file:
                    with open(audio_file, "rb") as f:
                        st.audio(f.read(), format="audio/mp3")
                c1, c2, c3 = st.columns(3)
                c1.download_button("📄 TXT", export([{"role":"assistant","content":answer}], 'txt'), file_name="resposta.txt")
                c2.download_button("📑 DOCX", export([{"role":"assistant","content":answer}], 'docx'), file_name="resposta.docx")
                c3.download_button("📝 Conversa", export(st.session_state.messages, 'md'), file_name="conversa.md")
        except Exception as e:
            answer = f"❌ Erro no modelo de visão: {str(e)}"
            with st.chat_message("assistant"):
                st.error(answer)
            used_model = "error"
    else:
        # Modo texto normal
        system_text = f"""Você é um assistente útil com acesso a informações externas.
📚 Documentos: {rag_text if rag_text else 'Nenhum'}
🌐 Web: {web_text if web_text else 'Nenhuma'}
Responda em português."""
        recent = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-MAX_HISTORY:]]
        final_msgs = [{"role": "system", "content": system_text}] + recent
        
        with st.chat_message("assistant"):
            if st.session_state.streaming:
                placeholder = st.empty()
                full = ""
                try:
                    for chunk in llm.generate_stream(final_msgs, model_id, temperature, max_tokens):
                        full += chunk
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                    answer = full
                    used_model = model_id
                except Exception as e:
                    answer = f"❌ Erro: {str(e)}"
                    placeholder.markdown(answer)
                    used_model = "error"
            else:
                with st.spinner("Pensando..."):
                    try:
                        answer, used_model = llm.text_generate(final_msgs, model_id, temperature, max_tokens)
                    except Exception as e:
                        answer = f"❌ Erro: {str(e)}"
                        used_model = "error"
                    st.markdown(answer)
            st.caption(f"🤖 Modelo: {used_model}")
            audio_file = tts(answer)
            if audio_file:
                with open(audio_file, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
            c1, c2, c3 = st.columns(3)
            c1.download_button("📄 TXT", export([{"role":"assistant","content":answer}], 'txt'), file_name="resposta.txt")
            c2.download_button("📑 DOCX", export([{"role":"assistant","content":answer}], 'docx'), file_name="resposta.docx")
            c3.download_button("📝 Conversa", export(st.session_state.messages, 'md'), file_name="conversa.md")
    
    # Salva resposta
    st.session_state.messages.append({"role": "assistant", "content": answer})
    conn.execute("INSERT INTO messages (conversation_id, role, content, model) VALUES (?,?,?,?)",
                 (st.session_state.conv_id, "assistant", answer, used_model))
    conn.commit()
    st.rerun()
