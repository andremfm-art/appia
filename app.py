# =========================================================
# APP IA COMPLETO - NVIDIA NIM (PRODUÇÃO READY)
# =========================================================
# FEATURES
# =========================================================
# ✅ Chat multimodal estilo ChatGPT
# ✅ Histórico persistente SQLite multi-conversa
# ✅ RAG com PDFs e ChromaDB persistente
# ✅ Busca Web inteligente
# ✅ Geração de Imagem via NVIDIA
# ✅ Text-to-Speech multilíngue
# ✅ Download TXT/DOCX/Markdown
# ✅ Seleção de múltiplos modelos
# ✅ Fallback automático entre modelos
# ✅ Streaming de tokens
# ✅ Memória deslizante (janela de contexto)
# ✅ Timeouts e tratamento de erros
# ✅ Upload de arquivo com chave API
# ✅ Input manual da chave
# ✅ Arquitetura limpa e modular
# ✅ Otimizado para notebook comum
# ✅ Pronto para produção e portfólio
# ✅ GitHub Safe - sem chaves expostas
# =========================================================

# =========================================================
# INSTALAR:
# =========================================================
# pip install -r requirements.txt
# streamlit run app.py
# =========================================================

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
from pathlib import Path
import atexit

# =========================================================
# CONFIGURAÇÃO DE LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# FUNÇÃO DE CARREGAMENTO SIMPLES DA API KEY
# =========================================================

def load_api_key() -> Optional[str]:
    """
    Carrega a API key da NVIDIA.
    
    Ordem de prioridade:
    1. Streamlit Secrets (Cloud)
    2. Upload de arquivo .env ou .txt
    3. Input manual
    """
    api_key = None
    
    # Método 1: Streamlit Secrets (Cloud)
    try:
        if hasattr(st, 'secrets') and 'NVIDIA_API_KEY' in st.secrets:
            api_key = st.secrets['NVIDIA_API_KEY']
            logger.info("✅ API key carregada do Streamlit Secrets")
    except:
        pass
    
    # Método 2: Upload de arquivo com a chave
    if not api_key:
        api_key = get_api_key_from_file()
        if api_key:
            logger.info("✅ API key carregada do arquivo enviado")
    
    # Método 3: Variável de ambiente
    if not api_key:
        api_key = os.environ.get('NVIDIA_API_KEY')
        if api_key:
            logger.info("✅ API key carregada da variável de ambiente")
    
    # Método 4: Input manual (último recurso)
    if not api_key:
        api_key = get_api_key_manual()
        if api_key:
            logger.info("✅ API key fornecida manualmente")
    
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")
        return api_key
    
    return None


def get_api_key_from_file() -> Optional[str]:
    """Solicita upload de arquivo .env ou .txt com a chave."""
    
    if 'api_key_uploaded' not in st.session_state:
        st.session_state.api_key_uploaded = False
        st.session_state.api_key_value = None
    
    if st.session_state.api_key_uploaded and st.session_state.api_key_value:
        return st.session_state.api_key_value
    
    st.markdown("## 🤖 NVIDIA NIM Assistant")
    st.markdown("### 🔑 Configuração da API Key")
    st.markdown("**Para usar o app, você precisa de uma chave gratuita da NVIDIA.**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        #### 📤 Upload do arquivo da chave
        
        Faça upload do arquivo `.env` ou `.txt` com sua chave:
        """)
        
        uploaded_file = st.file_uploader(
            "Arraste o arquivo aqui",
            type=['env', 'txt'],
            help="Arquivo .env ou .txt com a chave"
        )
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8').strip()
                
                if '=' in content:
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('NVIDIA_API_KEY='):
                            api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                            if api_key.startswith('nvapi-'):
                                st.session_state.api_key_value = api_key
                                st.session_state.api_key_uploaded = True
                                st.success("✅ Chave extraída do arquivo .env!")
                                st.rerun()
                            break
                else:
                    if content.startswith('nvapi-'):
                        st.session_state.api_key_value = content
                        st.session_state.api_key_uploaded = True
                        st.success("✅ Chave carregada do arquivo!")
                        st.rerun()
                    else:
                        st.error("❌ A chave deve começar com 'nvapi-'")
            except Exception as e:
                st.error(f"❌ Erro ao ler arquivo: {str(e)}")
    
    with col2:
        st.markdown("""
        #### ✍️ Digitar a chave
        
        Cole sua chave manualmente:
        """)
        
        manual_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-..."
        )
        
        if manual_key:
            if manual_key.startswith('nvapi-'):
                st.session_state.api_key_value = manual_key
                st.session_state.api_key_uploaded = True
                st.success("✅ Chave aceita!")
                st.rerun()
            else:
                st.error("❌ Deve começar com 'nvapi-'")
    
    st.divider()
    st.markdown("""
    ### 📝 Como obter sua chave gratuita:
    
    1. Acesse **[build.nvidia.com](https://build.nvidia.com)**
    2. Crie uma conta gratuita
    3. Gere sua API key
    4. Volte aqui e faça upload ou cole a chave
    """)
    
    return None


def get_api_key_manual() -> Optional[str]:
    """Fallback: input manual da chave."""
    return None


# =========================================================
# CARREGAR API KEY
# =========================================================

NVIDIA_API_KEY = load_api_key()

if not NVIDIA_API_KEY:
    st.error("## ⚠️ API Key não configurada!")
    st.info("Faça upload do arquivo com a chave ou cole manualmente no campo acima.")
    st.stop()

# Validação rápida
if not NVIDIA_API_KEY.startswith('nvapi-'):
    st.error("## ❌ API Key inválida!")
    st.info("A chave deve começar com 'nvapi-'")
    st.stop()

# =========================================================
# CONFIGURAÇÕES GLOBAIS
# =========================================================

BASE_URL = "https://integrate.api.nvidia.com/v1"

CHAT_MODELS = {
    "🌟 Llama 3.1 405B (Top)": "meta/llama-3.1-405b-instruct",
    "💪 Llama 3.1 70B": "meta/llama-3.1-70b-instruct",
    "⚡ Llama 3.1 8B (Rápido)": "meta/llama-3.1-8b-instruct",
    "🎯 DeepSeek-R1 (Raciocínio)": "deepseek-ai/deepseek-r1-distill-llama-8b",
    "🧠 Nemotron 70B": "nvidia/nemotron-70b-instruct",
    "🔬 Mistral Large 2": "mistralai/mistral-large-2-instruct",
    "📚 Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct-v0.1",
    "💻 Codestral 22B (Código)": "mistralai/codestral-22b-instruct-v0.1",
    "📝 Gemma 2 9B": "google/gemma-2-9b-it",
}

EMBEDDING_MODELS = {
    "NV-Embed-QA v5 (Melhor)": "nvidia/nv-embedqa-e5-v5",
    "NV-EmbedQA 1B v2 (Rápido)": "nvidia/llama-3.2-nv-embedqa-1b-v2",
}

MAX_HISTORY_MESSAGES = 10
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT = 30
MAX_RAG_RESULTS = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TTS_CHARS = 500

# =========================================================
# CLIENT NVIDIA (com cache)
# =========================================================

@st.cache_resource
def get_nvidia_client(_api_key: str):
    """Retorna cliente NVIDIA com cache."""
    return OpenAI(base_url=BASE_URL, api_key=_api_key)

client = get_nvidia_client(NVIDIA_API_KEY)

# =========================================================
# BANCO DE DADOS SQLITE
# =========================================================

@st.cache_resource
def get_sqlite_connection():
    """Retorna conexão SQLite com cache."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/chat_history.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT DEFAULT 'Nova Conversa',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    return conn

conn = get_sqlite_connection()

# =========================================================
# CHROMADB PERSISTENTE
# =========================================================

@st.cache_resource
def get_chroma_client():
    """Retorna cliente ChromaDB persistente."""
    os.makedirs("data/chroma_db", exist_ok=True)
    return chromadb.PersistentClient(path="data/chroma_db")

chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection(name="documents")

# =========================================================
# FUNÇÕES UTILITÁRIAS
# =========================================================

def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Acesso seguro a dicionários."""
    try:
        return obj.get(key, default) if isinstance(obj, dict) else default
    except:
        return default

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Divide texto em chunks com overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (size - overlap)
    
    return chunks

def clean_text(text: str) -> str:
    """Limpa texto para melhor processamento."""
    if not text:
        return ""
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

# =========================================================
# SERVIÇO DE LLM COM FALLBACK
# =========================================================

class LLMService:
    """Serviço de LLM com retry e fallback automático."""
    
    def __init__(self):
        self.client = client
        self.fallback_chain = [
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
        ]
    
    def generate(self, messages: List[Dict], model: str,
                 temperature: float = 0.7, max_tokens: int = DEFAULT_MAX_TOKENS,
                 timeout: int = REQUEST_TIMEOUT) -> Tuple[str, str]:
        """Gera resposta com fallback automático."""
        models_to_try = [model] + [m for m in self.fallback_chain if m != model]
        last_error = None
        
        for attempt, current_model in enumerate(models_to_try):
            try:
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    timeout=timeout
                )
                return response.choices[0].message.content, current_model
            except Exception as e:
                last_error = str(e)
                if attempt < len(models_to_try) - 1:
                    time.sleep(1)
                    continue
        
        return f"❌ Erro: {last_error}", "error"
    
    def generate_stream(self, messages: List[Dict], model: str,
                        temperature: float = 0.7, max_tokens: int = DEFAULT_MAX_TOKENS):
        """Gera resposta com streaming de tokens."""
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        except Exception as e:
            yield f"\n\n❌ Erro: {str(e)}"

llm_service = LLMService()

# =========================================================
# SERVIÇO DE RAG
# =========================================================

class RAGService:
    """Serviço de RAG com embeddings NVIDIA."""
    
    def __init__(self):
        self.client = client
        self.collection = collection
    
    def get_embedding(self, text: str, model: str = "nvidia/nv-embedqa-e5-v5") -> Optional[List[float]]:
        """Gera embedding para um texto."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=model,
                encoding_format="float",
                timeout=REQUEST_TIMEOUT
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {str(e)}")
            return None
    
    def search(self, query: str, model: str = "nvidia/nv-embedqa-e5-v5",
               n_results: int = MAX_RAG_RESULTS) -> str:
        """Busca documentos similares."""
        try:
            query_embedding = self.get_embedding(query, model)
            if not query_embedding:
                return ""
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            if results and results.get('documents'):
                docs = results['documents'][0]
                return "\n\n".join(docs)
            return ""
        except Exception as e:
            logger.error(f"Erro na busca RAG: {str(e)}")
            return ""
    
    def add_document(self, file_name: str, text: str, model: str = "nvidia/nv-embedqa-e5-v5") -> int:
        """Adiciona documento ao índice RAG."""
        if not text:
            return 0
        
        text = clean_text(text)
        chunks = chunk_text(text)
        
        if not chunks:
            return 0
        
        total_added = 0
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.get_embedding(chunk, model)
                if embedding:
                    self.collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{"source": file_name, "chunk": i}],
                        ids=[f"{file_name}_{i}_{datetime.now().timestamp()}"]
                    )
                    total_added += 1
            except Exception as e:
                logger.warning(f"Erro ao adicionar chunk {i}: {str(e)}")
        
        return total_added

rag_service = RAGService()

# =========================================================
# SERVIÇO DE BUSCA WEB
# =========================================================

def web_search(query: str, max_results: int = 3) -> str:
    """Busca na web com DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                title = safe_get(r, 'title', 'Sem título')
                body = safe_get(r, 'body', '')
                results.append(f"• {title}\n{body}")
            return "\n\n".join(results) if results else ""
    except Exception as e:
        logger.error(f"Erro na busca web: {str(e)}")
        return ""

# =========================================================
# SERVIÇO DE IMAGEM
# =========================================================

def generate_image(prompt: str, model: str = "stabilityai/sdxl-turbo") -> Optional[Image.Image]:
    """Gera imagem via NVIDIA com fallback."""
    try:
        invoke_url = f"https://ai.api.nvidia.com/v1/genai/{model}"
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
        }
        
        payload = {
            "text_prompts": [{"text": prompt, "weight": 1.0}],
            "cfg_scale": 5,
            "steps": 25,
            "height": 1024,
            "width": 1024,
        }
        
        response = requests.post(invoke_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if "artifacts" in data:
                img_base64 = data["artifacts"][0]["base64"]
                img_data = base64.b64decode(img_base64)
                return Image.open(BytesIO(img_data))
        return None
    except Exception as e:
        logger.error(f"Erro na geração de imagem: {str(e)}")
        return None

# =========================================================
# SERVIÇO DE TTS
# =========================================================

def text_to_speech(text: str, lang: str = "pt") -> Optional[str]:
    """Converte texto em áudio."""
    try:
        if not text:
            return None
        text = text[:MAX_TTS_CHARS]
        tts = gTTS(text, lang=lang)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        return temp_audio.name
    except Exception as e:
        logger.error(f"Erro no TTS: {str(e)}")
        return None

# =========================================================
# SERVIÇO DE EXPORTAÇÃO
# =========================================================

def export_conversation(messages: List[Dict], format: str = "txt") -> Optional[bytes]:
    """Exporta conversa em diferentes formatos."""
    try:
        if format == "txt":
            text = "\n\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in messages])
            return text.encode('utf-8')
        
        elif format == "docx":
            doc = Document()
            doc.add_heading("Conversa - NVIDIA NIM Assistant", level=1)
            for msg in messages:
                doc.add_heading(msg['role'].upper(), level=2)
                doc.add_paragraph(msg['content'])
            buffer = BytesIO()
            doc.save(buffer)
            return buffer.getvalue()
        
        elif format == "md":
            text = "\n\n".join([f"## {msg['role'].upper()}\n\n{msg['content']}" for msg in messages])
            return text.encode('utf-8')
        
        return None
    except Exception as e:
        logger.error(f"Erro na exportação: {str(e)}")
        return None

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="NVIDIA NIM Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #76B900;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background-color: #76B900;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #5A8F00;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .model-badge {
        background: #76B900;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 NVIDIA NIM Assistant</p>', unsafe_allow_html=True)
st.caption("🚀 Assistente IA Multimodal • 100% Grátis • Sem Limites • Upload da Chave")

# Badge da chave
st.sidebar.success(f"🔒 Chave: {NVIDIA_API_KEY[:10]}...{NVIDIA_API_KEY[-4:]}")

# =========================================================
# INICIALIZAÇÃO DO ESTADO
# =========================================================

if "conversation_id" not in st.session_state:
    cursor = conn.cursor()
    conversations = cursor.execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 1"
    ).fetchall()
    
    if conversations:
        st.session_state.conversation_id = conversations[0]["id"]
    else:
        cursor.execute("INSERT INTO conversations (title) VALUES (?)", ("Nova Conversa",))
        conn.commit()
        st.session_state.conversation_id = cursor.lastrowid

if "messages" not in st.session_state:
    cursor = conn.cursor()
    messages = cursor.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
        (st.session_state.conversation_id,)
    ).fetchall()
    
    st.session_state.messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]

if "enable_streaming" not in st.session_state:
    st.session_state.enable_streaming = True

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Modelo LLM
    st.subheader("🧠 Modelo LLM")
    selected_model_name = st.selectbox(
        "Escolha o modelo",
        list(CHAT_MODELS.keys()),
        index=1
    )
    model_id = CHAT_MODELS[selected_model_name]
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("🌡️ Temperatura", 0.0, 1.0, 0.7, 0.1)
    with col2:
        max_tokens = st.slider("📏 Max Tokens", 256, 2048, DEFAULT_MAX_TOKENS, 128)
    
    st.session_state.enable_streaming = st.toggle("⚡ Streaming", value=True)
    enable_web_search = st.toggle("🌐 Busca Web", value=True)
    
    # Embedding
    st.subheader("🔍 Embedding")
    selected_embed_name = st.selectbox(
        "Modelo",
        list(EMBEDDING_MODELS.keys()),
        index=0
    )
    embed_model_id = EMBEDDING_MODELS[selected_embed_name]
    
    st.divider()
    
    # Upload PDF
    st.header("📄 PDFs para RAG")
    uploaded_files = st.file_uploader(
        "Upload de PDFs",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🔄 Processar", type="primary", use_container_width=True):
            total_chunks = 0
            with st.spinner("Processando..."):
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        os.makedirs("data/uploads", exist_ok=True)
                        file_path = f"data/uploads/{uploaded_file.name}"
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        pdf = PdfReader(file_path)
                        text = ""
                        for page in pdf.pages:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted
                        
                        chunks = rag_service.add_document(uploaded_file.name, text, embed_model_id)
                        total_chunks += chunks
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    except Exception as e:
                        st.error(f"Erro: {str(e)}")
                
                if total_chunks > 0:
                    st.success(f"✅ {total_chunks} chunks processados!")
    
    st.divider()
    
    # Conversas
    st.header("💬 Conversas")
    cursor = conn.cursor()
    conversations = cursor.execute(
        "SELECT id, title FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    
    for conv in conversations:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"📝 {conv['title']}", key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.conversation_id = conv["id"]
                messages = cursor.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
                    (conv["id"],)
                ).fetchall()
                st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in messages]
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{conv['id']}"):
                cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv["id"],))
                cursor.execute("DELETE FROM conversations WHERE id = ?", (conv["id"],))
                conn.commit()
                st.rerun()
    
    if st.button("➕ Nova Conversa", use_container_width=True):
        cursor.execute("INSERT INTO conversations (title) VALUES (?)", ("Nova Conversa",))
        conn.commit()
        st.session_state.conversation_id = cursor.lastrowid
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Limpar Tudo", use_container_width=True):
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        conn.commit()
        ids = collection.get().get('ids', [])
        if ids:
            collection.delete(ids=ids)
        st.session_state.messages = []
        st.rerun()
    
    # Status
    st.divider()
    st.header("📊 Status")
    msg_count = cursor.execute("SELECT COUNT(*) as count FROM messages").fetchone()["count"]
    doc_count = len(collection.get().get('ids', []))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mensagens", msg_count)
    with col2:
        st.metric("Documentos", doc_count)

# =========================================================
# CHAT
# =========================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# INPUT
# =========================================================

if prompt := st.chat_input("💬 Digite sua mensagem..."):
    # Salva mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, model) VALUES (?, ?, ?, ?)",
        (st.session_state.conversation_id, "user", prompt, model_id)
    )
    cursor.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (st.session_state.conversation_id,)
    )
    conn.commit()
    
    # Atualiza título
    if len(st.session_state.messages) <= 2:
        title = prompt[:50] + "..." if len(prompt) > 50 else prompt
        cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, st.session_state.conversation_id))
        conn.commit()
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Contexto
    with st.spinner("🔍 Buscando contexto..."):
        rag_context = rag_service.search(prompt, embed_model_id)
        web_context = web_search(prompt) if enable_web_search else ""
    
    # System prompt
    system_prompt = f"""Você é um assistente IA NVIDIA NIM.

📚 DOCUMENTOS:
{rag_context if rag_context else "Nenhum."}

🌐 WEB:
{web_context if web_context else "Desativada."}

Responda em português, seja útil e preciso."""

    recent_messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    final_messages = [{"role": "system", "content": system_prompt}] + recent_messages
    
    # Resposta
    with st.chat_message("assistant"):
        if st.session_state.enable_streaming:
            response_placeholder = st.empty()
            full_response = ""
            for chunk in llm_service.generate_stream(final_messages, model_id, temperature, max_tokens):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            answer = full_response
            used_model = model_id
        else:
            with st.spinner("🤔 Pensando..."):
                answer, used_model = llm_service.generate(final_messages, model_id, temperature, max_tokens)
                st.markdown(answer)
        
        st.caption(f"<span class='model-badge'>🤖 {used_model}</span>", unsafe_allow_html=True)
        
        # TTS
        audio_file = text_to_speech(answer)
        if audio_file:
            with open(audio_file, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
        
        # Downloads
        col1, col2, col3 = st.columns(3)
        with col1:
            txt_data = export_conversation([{"role": "assistant", "content": answer}], "txt")
            if txt_data:
                st.download_button("📄 TXT", txt_data, file_name=f"resposta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with col2:
            docx_data = export_conversation([{"role": "assistant", "content": answer}], "docx")
            if docx_data:
                st.download_button("📑 DOCX", docx_data, file_name=f"resposta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")
        with col3:
            md_data = export_conversation(st.session_state.messages, "md")
            if md_data:
                st.download_button("📝 Conversa", md_data, file_name=f"conversa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    # Salva resposta
    st.session_state.messages.append({"role": "assistant", "content": answer})
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, model) VALUES (?, ?, ?, ?)",
        (st.session_state.conversation_id, "assistant", answer, used_model)
    )
    cursor.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (st.session_state.conversation_id,)
    )
    conn.commit()
    st.rerun()

# =========================================================
# GERADOR DE IMAGENS
# =========================================================

st.divider()
st.header("🎨 Gerador de Imagens")

col1, col2 = st.columns([3, 1])
with col1:
    img_prompt = st.text_area("Descreva a imagem", height=100)
with col2:
    img_style = st.selectbox("Estilo", ["Fotorealista", "Arte digital", "3D", "Anime", "Pixel art"])

if st.button("🎨 Gerar Imagem", type="primary", use_container_width=True):
    if img_prompt:
        styled_prompt = f"{img_prompt}, estilo {img_style.lower()}, alta qualidade"
        with st.spinner("🎨 Criando..."):
            image = generate_image(styled_prompt)
            if image:
                st.image(image, use_column_width=True)
                buf = BytesIO()
                image.save(buf, format="PNG")
                st.download_button("⬇️ Baixar PNG", buf.getvalue(), file_name=f"imagem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            else:
                st.warning("⚠️ API indisponível, usando fallback...")
                fallback_url = f"https://image.pollinations.ai/prompt/{styled_prompt.replace(' ', '%20')}"
                st.image(fallback_url)
    else:
        st.warning("⚠️ Descreva a imagem!")

# =========================================================
# RODAPÉ
# =========================================================

st.divider()
st.caption("🚀 NVIDIA NIM Assistant • Upload de chave • GitHub Safe 🔒")

# =========================================================
# CLEANUP
# =========================================================

@atexit.register
def cleanup():
    try:
        conn.close()
    except:
        pass
