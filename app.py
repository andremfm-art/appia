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
# ✅ Download TXT/DOCX
# ✅ Seleção de múltiplos modelos
# ✅ Fallback automático entre modelos
# ✅ Streaming de tokens
# ✅ Memória deslizante (janela de contexto)
# ✅ Timeouts e tratamento de erros
# ✅ Segurança via .env
# ✅ Arquitetura limpa e modular
# ✅ Otimizado para notebook comum
# ✅ Pronto para produção e portfólio
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
import sys
import requests
import base64
import time
import logging
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv

# =========================================================
# CONFIGURAÇÃO DE LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# CARREGAR VARIÁVEIS DE AMBIENTE
# =========================================================

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
if not NVIDIA_API_KEY:
    st.error("⚠️ NVIDIA_API_KEY não encontrada! Crie um arquivo .env com sua chave.")
    st.stop()

# =========================================================
# CONFIGURAÇÕES GLOBAIS
# =========================================================

BASE_URL = "https://integrate.api.nvidia.com/v1"

# Modelos validados no catálogo NVIDIA (2026)
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

# Constantes de otimização
MAX_HISTORY_MESSAGES = 10  # Janela deslizante
DEFAULT_MAX_TOKENS = 1024  # Respostas mais rápidas
REQUEST_TIMEOUT = 30  # Timeout para APIs
MAX_RAG_RESULTS = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TTS_CHARS = 500

# =========================================================
# CLIENT NVIDIA
# =========================================================

@st.cache_resource
def get_nvidia_client():
    """Retorna cliente NVIDIA com cache."""
    return OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)

client = get_nvidia_client()

# =========================================================
# BANCO DE DADOS SQLITE (Multi-conversa)
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
    
    # Remove múltiplos espaços e quebras de linha
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
                logger.info(f"Tentando modelo: {current_model}")
                
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    timeout=timeout
                )
                
                logger.info(f"Sucesso com: {current_model}")
                return response.choices[0].message.content, current_model
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Falha no modelo {current_model}: {last_error}")
                
                if attempt < len(models_to_try) - 1:
                    time.sleep(1)  # Pausa antes do fallback
                    continue
        
        # Se todos falharam
        error_msg = f"❌ Erro: Todos os modelos falharam. Último erro: {last_error}"
        logger.error(error_msg)
        return error_msg, "error"
    
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
            logger.error(f"Erro no streaming: {str(e)}")
            yield f"\n\n❌ Erro no streaming: {str(e)}"

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
        
        # Limpa e divide o texto
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
        
        response = requests.post(
            invoke_url, 
            headers=headers, 
            json=payload, 
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            if "artifacts" in data:
                img_base64 = data["artifacts"][0]["base64"]
                img_data = base64.b64decode(img_base64)
                return Image.open(BytesIO(img_data))
        
        logger.warning(f"API de imagem falhou: {response.status_code}")
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
        
        # Limita tamanho para performance
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
            text = "\n\n".join([
                f"{msg['role'].upper()}:\n{msg['content']}" 
                for msg in messages
            ])
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
            text = "\n\n".join([
                f"## {msg['role'].upper()}\n\n{msg['content']}" 
                for msg in messages
            ])
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

# CSS profissional
st.markdown("""
<style>
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #76B900;
        margin-bottom: 0.5rem;
    }
    
    /* Botões */
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
    
    /* Métricas */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    /* Chat */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Loading */
    .stSpinner > div {
        border-color: #76B900 !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Badge do modelo */
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

# =========================================================
# HEADER
# =========================================================

st.markdown('<p class="main-header">🤖 NVIDIA NIM Assistant</p>', unsafe_allow_html=True)
st.caption("🚀 Assistente IA Multimodal • 100% Grátis • Sem Limites • Produção Ready")

# =========================================================
# INICIALIZAÇÃO DO ESTADO
# =========================================================

if "conversation_id" not in st.session_state:
    # Carrega ou cria conversa
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
# SIDEBAR - CONFIGURAÇÕES
# =========================================================

with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Modelo LLM
    st.subheader("🧠 Modelo LLM")
    selected_model_name = st.selectbox(
        "Escolha o modelo",
        list(CHAT_MODELS.keys()),
        index=1  # Llama 70B como padrão
    )
    model_id = CHAT_MODELS[selected_model_name]
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "🌡️ Temperatura",
            0.0, 1.0, 0.7, 0.1,
            help="Maior = mais criativo"
        )
    with col2:
        max_tokens = st.slider(
            "📏 Max Tokens",
            256, 2048, DEFAULT_MAX_TOKENS, 128,
            help="Limite de tokens na resposta"
        )
    
    # Streaming toggle
    st.session_state.enable_streaming = st.toggle(
        "⚡ Streaming de tokens",
        value=True,
        help="Resposta em tempo real estilo ChatGPT"
    )
    
    # Busca web toggle
    enable_web_search = st.toggle(
        "🌐 Busca Web",
        value=True,
        help="Buscar informações atualizadas"
    )
    
    # Embedding
    st.subheader("🔍 Modelo de Embedding")
    selected_embed_name = st.selectbox(
        "Embedding para RAG",
        list(EMBEDDING_MODELS.keys()),
        index=0
    )
    embed_model_id = EMBEDDING_MODELS[selected_embed_name]
    
    st.divider()
    
    # Upload PDF
    st.header("📄 Documentos PDF")
    uploaded_files = st.file_uploader(
        "Upload de PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Arraste PDFs para enriquecer o contexto"
    )
    
    if uploaded_files:
        if st.button("🔄 Processar PDFs", type="primary", use_container_width=True):
            total_chunks = 0
            
            with st.spinner("Processando documentos..."):
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Salva arquivo
                        os.makedirs("data/uploads", exist_ok=True)
                        file_path = f"data/uploads/{uploaded_file.name}"
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extrai texto
                        pdf = PdfReader(file_path)
                        text = ""
                        
                        for page in pdf.pages:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted
                        
                        # Adiciona ao RAG
                        chunks = rag_service.add_document(
                            uploaded_file.name, 
                            text, 
                            embed_model_id
                        )
                        total_chunks += chunks
                        
                        # Atualiza progresso
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.error(f"Erro no arquivo {uploaded_file.name}: {str(e)}")
                
                if total_chunks > 0:
                    st.success(f"✅ {total_chunks} chunks processados com sucesso!")
                else:
                    st.warning("⚠️ Nenhum texto extraído dos PDFs")
    
    st.divider()
    
    # Gerenciar conversas
    st.header("💬 Conversas")
    
    cursor = conn.cursor()
    conversations = cursor.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY updated_at DESC"
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
                
                st.session_state.messages = [
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in messages
                ]
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
    
    # Limpar tudo
    if st.button("🗑️ Limpar Tudo", use_container_width=True):
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        conn.commit()
        
        # Limpa ChromaDB
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
    conv_count = len(conversations)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mensagens", msg_count)
    with col2:
        st.metric("Documentos", doc_count)
    with col3:
        st.metric("Conversas", conv_count)
    
    st.caption(f"🤖 Modelo: {selected_model_name}")
    st.caption(f"📏 Janela: {MAX_HISTORY_MESSAGES} mensagens")
    st.caption(f"🎯 Max tokens: {max_tokens}")

# =========================================================
# ÁREA DO CHAT
# =========================================================

# Container do chat
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# =========================================================
# INPUT DO USUÁRIO
# =========================================================

if prompt := st.chat_input("💬 Digite sua mensagem..."):
    # Adiciona mensagem do usuário
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
    
    # Atualiza título da conversa
    if len(st.session_state.messages) <= 2:
        title = prompt[:50] + "..." if len(prompt) > 50 else prompt
        cursor.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, st.session_state.conversation_id)
        )
        conn.commit()
    
    # Mostra mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepara contexto
    with st.spinner("🔍 Buscando contexto..."):
        # RAG
        rag_context = rag_service.search(prompt, embed_model_id)
        
        # Web Search
        web_context = ""
        if enable_web_search:
            web_context = web_search(prompt)
    
    # System prompt otimizado
    system_prompt = f"""Você é um assistente IA avançado rodando na infraestrutura NVIDIA NIM.

📚 CONTEXTO DE DOCUMENTOS:
{rag_context if rag_context else "Nenhum documento relevante encontrado."}

🌐 INFORMAÇÕES DA WEB:
{web_context if web_context else "Busca web desativada."}

INSTRUÇÕES:
- Responda em português do Brasil
- Use os contextos quando relevante
- Seja preciso, útil e direto
- Se não souber, admita honestamente
- Para código, use formatação markdown"""

    # Janela deslizante (últimas N mensagens)
    recent_messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    
    final_messages = [
        {"role": "system", "content": system_prompt}
    ] + recent_messages
    
    # Resposta do assistente
    with st.chat_message("assistant"):
        if st.session_state.enable_streaming:
            # Streaming de tokens
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in llm_service.generate_stream(
                final_messages,
                model_id,
                temperature,
                max_tokens
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            answer = full_response
            used_model = model_id
        else:
            # Sem streaming
            with st.spinner("🤔 Pensando..."):
                answer, used_model = llm_service.generate(
                    final_messages,
                    model_id,
                    temperature,
                    max_tokens
                )
                st.markdown(answer)
        
        # Badge do modelo
        st.caption(f"<span class='model-badge'>🤖 {used_model}</span>", unsafe_allow_html=True)
        
        # TTS
        audio_file = text_to_speech(answer)
        if audio_file:
            with open(audio_file, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
        
        # Downloads
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # TXT
            txt_data = export_conversation(
                [{"role": "assistant", "content": answer}], 
                "txt"
            )
            if txt_data:
                st.download_button(
                    "📄 Baixar TXT",
                    txt_data,
                    file_name=f"resposta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            # DOCX
            docx_data = export_conversation(
                [{"role": "assistant", "content": answer}], 
                "docx"
            )
            if docx_data:
                st.download_button(
                    "📑 Baixar DOCX",
                    docx_data,
                    file_name=f"resposta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        
        with col3:
            # Conversa completa
            full_export = export_conversation(st.session_state.messages, "md")
            if full_export:
                st.download_button(
                    "📝 Exportar Conversa",
                    full_export,
                    file_name=f"conversa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
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
    
    # Recarrega para mostrar mensagens
    st.rerun()

# =========================================================
# GERADOR DE IMAGENS
# =========================================================

st.divider()
st.header("🎨 Gerador de Imagens NVIDIA")

col1, col2 = st.columns([3, 1])

with col1:
    img_prompt = st.text_area(
        "Descreva a imagem que deseja gerar",
        placeholder="Ex: Uma cidade futurista brasileira com palmeiras e carros voadores ao pôr do sol...",
        height=100
    )

with col2:
    img_style = st.selectbox(
        "Estilo",
        ["Fotorealista", "Arte digital", "3D", "Anime", "Pixel art"],
        index=0
    )
    
    img_size = st.selectbox(
        "Tamanho",
        ["1024x1024", "512x512", "768x768"],
        index=0
    )

if st.button("🎨 Gerar Imagem", type="primary", use_container_width=True):
    if img_prompt:
        # Adiciona estilo ao prompt
        styled_prompt = f"{img_prompt}, estilo {img_style.lower()}, alta qualidade, detalhado"
        
        with st.spinner("🎨 Criando sua imagem..."):
            # Tenta NVIDIA primeiro
            image = generate_image(styled_prompt)
            
            if image:
                st.image(image, caption=f"Gerado por NVIDIA - {img_style}", use_column_width=True)
                
                # Download
                buf = BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Baixar Imagem PNG",
                    buf.getvalue(),
                    file_name=f"nvidia_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            else:
                # Fallback para API gratuita
                st.warning("⚠️ API NVIDIA indisponível, usando fallback gratuito...")
                
                fallback_url = f"https://image.pollinations.ai/prompt/{styled_prompt.replace(' ', '%20')}"
                
                try:
                    response = requests.get(fallback_url, timeout=REQUEST_TIMEOUT)
                    if response.status_code == 200:
                        st.image(response.content, caption="Imagem gerada (fallback)", use_column_width=True)
                    else:
                        st.image(fallback_url, caption="Imagem gerada (fallback)")
                except:
                    st.image(fallback_url, caption="Imagem gerada (fallback)")
    else:
        st.warning("⚠️ Por favor, descreva a imagem que deseja gerar.")

# =========================================================
# RODAPÉ
# =========================================================

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>🚀 <strong>NVIDIA NIM Assistant</strong> • v2.0 • Produção Ready</p>
        <p><small>100% Grátis • Sem Limites • Open Source</small></p>
        <p style='font-size: 0.8rem;'>
            Modelos: NVIDIA NIM • Embeddings: ChromaDB • Framework: Streamlit<br>
            © 2026 - Projeto de Portfólio Profissional
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# CLEANUP
# =========================================================

# Fecha conexões ao final (quando o app for desligado)
import atexit

@atexit.register
def cleanup():
    """Limpa recursos ao fechar o app."""
    try:
        conn.close()
        logger.info("Conexão SQLite fechada")
    except:
        pass
