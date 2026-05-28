"""
Moura IA v3.2
- SEM sentence-transformers (elimina torchvision/transformers pesados)
- RAG com TF-IDF puro via numpy (leve, sem dependências extras)
- selectbox com label correto (fix label vazio)
- sem st.rerun() após indexar PDFs
- compatível com Python 3.14 + Streamlit Cloud
"""

import streamlit as st

st.set_page_config(
    page_title="Moura IA",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Moura IA v3.2"},
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*,*::before,*::after{box-sizing:border-box}
:root{
  --green:#22c55e;--green2:#16a34a;--teal:#2dd4bf;
  --dark:#080b10;--surface:#0f1319;--card:#141922;
  --border:#1e2736;--border2:#273040;
  --text:#e2e8f0;--muted:#64748b;--muted2:#94a3b8;
  --amber:#f59e0b;--red:#ef4444;
}
html,body{margin:0;padding:0}
[data-testid="stAppViewContainer"]{background:var(--dark)!important;font-family:'Outfit',sans-serif!important}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;padding-top:0!important}
[data-testid="stSidebar"]>div:first-child{padding-top:0!important}
[data-testid="stSidebar"] *{font-family:'Outfit',sans-serif!important}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span:not(.sc){color:var(--muted2)!important}
.stChatMessage{background:transparent!important;border:none!important;gap:12px!important}
[data-testid="stChatMessageContent"]{
  background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:14px!important;padding:18px 22px!important;
  font-family:'Outfit',sans-serif!important;font-size:.95rem!important;
  line-height:1.75!important;color:var(--text)!important;
  box-shadow:0 2px 12px rgba(0,0,0,.3)!important
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  background:#0f1f16!important;border-color:rgba(34,197,94,.25)!important
}
[data-testid="stChatMessageContent"] p{color:var(--text)!important}
[data-testid="stChatMessageContent"] strong{color:#fff!important}
[data-testid="stChatMessageContent"] code{
  background:#0d1117!important;color:var(--green)!important;
  padding:2px 7px!important;border-radius:5px!important;
  font-family:'JetBrains Mono',monospace!important;font-size:.85em!important;
  border:1px solid var(--border)!important
}
[data-testid="stChatMessageContent"] pre{
  background:#0d1117!important;border:1px solid var(--border)!important;
  border-left:3px solid var(--green)!important;border-radius:10px!important;
  padding:18px!important;overflow-x:auto!important
}
[data-testid="stChatMessageContent"] pre code{background:transparent!important;border:none!important;padding:0!important;color:#e2e8f0!important}
[data-testid="stChatMessageContent"] table{width:100%!important;border-collapse:collapse!important}
[data-testid="stChatMessageContent"] th{background:var(--border)!important;color:var(--green)!important;padding:8px 12px!important;font-weight:600!important;text-align:left!important}
[data-testid="stChatMessageContent"] td{padding:7px 12px!important;border-bottom:1px solid var(--border)!important;color:var(--text)!important}
[data-testid="stChatInput"]{background:var(--card)!important;border:1.5px solid var(--border2)!important;border-radius:16px!important;transition:border-color .2s!important}
[data-testid="stChatInput"]:focus-within{border-color:var(--green)!important;box-shadow:0 0 0 3px rgba(34,197,94,.1)!important}
[data-testid="stChatInput"] textarea{background:transparent!important;color:var(--text)!important;font-family:'Outfit',sans-serif!important;font-size:.95rem!important}
[data-testid="stChatInput"] button{background:var(--green)!important;border-radius:10px!important}
.stButton>button{background:var(--card)!important;color:var(--muted2)!important;border:1px solid var(--border)!important;border-radius:9px!important;font-family:'Outfit',sans-serif!important;font-weight:500!important;font-size:.88rem!important;transition:all .18s ease!important;padding:6px 14px!important}
.stButton>button:hover{border-color:var(--green)!important;color:var(--green)!important;background:rgba(34,197,94,.06)!important;transform:translateY(-1px)!important}
.stDownloadButton>button{background:var(--card)!important;color:var(--muted2)!important;border:1px solid var(--border)!important;border-radius:8px!important;font-size:.8rem!important;padding:5px 12px!important;font-family:'JetBrains Mono',monospace!important}
.stDownloadButton>button:hover{border-color:var(--teal)!important;color:var(--teal)!important}
.stSelectbox>div>div{background:var(--card)!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:9px!important;font-family:'Outfit',sans-serif!important}
.stSelectbox [data-baseweb="select"]{background:var(--card)!important}
.stCheckbox label{color:var(--muted2)!important;font-size:.88rem!important}
[data-testid="stFileUploader"]{background:var(--card)!important;border:1.5px dashed var(--border2)!important;border-radius:12px!important}
[data-testid="stFileUploader"] label{color:var(--muted2)!important}
.stProgress>div>div{background:var(--green)!important;border-radius:99px!important}
hr{border-color:var(--border)!important;margin:16px 0!important}
.stSpinner>div{color:var(--green)!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:99px}

/* Custom */
.sidebar-header{background:linear-gradient(180deg,#0a1a0f 0%,var(--surface) 100%);padding:24px 20px 20px;border-bottom:1px solid var(--border);margin-bottom:8px}
.logo{font-family:'Outfit',sans-serif;font-weight:800;font-size:1.5rem;letter-spacing:-.5px;background:linear-gradient(135deg,#22c55e,#2dd4bf);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1}
.tagline{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:4px}
.sec{font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:var(--muted)!important;margin:16px 0 8px;padding:0 2px}
.rag-stat{display:flex;align-items:center;justify-content:space-between;background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.2);border-radius:10px;padding:10px 14px;margin:6px 0}
.rag-stat-lbl{font-size:.75rem;color:var(--muted);font-family:'JetBrains Mono',monospace}
.rag-stat-val{font-size:.9rem;color:var(--green);font-weight:700}
.sc{display:inline-flex;align-items:center;gap:5px;background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);border-radius:99px;padding:3px 10px;font-size:.72rem;color:var(--green);font-family:'JetBrains Mono',monospace;margin:2px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.main-hdr{display:flex;align-items:center;justify-content:space-between;padding:20px 0 16px;border-bottom:1px solid var(--border);margin-bottom:28px}
.main-title{font-family:'Outfit',sans-serif;font-weight:800;font-size:1.5rem;letter-spacing:-.5px;background:linear-gradient(135deg,#22c55e,#2dd4bf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.bdg{font-family:'JetBrains Mono',monospace;font-size:.62rem;font-weight:600;letter-spacing:1.5px;padding:4px 10px;border-radius:99px;display:inline-flex;align-items:center;gap:5px}
.bg{background:rgba(34,197,94,.12);color:var(--green);border:1px solid rgba(34,197,94,.25)}
.bt{background:rgba(45,212,191,.12);color:var(--teal);border:1px solid rgba(45,212,191,.25)}
.ba{background:rgba(245,158,11,.12);color:var(--amber);border:1px solid rgba(245,158,11,.25)}
.bm{background:rgba(100,116,139,.12);color:var(--muted2);border:1px solid rgba(100,116,139,.2)}
.ctx-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.mftr{display:flex;align-items:center;gap:10px;margin-top:12px;padding-top:10px;border-top:1px solid var(--border)}
.mtag{font-family:'JetBrains Mono',monospace;font-size:.67rem;color:var(--muted);display:inline-flex;align-items:center;gap:5px}
.ww{text-align:center;padding:64px 0 32px;max-width:560px;margin:0 auto}
.wi{font-size:3.5rem;margin-bottom:20px;line-height:1}
.wt{font-family:'Outfit',sans-serif;font-weight:800;font-size:2rem;letter-spacing:-1px;background:linear-gradient(135deg,#22c55e,#2dd4bf);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}
.ws{font-size:.9rem;color:var(--muted2);line-height:1.6}
.cg{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:32px auto 0;max-width:520px}
.cc{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;text-align:left;transition:border-color .2s}
.ci{font-size:1.3rem;margin-bottom:8px}
.ct{font-weight:600;font-size:.9rem;color:var(--text);margin-bottom:4px}
.cd{font-size:.8rem;color:var(--muted2);line-height:1.5}
</style>
""", unsafe_allow_html=True)

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import os, base64, hashlib, logging, pickle, sqlite3, atexit, re, math
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import Counter

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from gtts import gTTS
from docx import Document
from PIL import Image

try:
    from duckduckgo_search import DDGS
    DDGS_OK = True
except ImportError:
    DDGS_OK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moura_ia")

for _d in ["data", "data/images", "data/tts"]:
    os.makedirs(_d, exist_ok=True)

# ─── MODELOS ─────────────────────────────────────────────────────────────────
TEXT_MODELS = {
    "Llama 3.1 70B":  "meta/llama-3.1-70b-instruct",
    "Llama 3.1 8B":   "meta/llama-3.1-8b-instruct",
    "Llama 3.3 70B":  "meta/llama-3.3-70b-instruct",
    "Nemotron 70B":   "nvidia/nemotron-70b-instruct",
    "Mistral NeMo":   "nv-mistralai/mistral-nemo-12b-instruct",
}
VISION_MODELS = [
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
]
FALLBACK    = "meta/llama-3.1-8b-instruct"
VECTORS_FILE = "data/vectors.pkl"
CHUNK_SIZE   = 600
CHUNK_OVERLAP = 100
TOP_K        = 5

# ─── API KEY ─────────────────────────────────────────────────────────────────
def get_api_key() -> Optional[str]:
    try:
        if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
            k = st.secrets["NVIDIA_API_KEY"]
            if k and k.startswith("nvapi-"): return k
    except: pass
    k = os.environ.get("NVIDIA_API_KEY", "")
    if k.startswith("nvapi-"): return k
    return st.session_state.get("api_key")

def show_api_screen():
    st.markdown("""
    <div style='max-width:440px;margin:100px auto 0;text-align:center'>
      <div style='font-family:Outfit,sans-serif;font-weight:800;font-size:2.2rem;
           background:linear-gradient(135deg,#22c55e,#2dd4bf);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           margin-bottom:6px'>Moura IA</div>
      <div style='color:#64748b;font-size:.88rem;margin-bottom:36px'>
           Configure sua NVIDIA API Key para começar
      </div>
    </div>""", unsafe_allow_html=True)
    col = st.columns([1,2,1])[1]
    with col:
        k = st.text_input("NVIDIA API Key", type="password", placeholder="nvapi-...")
        up = st.file_uploader("Ou arquivo .env / .txt", type=["env","txt","key"])
        if up:
            c = up.read().decode(errors="ignore")
            for ln in c.splitlines():
                ln = ln.strip()
                if ln.startswith("NVIDIA_API_KEY="):
                    k = ln.split("=",1)[1].strip().strip("\"'")
                elif ln.startswith("nvapi-"):
                    k = ln
                if k and k.startswith("nvapi-"): break
        if st.button("Entrar", use_container_width=True):
            if k and k.startswith("nvapi-"):
                st.session_state.api_key = k; st.rerun()
            else:
                st.error("Chave inválida — deve começar com `nvapi-`")
        st.markdown("<div style='text-align:center;margin-top:12px'><a href='https://build.nvidia.com' target='_blank' style='color:#22c55e;font-size:.8rem'>Obter chave gratuita</a></div>",
                    unsafe_allow_html=True)

# ─── DATABASE ─────────────────────────────────────────────────────────────────
class DB:
    def __init__(self, path="data/chat.db"):
        self.path = path; self._c = None; self._init()

    def _connect(self):
        c = sqlite3.connect(self.path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def _init(self):
        c = self._connect()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS conversations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT DEFAULT 'Nova Conversa',
                created_at TEXT DEFAULT(datetime('now','localtime')),
                updated_at TEXT DEFAULT(datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                image_path TEXT,
                model TEXT,
                ts TEXT DEFAULT(datetime('now','localtime')),
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_mc ON messages(conversation_id);
        """)
        c.commit(); c.close()

    @property
    def conn(self):
        if self._c is None: self._c = self._connect()
        return self._c

    def ex(self, sql, p=()):
        try:
            cur = self.conn.execute(sql, p); self.conn.commit(); return cur
        except:
            self._c = self._connect()
            cur = self.conn.execute(sql, p); self.conn.commit(); return cur

    def all(self, sql, p=()): return self.ex(sql,p).fetchall()
    def one(self, sql, p=()): return self.ex(sql,p).fetchone()
    def close(self):
        if self._c: self._c.close(); self._c = None

# ─── TF-IDF RAG (sem dependências externas) ──────────────────────────────────
def _tokenize(text: str) -> List[str]:
    """Tokeniza texto em palavras lowercase, removendo stopwords PT/EN."""
    STOP = {
        "a","o","e","de","do","da","em","um","uma","para","com","que","se",
        "por","mais","como","mas","ao","dos","das","na","no","os","as","seu",
        "sua","ou","quando","muito","nos","já","eu","também","ele","ela","são",
        "the","and","of","to","in","is","it","that","this","was","for","on",
        "are","as","with","be","at","by","from","or","an","not","have","but",
    }
    words = re.findall(r"[a-záàâãéêíóôõúüçA-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ]{3,}", text.lower())
    return [w for w in words if w not in STOP]

def _tfidf_vector(tokens: List[str], idf: Dict[str,float]) -> Dict[str,float]:
    """Cria vetor TF-IDF esparso."""
    if not tokens: return {}
    tf = Counter(tokens)
    total = len(tokens)
    vec = {}
    for t, count in tf.items():
        if t in idf:
            vec[t] = (count / total) * idf[t]
    return vec

def _cosine_sparse(a: Dict[str,float], b: Dict[str,float]) -> float:
    """Similaridade de cosseno entre dois vetores esparsos."""
    if not a or not b: return 0.0
    common = set(a) & set(b)
    if not common: return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na  = math.sqrt(sum(v*v for v in a.values()))
    nb  = math.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def _empty_store() -> dict:
    return {"docs": [], "ids": [], "srcs": [], "token_lists": [], "idf": {}}

def load_vs() -> dict:
    if not os.path.exists(VECTORS_FILE): return _empty_store()
    try:
        with open(VECTORS_FILE,"rb") as f: data = pickle.load(f)
        if not all(k in data for k in ("docs","ids","srcs","token_lists","idf")):
            return _empty_store()
        return data
    except: return _empty_store()

def save_vs(data: dict):
    with open(VECTORS_FILE,"wb") as f: pickle.dump(data, f)

def rebuild_idf(token_lists: List[List[str]]) -> Dict[str,float]:
    """Recalcula IDF completo após adicionar documentos."""
    N = len(token_lists)
    if N == 0: return {}
    df: Dict[str,int] = {}
    for tl in token_lists:
        for t in set(tl):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def make_chunks(text: str) -> List[str]:
    text = clean(text)
    if not text: return []
    result, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in [". ",".\n","! ","? ","\n\n","\n"]:
                idx = text.rfind(sep, start + CHUNK_SIZE//2, end)
                if idx != -1: end = idx + len(sep); break
        chunk = text[start:end].strip()
        if len(chunk) > 40: result.append(chunk)
        start = end - CHUNK_OVERLAP
        if start >= len(text): break
    return result

# ─── RAG SERVICE ──────────────────────────────────────────────────────────────
class RAG:
    def __init__(self):
        self.vs = load_vs()

    def add(self, filename: str, text: str) -> int:
        cks = make_chunks(text)
        added = 0
        for i, ck in enumerate(cks):
            uid = f"{filename}|{i}|{hashlib.md5(ck.encode()).hexdigest()[:8]}"
            if uid in self.vs["ids"]: continue
            tokens = _tokenize(ck)
            if not tokens: continue
            self.vs["docs"].append(ck)
            self.vs["ids"].append(uid)
            self.vs["srcs"].append(filename)
            self.vs["token_lists"].append(tokens)
            added += 1
        if added:
            self.vs["idf"] = rebuild_idf(self.vs["token_lists"])
            save_vs(self.vs)
        return added

    def search(self, query: str, k: int = TOP_K) -> Tuple[str, List[dict]]:
        if not self.vs["docs"]: return "", []
        q_tokens = _tokenize(query)
        if not q_tokens: return "", []
        q_vec = _tfidf_vector(q_tokens, self.vs["idf"])
        scores = []
        for i, tl in enumerate(self.vs["token_lists"]):
            d_vec = _tfidf_vector(tl, self.vs["idf"])
            scores.append((_cosine_sparse(q_vec, d_vec), i))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:k]
        if not top or top[0][0] == 0: return "", []
        results, parts = [], []
        for rank, (sim, idx) in enumerate(top):
            if sim < 0.01: continue
            src = self.vs["srcs"][idx]
            doc = self.vs["docs"][idx]
            results.append({"rank": rank+1, "sim": sim, "src": src, "text": doc})
            parts.append(f"[Fonte: {src} | Score: {sim:.3f}]\n{doc}")
        if not parts: return "", []
        return "\n\n---\n\n".join(parts), results

    def clear(self):
        self.vs = _empty_store(); save_vs(self.vs)

    @property
    def count(self): return len(self.vs["docs"])

    @property
    def sources(self): return list(dict.fromkeys(self.vs.get("srcs",[])))

# ─── WEB SEARCH ───────────────────────────────────────────────────────────────
def web_search(query: str, n: int = 4) -> str:
    if not DDGS_OK: return ""
    try:
        with DDGS() as d:
            res = list(d.text(query, max_results=n))
        if not res: return ""
        out = []
        for r in res:
            title = r.get("title","")
            body  = r.get("body","")[:400]
            href  = r.get("href","")
            out.append(f"• {title}\n  {body}\n  {href}")
        return "\n\n".join(out)
    except Exception as e:
        log.warning(f"Web search: {e}"); return ""

# ─── LLM ──────────────────────────────────────────────────────────────────────
class LLM:
    def __init__(self, key: str):
        self.client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1")

    def _call(self, msgs, model, temp, toks, stream=False):
        return self.client.chat.completions.create(
            model=model, messages=msgs, temperature=temp,
            max_tokens=toks, stream=stream, timeout=60,
        )

    def gen(self, msgs, model, temp=0.7, toks=1024) -> Tuple[str,str]:
        for m in ([model, FALLBACK] if model != FALLBACK else [model]):
            try:
                r = self._call(msgs, m, temp, toks)
                return r.choices[0].message.content or "", m
            except Exception as e:
                log.warning(f"Model {m}: {e}")
        return "Erro ao gerar resposta.", "error"

    def stream(self, msgs, model, temp=0.7, toks=1024):
        try:
            for chunk in self._call(msgs, model, temp, toks, stream=True):
                d = chunk.choices[0].delta
                if d and d.content: yield d.content
        except Exception as e:
            log.warning(f"Stream: {e}")
            text, _ = self.gen(msgs, model, temp, toks)
            yield text

    def vision(self, b64, prompt, mime="image/jpeg", temp=0.7, toks=1024) -> Tuple[str,str]:
        msgs = [{"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}},
            {"type":"text","text":prompt},
        ]}]
        for vm in VISION_MODELS:
            try:
                r = self._call(msgs, vm, temp, toks)
                return r.choices[0].message.content or "", vm
            except Exception as e:
                log.warning(f"Vision {vm}: {e}")
        return "Erro no modelo de visão.", "error"

# ─── TTS ──────────────────────────────────────────────────────────────────────
def tts(text: str, lang: str = "pt") -> Optional[str]:
    try:
        c = re.sub(r"[*_`#\[\](){}<>~]","", text)
        c = re.sub(r"https?://\S+","", c).strip()
        if not c: return None
        h = hashlib.md5(c[:800].encode()).hexdigest()
        p = f"data/tts/{h}.mp3"
        if not os.path.exists(p):
            gTTS(text=c[:5000], lang=lang, slow=False).save(p)
        return p
    except Exception as e:
        log.error(f"TTS: {e}"); return None

# ─── EXPORT ───────────────────────────────────────────────────────────────────
def export(msgs: List[Dict], fmt: str = "txt") -> bytes:
    if not msgs: return b""
    try:
        if fmt == "txt":
            return "\n\n".join(
                f"[{'Usuário' if m['role']=='user' else 'Moura IA'}]\n{m['content']}"
                for m in msgs
            ).encode("utf-8")
        if fmt == "md":
            lines = ["# Conversa — Moura IA\n"]
            for m in msgs:
                r = "👤 Usuário" if m["role"]=="user" else "🤖 Moura IA"
                lines.append(f"## {r}\n{m['content']}\n")
            return "\n".join(lines).encode("utf-8")
        if fmt == "docx":
            doc = Document()
            doc.add_heading("Conversa — Moura IA", 1)
            for m in msgs:
                doc.add_heading("Usuário" if m["role"]=="user" else "Moura IA", 2)
                doc.add_paragraph(m["content"]); doc.add_paragraph("")
            buf = BytesIO(); doc.save(buf); return buf.getvalue()
    except Exception as e:
        log.error(f"Export {fmt}: {e}")
    return b""

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
def build_system(rag_ctx: str, web_ctx: str) -> str:
    base = (
        "Você é Moura IA, assistente inteligente e multimodal. "
        "Responda sempre em português brasileiro, de forma clara e bem formatada. "
        "Use Markdown quando útil (listas, negrito, código, tabelas)."
    )
    if rag_ctx:
        base += (
            "\n\nCONTEXTO DOS DOCUMENTOS (use como fonte primária):\n"
            "Se a resposta estiver nos documentos, cite a fonte. "
            "Se não estiver, diga claramente que a informação não consta nos documentos.\n\n"
            + rag_ctx
        )
    if web_ctx:
        base += "\n\nINFORMAÇÕES DA WEB (use como complemento):\n" + web_ctx
    if not rag_ctx and not web_ctx:
        base += "\n\nResponda com base no seu conhecimento geral. Seja honesto quando não souber."
    return base

def fmt_model(mid: str) -> str:
    return mid.split("/")[-1] if "/" in mid else mid

# ═══════════════════════════════════════════════════════════════════════════════
# BOOT
# ═══════════════════════════════════════════════════════════════════════════════
API_KEY = get_api_key()
if not API_KEY:
    show_api_screen(); st.stop()

# Singleton services via session_state
if "db"  not in st.session_state: st.session_state.db  = DB()
if "rag" not in st.session_state:
    try: st.session_state.rag = RAG()
    except Exception as e:
        log.error(f"RAG init: {e}"); st.session_state.rag = RAG.__new__(RAG); st.session_state.rag.vs = _empty_store()
if "llm" not in st.session_state: st.session_state.llm = LLM(API_KEY)

db:  DB  = st.session_state.db
rag: RAG = st.session_state.rag
llm: LLM = st.session_state.llm

def _ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

_ss("conv_id",    None)
_ss("messages",   None)
_ss("streaming",  True)
_ss("web_on",     True)
_ss("tts_on",     False)
_ss("temp",       0.7)
_ss("maxtok",     1024)
_ss("model_name", list(TEXT_MODELS.keys())[0])
_ss("img_key",    0)
_ss("show_debug", False)

# Load / create conversation
if st.session_state.conv_id is None:
    rows = db.all("SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1")
    if rows:
        st.session_state.conv_id = rows[0]["id"]
    else:
        cur = db.ex("INSERT INTO conversations(title) VALUES('Nova Conversa')")
        st.session_state.conv_id = cur.lastrowid

if st.session_state.messages is None:
    rows = db.all(
        "SELECT role,content,image_path FROM messages WHERE conversation_id=? ORDER BY id",
        (st.session_state.conv_id,),
    )
    st.session_state.messages = []
    for r in rows:
        m = {"role": r["role"], "content": r["content"]}
        if r["image_path"]: m["image"] = r["image_path"]
        st.session_state.messages.append(m)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='sidebar-header'>
      <div class='logo'>Moura IA</div>
      <div class='tagline'>Assistente Multimodal · NVIDIA NIM</div>
    </div>""", unsafe_allow_html=True)

    # ── Conversas ────────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Conversas</div>", unsafe_allow_html=True)
    if st.button("＋  Nova conversa", use_container_width=True):
        cur = db.ex("INSERT INTO conversations(title) VALUES('Nova Conversa')")
        st.session_state.conv_id  = cur.lastrowid
        st.session_state.messages = []
        st.session_state.img_key += 1
        st.rerun()

    for c in db.all("SELECT id,title FROM conversations ORDER BY updated_at DESC LIMIT 20"):
        active = c["id"] == st.session_state.conv_id
        label  = (c["title"] or "Nova Conversa")[:38]
        prefix = "▶ " if active else "   "
        if st.button(f"{prefix}{label}", key=f"cv_{c['id']}", use_container_width=True):
            if c["id"] != st.session_state.conv_id:
                st.session_state.conv_id = c["id"]
                rows = db.all(
                    "SELECT role,content,image_path FROM messages WHERE conversation_id=? ORDER BY id",
                    (c["id"],),
                )
                st.session_state.messages = []
                for r in rows:
                    msg = {"role": r["role"], "content": r["content"]}
                    if r["image_path"]: msg["image"] = r["image_path"]
                    st.session_state.messages.append(msg)
                st.session_state.img_key += 1
                st.rerun()

    st.divider()

    # ── Modelo ───────────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Modelo</div>", unsafe_allow_html=True)
    idx = list(TEXT_MODELS.keys()).index(st.session_state.model_name) \
          if st.session_state.model_name in TEXT_MODELS else 0
    # CORREÇÃO: label não-vazio com label_visibility="collapsed"
    mn = st.selectbox("Modelo de texto", list(TEXT_MODELS.keys()),
                      index=idx, label_visibility="collapsed")
    st.session_state.model_name = mn
    MODEL_ID = TEXT_MODELS[mn]

    st.markdown("<div class='sec'>Parâmetros</div>", unsafe_allow_html=True)
    st.session_state.temp   = st.slider("Temperatura", 0.0, 1.0,
                                         st.session_state.temp, 0.05)
    st.session_state.maxtok = st.slider("Max tokens",  256, 4096,
                                         st.session_state.maxtok, 128)
    c1, c2 = st.columns(2)
    with c1: st.session_state.streaming = st.checkbox("Streaming",  st.session_state.streaming)
    with c2: st.session_state.tts_on    = st.checkbox("Voz (TTS)",  st.session_state.tts_on)
    st.session_state.web_on     = st.checkbox("🌐 Busca Web",  st.session_state.web_on)
    st.session_state.show_debug = st.checkbox("🔍 Debug RAG",  st.session_state.show_debug)

    st.divider()

    # ── Documentos RAG ───────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Documentos (RAG)</div>", unsafe_allow_html=True)

    if rag.count > 0:
        chips = "".join(f"<span class='sc'>📄 {s[:24]}</span>" for s in rag.sources[:6])
        st.markdown(f"""
        <div class='rag-stat'>
          <span class='rag-stat-lbl'>CHUNKS INDEXADOS</span>
          <span class='rag-stat-val'>{rag.count}</span>
        </div>
        <div style='margin:6px 0 10px;display:flex;flex-wrap:wrap;gap:4px'>{chips}</div>
        """, unsafe_allow_html=True)

    pdfs = st.file_uploader(
        "PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"pdf_{st.session_state.img_key}",
    )

    if pdfs:
        if st.button("⚡ Indexar PDFs", use_container_width=True):
            total, errs = 0, []
            prog = st.progress(0)
            for i, f in enumerate(pdfs):
                try:
                    reader = PdfReader(f)
                    text = "".join(
                        (page.extract_text() or "") + "\n"
                        for page in reader.pages
                    )
                    if not text.strip():
                        errs.append(f"'{f.name}': sem texto extraível")
                    else:
                        n = rag.add(f.name, text)
                        total += n
                        log.info(f"Indexado '{f.name}': {n} chunks")
                except Exception as e:
                    errs.append(f"'{f.name}': {e}")
                prog.progress((i+1)/len(pdfs))
            prog.empty()
            if total > 0:
                st.success(f"✅ {total} chunks indexados ({len(pdfs)} arquivo(s))")
                # Sem st.rerun() aqui — evita crash do file_uploader
            elif not errs:
                st.info("Nenhum chunk novo adicionado (arquivos já indexados?)")
            for e in errs:
                st.warning(e)

    if rag.count > 0:
        if st.button("🗑️ Limpar documentos", use_container_width=True):
            rag.clear()
            st.success("Documentos removidos")

    st.divider()

    # ── Exportar ─────────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Exportar conversa</div>", unsafe_allow_html=True)
    exp = [m for m in st.session_state.messages if m["role"] in ("user","assistant")]
    c1, c2, c3 = st.columns(3)
    c1.download_button("TXT",  export(exp,"txt"),  "conversa.txt",  "text/plain",      use_container_width=True)
    c2.download_button("MD",   export(exp,"md"),   "conversa.md",   "text/markdown",   use_container_width=True)
    c3.download_button("DOCX", export(exp,"docx"), "conversa.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True)

    if st.button("🗑️ Limpar conversa", use_container_width=True):
        db.ex("DELETE FROM messages WHERE conversation_id=?", (st.session_state.conv_id,))
        st.session_state.messages = []
        st.session_state.img_key += 1
        st.rerun()

    if st.button("🔑 Trocar API Key", use_container_width=True):
        for k in ["api_key","llm"]: st.session_state.pop(k, None)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ÁREA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_ID = TEXT_MODELS[st.session_state.model_name]

rb = f"<span class='bdg bg'>📚 {rag.count} chunks</span>" if rag.count > 0 else ""
wb = "<span class='bdg bt'>🌐 Web ON</span>"              if st.session_state.web_on else ""
mb = f"<span class='bdg bm'>{fmt_model(MODEL_ID)}</span>"
st.markdown(f"""
<div class='main-hdr'>
  <div class='main-title'>Moura IA</div>
  <div style='display:flex;gap:6px;flex-wrap:wrap;align-items:center'>{rb}{wb}{mb}</div>
</div>""", unsafe_allow_html=True)

# Histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image") and os.path.exists(msg["image"]):
            st.image(msg["image"], width=380)
        st.markdown(msg["content"])

# Tela vazia
if not st.session_state.messages:
    st.markdown("""
    <div class='ww'>
      <div class='wi'>🟢</div>
      <div class='wt'>Moura IA</div>
      <div class='ws'>Assistente multimodal com RAG, busca web e visão.<br>
      Carregue documentos, envie imagens ou pergunte qualquer coisa.</div>
    </div>
    <div class='cg'>
      <div class='cc'><div class='ci'>📄</div><div class='ct'>PDFs com RAG</div>
        <div class='cd'>Indexa documentos e responde com base no conteúdo</div></div>
      <div class='cc'><div class='ci'>🖼️</div><div class='ct'>Visão Computacional</div>
        <div class='cd'>Analise imagens, diagramas e screenshots</div></div>
      <div class='cc'><div class='ci'>🌐</div><div class='ct'>Busca em Tempo Real</div>
        <div class='cd'>Respostas baseadas em dados atuais da internet</div></div>
      <div class='cc'><div class='ci'>🔊</div><div class='ct'>Voz (TTS)</div>
        <div class='cd'>Ouça as respostas em português</div></div>
    </div>""", unsafe_allow_html=True)

# Input
uploaded_img = st.file_uploader(
    "Imagem",
    type=["png","jpg","jpeg","webp"],
    key=f"img_{st.session_state.img_key}",
    label_visibility="collapsed",
)
prompt = st.chat_input("Pergunte qualquer coisa…")

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════════════════
if prompt:
    # Imagem
    img_path, img_b64, img_mime = None, None, "image/jpeg"
    if uploaded_img:
        try:
            data     = uploaded_img.read()
            ext      = uploaded_img.name.rsplit(".",1)[-1].lower()
            img_mime = f"image/{'jpeg' if ext=='jpg' else ext}"
            img_b64  = base64.b64encode(data).decode()
            fname    = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
            img_path = f"data/images/{fname}"
            with open(img_path,"wb") as f: f.write(data)
        except Exception as e:
            st.error(f"Erro ao processar imagem: {e}")

    # Salvar user msg
    user_msg = {"role":"user","content":prompt}
    if img_path: user_msg["image"] = img_path
    st.session_state.messages.append(user_msg)
    db.ex(
        "INSERT INTO messages(conversation_id,role,content,image_path,model) VALUES(?,?,?,?,?)",
        (st.session_state.conv_id,"user",prompt,img_path,MODEL_ID),
    )
    if len(st.session_state.messages) == 1:
        db.ex(
            "UPDATE conversations SET title=?,updated_at=datetime('now','localtime') WHERE id=?",
            (prompt[:60].replace("\n"," "), st.session_state.conv_id),
        )

    # Exibir user
    with st.chat_message("user"):
        if img_path and os.path.exists(img_path):
            st.image(img_path, width=380)
        st.markdown(prompt)

    # Contexto
    rag_ctx, rag_results, web_ctx, badges = "", [], "", []

    if rag.count > 0 and not img_b64:
        with st.spinner("🔍 Buscando nos documentos…"):
            rag_ctx, rag_results = rag.search(prompt)
        if rag_ctx:
            badges.append(("📚 RAG", "bg"))
            log.info(f"RAG: {len(rag_results)} chunks retornados")
        else:
            log.warning("RAG: sem resultados relevantes")

    if st.session_state.web_on and not img_b64:
        with st.spinner("🌐 Buscando na web…"):
            web_ctx = web_search(prompt)
        if web_ctx: badges.append(("🌐 Web", "bt"))

    if img_b64: badges.append(("🖼️ Visão", "ba"))

    # Resposta
    with st.chat_message("assistant"):
        if badges:
            bhtml = "".join(f"<span class='bdg {cls}'>{lbl}</span>" for lbl,cls in badges)
            st.markdown(f"<div class='ctx-row'>{bhtml}</div>", unsafe_allow_html=True)

        answer, used_model = "", MODEL_ID

        if img_b64:
            with st.spinner("🧠 Analisando imagem…"):
                answer, used_model = llm.vision(
                    img_b64, prompt, img_mime,
                    st.session_state.temp, st.session_state.maxtok,
                )
            st.markdown(answer)
        else:
            system = build_system(rag_ctx, web_ctx)
            history = [{"role": m["role"],"content": m["content"]}
                       for m in st.session_state.messages]
            final = [{"role":"system","content":system}] + history

            if st.session_state.streaming:
                ph = st.empty(); full = ""
                try:
                    for chunk in llm.stream(final, MODEL_ID,
                                            st.session_state.temp, st.session_state.maxtok):
                        full += chunk; ph.markdown(full + "▌")
                    ph.markdown(full); answer = full
                except Exception as e:
                    answer = f"Erro: {e}"; ph.markdown(answer); used_model = "error"
            else:
                with st.spinner("💭 Pensando…"):
                    answer, used_model = llm.gen(final, MODEL_ID,
                                                  st.session_state.temp, st.session_state.maxtok)
                st.markdown(answer)

        st.markdown(
            f"<div class='mftr'><span class='mtag'>🤖 {fmt_model(used_model)}</span></div>",
            unsafe_allow_html=True,
        )

        # Debug RAG
        if st.session_state.show_debug and rag_results:
            with st.expander(f"🔍 Debug RAG — {len(rag_results)} chunks", expanded=False):
                for r in rag_results:
                    st.markdown(f"**#{r['rank']} · {r['src']} · score={r['sim']:.4f}**")
                    st.code(r["text"][:500], language=None)

        # TTS
        if st.session_state.tts_on and answer and used_model != "error":
            with st.spinner("🔊 Gerando áudio…"):
                ap = tts(answer)
            if ap and os.path.exists(ap):
                with open(ap,"rb") as af: st.audio(af.read(), format="audio/mp3")

        # Downloads
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ TXT",
            export([{"role":"assistant","content":answer}],"txt"),
            "resposta.txt","text/plain")
        c2.download_button("⬇️ DOCX",
            export([{"role":"assistant","content":answer}],"docx"),
            "resposta.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    # Salvar resposta
    st.session_state.messages.append({"role":"assistant","content":answer})
    db.ex(
        "INSERT INTO messages(conversation_id,role,content,model) VALUES(?,?,?,?)",
        (st.session_state.conv_id,"assistant",answer,used_model),
    )
    db.ex(
        "UPDATE conversations SET updated_at=datetime('now','localtime') WHERE id=?",
        (st.session_state.conv_id,),
    )
    st.session_state.img_key += 1

# ─── CLEANUP ──────────────────────────────────────────────────────────────────
@atexit.register
def _bye():
    if "db" in st.session_state:
        try: st.session_state.db.close()
        except: pass
