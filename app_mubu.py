from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import AppPaths, QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL, ensure_dirs
from core_rag import RAGIndex

ensure_dirs()
paths = AppPaths()
rag = RAGIndex(paths)
app = FastAPI(title="RAG知识库", version="1.0.0")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    source: str
    contexts: list[dict] = Field(default_factory=list)


class BuildResponse(BaseModel):
    document_count: int
    message: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG知识库</title>
  <style>
    body{margin:0;font-family:Inter,Arial,sans-serif;background:linear-gradient(180deg,#eef4ff,#f8fafc);color:#0f172a}
    .wrap{max-width:1000px;margin:0 auto;padding:24px}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:20px;box-shadow:0 20px 40px rgba(15,23,42,.08);overflow:hidden}
    .header{padding:20px 24px;border-bottom:1px solid #e5e7eb}
    .header h1{margin:0;font-size:24px}
    .header p{margin:8px 0 0;color:#64748b}
    .chat{height:68vh;overflow:auto;padding:24px;display:flex;flex-direction:column;gap:14px}
    .msg{max-width:82%;padding:14px 16px;border-radius:16px;line-height:1.65;white-space:pre-wrap}
    .user{align-self:flex-end;background:#2563eb;color:#fff;border-bottom-right-radius:6px}
    .bot{align-self:flex-start;background:#f1f5f9;color:#111827;border-bottom-left-radius:6px}
    .input{display:flex;gap:12px;padding:18px 24px;border-top:1px solid #e5e7eb;background:#fff}
    textarea{flex:1;resize:none;border:1px solid #cbd5e1;border-radius:14px;padding:14px 16px;font-size:15px;min-height:56px;outline:none}
    button{border:none;background:linear-gradient(135deg,#2563eb,#7c3aed);color:#fff;padding:0 20px;border-radius:14px;font-size:15px;min-width:110px;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    .meta{font-size:12px;color:#64748b;margin-top:8px}
    .toolbar{display:flex;gap:10px;padding:0 24px 18px}
    .ghost{background:#e2e8f0;color:#0f172a}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <h1>RAG知识库</h1>
        <p>支持标题结构化切片、BM25 + 向量混合检索、千问模型回答。</p>
      </div>
      <div id="chat" class="chat"></div>
      <div class="toolbar">
        <button class="ghost" id="build">重建索引</button>
      </div>
      <div class="input">
        <textarea id="msg" placeholder="请输入问题，例如：什么是假设检验？"></textarea>
        <button id="send">发送</button>
      </div>
    </div>
  </div>
<script>
const chat=document.getElementById('chat');
const msg=document.getElementById('msg');
const send=document.getElementById('send');
const build=document.getElementById('build');

function escapeHtml(text) {
  return text.replace(/[&<>\"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;','\'':'&#39;'}[ch]));
}

function addMessage(text, cls, meta='') {
  const el=document.createElement('div');
  el.className=`msg ${cls}`;
  el.innerHTML=`<div>${escapeHtml(text).replace(/\\n/g,'<br>')}</div>${meta?`<div class="meta">${escapeHtml(meta)}</div>`:''}`;
  chat.appendChild(el);
  chat.scrollTop=chat.scrollHeight;
}

async function postJson(url, body){
  const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json().catch(()=>({}));
  if(!res.ok){
    throw new Error(data.detail || data.message || '请求失败');
  }
  return data;
}

async function handleSend(){
  const text=msg.value.trim();
  if(!text) return;
  addMessage(text,'user');
  msg.value='';
  send.disabled=true;
  const think=document.createElement('div');
  think.className='msg bot';
  think.textContent='思考中...';
  chat.appendChild(think);
  chat.scrollTop=chat.scrollHeight;
  try{
    const data=await postJson('/api/chat',{message:text});
    think.remove();
    const meta=data.source==='knowledge_base'
      ? `来源：知识库命中 ${data.contexts.map(x=>x.title_path).join('；')}`
      : '未命中知识库或当前未配置千问 API';
    addMessage(data.answer,'bot',meta);
  }catch(e){
    think.remove();
    addMessage(e.message || '请求失败，请稍后重试。','bot');
  }finally{
    send.disabled=false;
  }
}

send.addEventListener('click',handleSend);
msg.addEventListener('keydown',(e)=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleSend();}});
build.addEventListener('click',async()=>{
  build.disabled=true;
  const think=document.createElement('div');
  think.className='msg bot';
  think.textContent='正在重建索引...';
  chat.appendChild(think);
  chat.scrollTop=chat.scrollHeight;
  try{
    const res=await fetch('/api/rebuild',{method:'POST'});
    const data=await res.json().catch(()=>({}));
    think.remove();
    if(!res.ok){
      addMessage(data.detail || data.message || '重建索引失败，请检查后端日志。','bot');
      return;
    }
    addMessage(`索引重建完成，共处理 ${data.document_count} 个切片。`,'bot');
  }catch(e){
    think.remove();
    addMessage(e.message || '重建索引失败，请检查文档和依赖。','bot');
  }finally{
    build.disabled=false;
  }
});

addMessage('你好，我是RAG知识库助手。你可以直接提问，也可以先点击“重建索引”。','bot');
</script>
</body>
</html>
"""


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/rebuild", response_model=BuildResponse)
def rebuild() -> BuildResponse:
    try:
        count = rag.build(paths.doc_dir)
        return BuildResponse(document_count=count, message="索引重建成功")
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc


@app.get("/api/config")
def config_info() -> dict:
    return {
        "qwen_api_key_configured": bool(QWEN_API_KEY),
        "qwen_base_url": QWEN_BASE_URL,
        "qwen_model": QWEN_MODEL,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_api(payload: ChatRequest) -> ChatResponse:
    try:
        result = rag.answer(payload.message)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"聊天请求失败: {exc}") from exc
