# -*- coding: utf-8 -*-
"""系统入口：启动后浏览器访问 http://127.0.0.1:8000"""
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import StreamingResponse, FileResponse
import config
import pipeline
import stores

app = FastAPI(title="RAG 多源知识库问答系统")
WEB_DIR = Path(__file__).resolve().parent / "web"

def sse(evt):
    return "data: " + json.dumps(evt, ensure_ascii=False) + "\n\n"

@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")

@app.post("/api/kb/files")
async def upload_files(files: list[UploadFile] = File(...)):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        dest = config.UPLOAD_DIR / f.filename
        with open(dest, "wb") as w:
            shutil.copyfileobj(f.file, w)
        results.append(pipeline.ingest_file(str(dest)))
    return {"results": results}

@app.get("/api/kb/sources")
def get_sources():
    return {"sources": stores.list_sources()}

@app.delete("/api/kb/sources/{sid}")
def del_source(sid: int):
    stores.delete_source(sid)
    pipeline.rebuild_index()
    return {"ok": True}

@app.post("/api/kb/rebuild")
def rebuild():
    pipeline.rebuild_index()
    return {"ok": True, "chunks": len(stores.all_chunks())}

@app.post("/api/kb/import-db")
def import_db(payload: dict = Body(default={})):
    url = payload.get("db_url") or config.DB_URL
    return pipeline.ingest_db(url)

@app.post("/api/search")
def search(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    if not q:
        return {"error": "问题为空"}
    stages, err = pipeline.search_stages(q, payload.get("rerank", True))
    if err:
        return {"error": err}
    return stages

@app.post("/api/chat/stream")
def chat_stream(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    def gen():
        if not q:
            yield sse({"type": "error", "message": "问题为空"})
            return
        for evt in pipeline.answer_stream(q, payload.get("llm", "cloud"), payload.get("rerank", True)):
            yield sse(evt)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/config")
def get_config():
    return {"embed_model": config.EMBED_MODEL,
            "llm_model": config.DEEPSEEK_MODEL,
            "rerank_enabled": config.RERANK_ENABLED,
            "deepseek_key": bool(config.DEEPSEEK_API_KEY),
            "siliconflow_key": bool(config.SILICONFLOW_API_KEY)}

if __name__ == "__main__":
    n = len(pipeline.INDEX.ids)
    print("=" * 56)
    print("系统启动中…… 浏览器访问： http://127.0.0.1:8000")
    print("当前索引片段数：", n)
    print("DeepSeek Key：", "已配置" if config.DEEPSEEK_API_KEY else "未配置（回答降级为原文摘录）")
    print("硅基流动 Key：", "已配置" if config.SILICONFLOW_API_KEY else "未配置（嵌入为演示模式）")
    print("=" * 56)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)