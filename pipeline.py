# -*- coding: utf-8 -*-
"""核心流程：入库（解析→分块→向量化→索引）与问答（检索→融合→重排→生成）"""
import time
from pathlib import Path
import config
import parsers
import models_api
import stores
from config import TOP_K, FUSE_TOP_M, RERANK_TOP_N, RRF_K

INDEX = stores.SearchIndex()

# ================= 入库 =================

def ingest_parts(name, dtype, recs):
    sid = stores.add_source(name, dtype)
    try:
        items = []
        for seg in recs:
            for c in parsers.chunk_text(seg["text"]):
                meta = dict(seg["meta"])
                meta["file"] = name
                items.append({"text": c, "meta": meta})
        if not items:
            raise ValueError("未解析出可用文本")
        stores.add_chunks(sid, items)
        rebuild_index()
        stores.finish_source(sid, len(items))
        return {"ok": True, "name": name, "chunks": len(items), "type": dtype}
    except Exception as e:
        stores.fail_source(sid, str(e))
        return {"ok": False, "name": name, "error": str(e)}

def ingest_file(path, name=None):
    name = name or Path(path).name
    recs, dtype = parsers.parse_file(path)
    return ingest_parts(name, dtype, recs)

def ingest_db(db_url, tables=None):
    recs = parsers.parse_db(db_url, tables)
    return ingest_parts("数据库 " + db_url.split("/")[-1], "数据库", recs)

def rebuild_index():
    chunks = stores.all_chunks()
    if not chunks:
        INDEX.build([], [], "hash")
        return
    vectors, mode = models_api.embed_texts([c["text"] for c in chunks])
    INDEX.build(chunks, vectors, mode)

# ================= 检索 =================

def _filter_sparse(sparse, ratio=None):
    """关键词路阈值过滤：丢掉分数明显偏低的候选，避免弱路污染融合结果

    ratio 为相对阈值：只保留分数不低于「本路最高分 × ratio」的候选。
    ratio<=0 时不做过滤（等价于原始的等权融合输入）。
    """
    if not sparse:
        return []
    if ratio is None:
        ratio = getattr(config, "SPARSE_MIN_RATIO", 0.0)
    if not ratio or ratio <= 0:
        return sparse
    top = max(e.get("score") or 0 for e in sparse)
    if top <= 0:
        return []
    kept = [e for e in sparse if (e.get("score") or 0) >= top * ratio]
    return kept or sparse[:1]

def _rrf(dense, sparse, k=None, w_dense=None, w_sparse=None, sp_ratio=None):
    """加权倒数排序融合（Weighted RRF）

    score(d) = w_dense / (k + rank_dense(d) + 1) + w_sparse / (k + rank_sparse(d) + 1)
    权重相等（1,1）时退化为原始 RRF。
    """
    k = RRF_K if k is None else k
    w_dense = getattr(config, "RRF_W_DENSE", 1.0) if w_dense is None else w_dense
    w_sparse = getattr(config, "RRF_W_SPARSE", 1.0) if w_sparse is None else w_sparse
    sparse = _filter_sparse(sparse, sp_ratio)
    score = {}
    for rank, e in enumerate(dense):
        score[e["id"]] = score.get(e["id"], 0.0) + w_dense / (k + rank + 1)
    for rank, e in enumerate(sparse):
        score[e["id"]] = score.get(e["id"], 0.0) + w_sparse / (k + rank + 1)
    return [{"id": i, "score": round(s, 6)} for i, s in sorted(score.items(), key=lambda x: -x[1])]

def _candidate_pool(dense, sparse, fused, limit=None):
    """重排序候选池：优先保住向量路与关键词路的头部候选，再用融合结果补齐

    原始实现直接取融合后的前 M 条；当某一路质量较差时，正确答案可能被挤出候选池，
    重排序再强也救不回来。这里改为「两路头部保底 + 融合补齐」。
    """
    limit = FUSE_TOP_M if limit is None else limit
    ids, seen = [], set()

    def add(cid):
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)

    for e in dense[:max(10, limit // 2)]:
        add(e["id"])
    for e in _filter_sparse(sparse)[:max(5, limit // 4)]:
        add(e["id"])
    for x in fused:
        if len(ids) >= limit:
            break
        add(x["id"])
    return ids[:limit]

def _full(entries, limit=10):
    info = stores.get_chunks([e["id"] for e in entries[:limit]])
    out = []
    for e in entries[:limit]:
        d = info.get(e["id"])
        if d:
            out.append({"id": e["id"], "score": e["score"], "text": d["text"], "meta": d["meta"]})
    return out

def search_stages(question, rerank_on=True):
    """四阶段检索；返回 (stages, error)"""
    if INDEX.faiss is None:
        return None, "知识库为空，请先到“知识库管理”上传文件或导入数据库"
    stages = {"dense": [], "sparse": [], "fused": [], "reranked": [], "timing": {}}
    t = time.time()
    vec = models_api.embed_query(question, INDEX.dim)
    if vec is None:
        return None, "查询失败：嵌入模式与索引不一致，请点“重建索引”"
    stages["timing"]["查询向量化"] = int((time.time() - t) * 1000)

    t = time.time()
    dense = INDEX.search_dense(vec, TOP_K)
    stages["timing"]["向量检索"] = int((time.time() - t) * 1000)

    t = time.time()
    sparse = INDEX.search_sparse(question, TOP_K)
    stages["timing"]["BM25检索"] = int((time.time() - t) * 1000)

    t = time.time()
    fused = _rrf(dense, sparse)
    stages["timing"]["RRF融合"] = int((time.time() - t) * 1000)

    pool_ids = _candidate_pool(dense, sparse, fused)
    info = stores.get_chunks(pool_ids)
    fmap = {x["id"]: x["score"] for x in fused}
    cand = [{"id": cid, "text": info[cid]["text"], "meta": info[cid]["meta"],
             "score": fmap.get(cid, 0.0)} for cid in pool_ids if cid in info]

    reranked = None
    if rerank_on:
        t = time.time()
        rr = models_api.rerank_docs(question, cand, RERANK_TOP_N)
        stages["timing"]["重排序"] = int((time.time() - t) * 1000)
        if rr:
            reranked = [{"id": d["id"], "text": d["text"], "meta": d["meta"], "score": float(s)} for d, s in rr]
    if not reranked:
        reranked = cand[:RERANK_TOP_N]

    stages["dense"] = _full(dense)
    stages["sparse"] = _full(sparse)
    stages["fused"] = _full(fused)
    stages["reranked"] = reranked
    return stages, None

# ================= 生成 =================

def _source_label(meta):
    f = meta.get("file", "未知来源")
    seg = []
    if meta.get("page"):
        seg.append(f"第{meta['page']}页")
    if meta.get("sheet"):
        seg.append(f"工作表{meta['sheet']}第{meta['row']}行")
    if meta.get("table"):
        seg.append(f"表{meta['table']}" + (f"第{meta['row']}行" if meta.get("row") else ""))
    return f + "·" + "·".join(seg) if seg else f

def build_context(reranked):
    parts, citations = [], []
    for i, d in enumerate(reranked, start=1):
        src = _source_label(d["meta"])
        parts.append(f"[{i}] 来源：{src}\n{d['text']}")
        citations.append({"no": i, "source": src, "text": d["text"][:300]})
    return "\n\n".join(parts), citations

def answer_stream(question, llm_mode="cloud", rerank_on=True):
    """生成SSE事件流"""
    stages, err = search_stages(question, rerank_on)
    if err:
        yield {"type": "error", "message": err}
        return
    reranked = stages["reranked"]
    if not reranked:
        yield {"type": "error", "message": "知识库中未找到相关内容"}
        return
    context, citations = build_context(reranked)
    yield {"type": "retrieval", "citations": citations, "timing": stages["timing"]}
    try:
        got = False
        for piece in models_api.stream_llm(question, context, llm_mode):
            got = True
            yield {"type": "delta", "text": piece}
        if not got:
            raise RuntimeError("模型无输出")
    except Exception as e:
        print("!! 大模型调用失败，降级为原文摘录：", e)
        yield {"type": "delta", "text": "（大模型接口暂不可用，以下为知识库检索到的相关原文）\n\n"}
        for c in citations[:3]:
            yield {"type": "delta", "text": f"[{c['no']}] {c['source']}\n{c['text'][:200]}\n\n"}
    yield {"type": "done", "citations": citations}
