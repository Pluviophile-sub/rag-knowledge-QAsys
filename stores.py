# -*- coding: utf-8 -*-
"""数据存取：SQLite元数据 + FAISS向量索引 + BM25关键词索引"""
import json
import pickle
import sqlite3
import time
import numpy as np
import faiss
import jieba
from rank_bm25 import BM25Okapi
from config import DB_PATH, INDEX_DIR

def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS sources(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, type TEXT,
        status TEXT, chunk_count INTEGER, created_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER,
        text TEXT, meta TEXT)""")
    return c

def add_source(name, type_):
    c = _conn()
    cur = c.execute("INSERT INTO sources(name,type,status,chunk_count,created_at) VALUES(?,?,?,?,?)",
                    (name, type_, "处理中", 0, time.strftime("%Y-%m-%d %H:%M:%S")))
    c.commit()
    sid = cur.lastrowid
    c.close()
    return sid

def finish_source(sid, count):
    c = _conn()
    c.execute("UPDATE sources SET status='已完成', chunk_count=? WHERE id=?", (count, sid))
    c.commit(); c.close()

def fail_source(sid, reason=""):
    c = _conn()
    c.execute("UPDATE sources SET status=? WHERE id=?", ("失败:" + str(reason)[:80], sid))
    c.commit(); c.close()

def list_sources():
    c = _conn()
    rows = c.execute("SELECT id,name,type,status,chunk_count,created_at FROM sources ORDER BY id DESC").fetchall()
    c.close()
    return [{"id": r[0], "name": r[1], "type": r[2], "status": r[3],
             "chunk_count": r[4], "created_at": r[5]} for r in rows]

def delete_source(sid):
    c = _conn()
    c.execute("DELETE FROM chunks WHERE source_id=?", (sid,))
    c.execute("DELETE FROM sources WHERE id=?", (sid,))
    c.commit(); c.close()

def add_chunks(sid, items):
    c = _conn()
    for it in items:
        c.execute("INSERT INTO chunks(source_id,text,meta) VALUES(?,?,?)",
                  (sid, it["text"], json.dumps(it["meta"], ensure_ascii=False)))
    c.commit(); c.close()

def all_chunks():
    c = _conn()
    rows = c.execute("SELECT id,text,meta FROM chunks ORDER BY id").fetchall()
    c.close()
    return [{"id": r[0], "text": r[1], "meta": json.loads(r[2])} for r in rows]

def get_chunks(ids):
    c = _conn()
    out = {}
    for i in ids:
        r = c.execute("SELECT id,text,meta FROM chunks WHERE id=?", (i,)).fetchone()
        if r:
            out[r[0]] = {"id": r[0], "text": r[1], "meta": json.loads(r[2])}
    c.close()
    return out

class SearchIndex:
    """FAISS + BM25 索引，与磁盘文件同步"""
    def __init__(self):
        self.faiss = None
        self.ids = []
        self.bm25 = None
        self.tokenized = []
        self.dim = None
        self.mode = None
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        fp = INDEX_DIR / "faiss.index"
        mp = INDEX_DIR / "index_meta.json"
        bp = INDEX_DIR / "bm25.pkl"
        if fp.exists() and mp.exists():
            try:
                buf = np.frombuffer(fp.read_bytes(), dtype="uint8").copy()
                self.faiss = faiss.deserialize_index(buf)
                meta = json.loads(mp.read_text(encoding="utf-8"))
                self.ids = meta.get("ids", [])
                self.dim = meta.get("dim")
                self.mode = meta.get("mode")
            except Exception as e:
                print("!! 索引加载失败：", e)
        if bp.exists():
            try:
                data = pickle.loads(bp.read_bytes())
                self.bm25 = data.get("bm25")
                self.tokenized = data.get("tokenized", [])
            except Exception as e:
                print("!! BM25加载失败：", e)

    def save(self):
        fp = INDEX_DIR / "faiss.index"
        if self.faiss is None:
            if fp.exists():
                fp.unlink()
        else:
            fp.write_bytes(faiss.serialize_index(self.faiss).tobytes())
        (INDEX_DIR / "index_meta.json").write_text(
            json.dumps({"ids": self.ids, "dim": self.dim, "mode": self.mode}), encoding="utf-8")
        bp = INDEX_DIR / "bm25.pkl"
        if self.bm25 is None:
            if bp.exists():
                bp.unlink()
        else:
            bp.write_bytes(pickle.dumps({"bm25": self.bm25, "tokenized": self.tokenized}))

    def build(self, chunks, vectors, mode):
        if not chunks:
            self.faiss = None; self.ids = []; self.bm25 = None
            self.tokenized = []; self.dim = None; self.mode = mode
            self.save(); return
        mat = np.array(vectors, dtype="float32")
        faiss.normalize_L2(mat)
        self.faiss = faiss.IndexFlatIP(mat.shape[1])
        self.faiss.add(mat)
        self.ids = [c["id"] for c in chunks]
        self.dim = mat.shape[1]
        self.mode = mode
        self.tokenized = [list(jieba.cut(c["text"])) for c in chunks]
        try:
            self.bm25 = BM25Okapi(self.tokenized)
        except Exception as e:
            print("!! BM25构建失败：", e)
            self.bm25 = None
        self.save()

    def search_dense(self, vec, k):
        if self.faiss is None or not self.ids:
            return []
        q = np.array([vec], dtype="float32")
        faiss.normalize_L2(q)
        D, I = self.faiss.search(q, min(k, len(self.ids)))
        return [{"id": self.ids[pos], "score": float(score)}
                for score, pos in zip(D[0], I[0]) if pos >= 0]

    def search_sparse(self, question, k):
        if self.bm25 is None or not self.ids:
            return []
        try:
            scores = self.bm25.get_scores(list(jieba.cut(question)))
        except Exception:
            return []
        order = np.argsort(scores)[::-1][:k]
        return [{"id": self.ids[i], "score": float(scores[i])} for i in order if scores[i] > 0]