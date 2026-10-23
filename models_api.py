# -*- coding: utf-8 -*-
"""模型调用：嵌入 / 重排序 / 大语言模型（全部带降级处理，嵌入与重排序带重试）"""
import hashlib
import math
import time
import requests
import config as C

def _post_json(url, headers, payload, timeout=60, retries=3, backoff=1.5):
    """带重试的 POST：网络抖动或限流时自动退避重试，仍失败才抛出"""
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
    raise last

def cloud_embed(texts):
    out = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        data = _post_json(C.SILICONFLOW_BASE_URL.rstrip("/") + "/embeddings",
                          {"Authorization": "Bearer " + C.SILICONFLOW_API_KEY},
                          {"model": C.EMBED_MODEL, "input": batch},
                          timeout=60, retries=3, backoff=1.5)["data"]
        data.sort(key=lambda x: x.get("index", 0))
        out.extend([d["embedding"] for d in data])
    return out

def hash_embed(texts, dim=256):
    """降级嵌入：字符二元组哈希，仅用于无Key时的界面演示"""
    out = []
    for t in texts:
        v = [0.0] * dim
        for i in range(max(len(t) - 1, 1)):
            g = t[i:i + 2]
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16) % dim
            v[h] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out

def embed_texts(texts):
    """批量嵌入；返回 (向量列表, 模式 cloud/hash)"""
    if C.SILICONFLOW_API_KEY:
        try:
            return cloud_embed(texts), "cloud"
        except Exception as e:
            print("!! 云端嵌入失败，已降级为演示模式：", e)
    return hash_embed(texts), "hash"

def embed_query(question, index_dim):
    """查询向量：确保维度与索引一致，不一致自动尝试另一种模式"""
    if C.SILICONFLOW_API_KEY:
        try:
            v = cloud_embed([question])[0]
            if len(v) == index_dim:
                return v
        except Exception as e:
            print("云端嵌入失败：", e)
    v = hash_embed([question])[0]
    if len(v) == index_dim:
        return v
    return None

def rerank_docs(question, docs, top_n):
    """重排序；失败返回 None（上层回退为融合排序）"""
    if not (C.RERANK_ENABLED and C.SILICONFLOW_API_KEY and docs):
        return None
    try:
        res = _post_json(C.SILICONFLOW_BASE_URL.rstrip("/") + "/rerank",
                         {"Authorization": "Bearer " + C.SILICONFLOW_API_KEY},
                         {"model": C.RERANK_MODEL, "query": question,
                          "documents": [d["text"] for d in docs],
                          "top_n": min(top_n, len(docs))},
                         timeout=60, retries=3, backoff=1.5)["results"]
        res.sort(key=lambda x: -x["relevance_score"])
        return [(docs[item["index"]], float(item["relevance_score"])) for item in res]
    except Exception as e:
        print("!! 重排序失败，回退融合排序：", e)
        return None

def stream_llm(question, context, llm_mode="cloud"):
    """流式生成；异常由上层捕获并降级"""
    from openai import OpenAI
    if llm_mode == "local":
        client = OpenAI(api_key="ollama", base_url=C.LOCAL_LLM_BASE_URL)
        model = C.LOCAL_LLM_MODEL
    else:
        if not C.DEEPSEEK_API_KEY:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        client = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.DEEPSEEK_BASE_URL)
        model = C.DEEPSEEK_MODEL
    messages = [
        {"role": "system", "content": C.PROMPT_SYSTEM},
        {"role": "user", "content": C.PROMPT_USER_TEMPLATE.format(context=context, question=question)},
    ]
    stream = client.chat.completions.create(model=model, messages=messages,
                                            stream=True, temperature=0.3)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
