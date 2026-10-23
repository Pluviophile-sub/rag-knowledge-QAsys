# -*- coding: utf-8 -*-
"""配置模板。复制为 config.py 后填入自己的 Key；两个 Key 都不填也能跑，会自动降级。"""
import os
from pathlib import Path

# ========== 1. API Key（留空则自动降级，不会报错） ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")          # 生成答案用；留空 → 回答降级为原文摘录
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")    # 嵌入 + 重排序用；留空 → 嵌入降级为演示模式

# ========== 2. 服务地址与模型 ==========
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434/v1"    # 本地模型（Ollama），装了才能用
LOCAL_LLM_MODEL = "qwen3:4b"

# ========== 3. 检索参数 ==========
TOP_K = 20           # 单路检索返回数量 K
FUSE_TOP_M = 30      # 融合后进入重排序数量 M
RERANK_TOP_N = 5     # 送入大模型的上下文数量 N
RRF_K = 60           # RRF 平滑常数
RERANK_ENABLED = True

# ========== 4. 路径（跟随项目文件夹，无需修改） ==========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = DATA_DIR / "sample"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "knowledge.db"
DEMO_DB_PATH = DATA_DIR / "sample" / "sample_laws.db"
DB_URL = "sqlite:///" + DEMO_DB_PATH.as_posix()

# ========== 5. 提示词模板 ==========
PROMPT_SYSTEM = (
    "你是一个严谨的知识库问答助手。请严格依据提供的资料回答问题，"
    "不得引入资料之外的信息；如果资料不足以回答，请明确说明“知识库中未找到相关内容”。"
    "回答中引用资料时，请在对应内容后标注资料编号，如[1]。"
)

PROMPT_USER_TEMPLATE = (
    "以下是知识库检索到的资料：\n\n{context}\n\n"
    "请根据以上资料回答用户问题：{question}"
)

# ========== 6. 融合策略（加权 RRF） ==========
RRF_W_DENSE = 0.75       # 融合中向量路权重
RRF_W_SPARSE = 0.25      # 融合中关键词路权重
SPARSE_MIN_RATIO = 0.5   # BM25 候选相对阈值：只保留不低于本路最高分*该比例的候选（0=不过滤）
