# -*- coding: utf-8 -*-
"""
检索效果评测脚本：四种方案对比（Recall@5 / MRR）+ 各阶段耗时统计

所有数字均为本机实测，可按 README 步骤复现。

-------------------------------- 使用步骤 --------------------------------
第 1 步：导出知识库全部片段，方便你标注正确答案
    python eval_retrieval.py --list-chunks
    → 生成 data/eval/chunks_export.csv（列：片段ID、来源文件、类型、页/表/行、正文前80字）

第 2 步：生成评测集模板
    python eval_retrieval.py --template
    → 生成 data/eval/eval_set_template.csv（列：question, category, gold_chunk_ids）

第 3 步：用 Excel 打开模板，照第 1 步导出的 CSV 填 30~40 个问题
    - question         ：你向系统提的问题
    - category         ：文档 / 表格 / 数据库（可留空，留空则不分项统计）
    - gold_chunk_ids   ：能回答这个问题的片段ID，多个用 | 隔开，例如  12|15
    填好后另存为 data/eval/eval_set.csv（必须是 UTF-8 或 Excel 默认的 CSV）

第 4 步：跑评测
    python eval_retrieval.py --eval data/eval/eval_set.csv
    → 生成三个文件：
        data/eval/results_summary.csv   总体结果
        data/eval/results_detail.csv    逐题明细（哪个问题没命中、排第几）
        data/eval/timing_summary.csv    各阶段平均耗时
    → 同时在控制台打印一张可复制的汇总表格

注意：
1. 必须在 backend 目录下运行（要 import config / stores / pipeline），命令示例见上。
2. 语料只有几十个片段时 Recall@5 会虚高，建议先把知识库扩到 300 个片段以上再评测。
3. 若控制台提示「演示模式」，说明嵌入 Key 不可用，此时数字无意义，仅可用于界面演示。
4. 重排序失败会自动回退为融合排序，汇总表会标注「回退」，这种数字也不能直接当方案四写。
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# 防止某些字符在 GBK 控制台里报编码错误；不影响 Excel 打开的 CSV（那些用 utf-8-sig 写）
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# 关掉 jieba 的初始化日志，控制台只留关键信息
try:
    import logging
    import jieba
    jieba.setLogLevel(logging.WARNING)
except Exception:
    pass

import config as C
import models_api
import pipeline
import stores

SCHEMES = [
    ("方案一（仅向量检索）", "dense"),
    ("方案二（仅BM25检索）", "sparse"),
    ("方案三（向量+BM25融合）", "fused"),
    ("方案四（融合+重排序）", "rerank"),
]


# ---------------------------------------------------------------- 输出目录
def out_dir(cli_dir=None):
    d = Path(cli_dir) if cli_dir else (C.DATA_DIR / "eval")
    d.mkdir(parents=True, exist_ok=True)
    return d


def preview(text, n=80):
    t = " ".join((text or "").split())
    return t[:n]


def source_desc(meta):
    meta = meta or {}
    bits = [meta.get("file", "未知来源")]
    if meta.get("page"):
        bits.append("第%s页" % meta["page"])
    if meta.get("sheet"):
        bits.append("表%s第%s行" % (meta.get("sheet"), meta.get("row")))
    if meta.get("table"):
        bits.append("表%s第%s行" % (meta.get("table"), meta.get("row")))
    return "·".join(bits)


# ---------------------------------------------------------------- 知识库导出
def list_chunks(d):
    chunks = stores.all_chunks()
    if not chunks:
        print("!! 知识库是空的，先到网页「知识库管理」上传文件或导入示例库。")
        return
    fp = d / "chunks_export.csv"
    with fp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["片段ID", "来源", "正文前80字"])
        for c in chunks:
            w.writerow([c["id"], source_desc(c["meta"]), preview(c["text"], 80)])
    print("已导出 %d 个片段 → %s" % (len(chunks), fp))
    print("提示：标注 gold_chunk_ids 时就是填第一列的「片段ID」，可以填多个，用 | 隔开。")


# ---------------------------------------------------------------- 评测集模板
def make_template(d):
    fp = d / "eval_set_template.csv"
    if fp.exists():
        print("模板已存在，未覆盖：%s" % fp)
        return
    with fp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question", "category", "gold_chunk_ids"])
        w.writerow(["（示例）个人信息保护法规定处理个人信息应当遵循什么原则", "文档", "3|4"])
        w.writerow(["（示例）重点法规清单里共有哪几部法规", "表格", "21"])
        w.writerow(["（示例）示例数据库 laws 表里收录了哪几条法规", "数据库", "26"])
    print("已生成模板 → %s" % fp)
    print("请删除示例行，填入你自己的 30~40 个问题，另存为 eval_set.csv 后再跑 --eval。")


# ---------------------------------------------------------------- 读取评测集
def load_eval(path):
    p = Path(path)
    if not p.exists():
        print("!! 找不到评测集文件：%s" % p)
        sys.exit(1)
    rows = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            q = (row.get("question") or "").strip()
            if not q or q.startswith("（示例）"):
                continue
            gold_raw = (row.get("gold_chunk_ids") or "").replace("|", " ").replace(";", " ").replace("，", " ")
            gold = [int(x) for x in gold_raw.replace(",", " ").split() if x.strip().isdigit()]
            if not gold:
                print("  (第%d行跳过：没有填 gold_chunk_ids) %s" % (i, q[:30]))
                continue
            rows.append({"question": q, "category": (row.get("category") or "").strip() or "未分类",
                         "gold": gold, "line": i})
    return rows


# ---------------------------------------------------------------- 指标
def score_at_n(ranked_ids, gold, n):
    """返回 (是否命中, 首个命中名次 1-based；未命中为 0)"""
    for pos, cid in enumerate(ranked_ids[:n], start=1):
        if cid in gold:
            return 1, pos
    return 0, 0


def hit_at_1(ranked_ids, gold):
    return 1 if ranked_ids and ranked_ids[0] in gold else 0


# ---------------------------------------------------------------- 主评测
def run_eval(path, n, k, m, use_rerank, limit, d):
    rows = load_eval(path)
    if limit:
        rows = rows[:limit]
    if not rows:
        print("!! 评测集里没有有效数据（所有行都缺 gold_chunk_ids）。")
        sys.exit(1)

    INDEX = pipeline.INDEX
    if INDEX.faiss is None:
        print("!! 索引为空，先到网页「知识库管理」点一次「重建索引」。")
        sys.exit(1)

    kb_ids = set(INDEX.ids)
    missing = sorted({g for r in rows for g in r["gold"] if g not in kb_ids})
    print("=" * 78)
    print("知识库片段数：%d   向量维度：%s   嵌入模式：%s" % (len(INDEX.ids), INDEX.dim, INDEX.mode))
    print("评测问题数：%d   单路召回 K=%d   融合候选 M=%d   最终截取 N=%d" % (len(rows), k, m, n))
    print("融合权重：向量 %.2f / 关键词 %.2f   BM25 相对阈值 %.2f"
          % (getattr(C, "RRF_W_DENSE", 1.0), getattr(C, "RRF_W_SPARSE", 1.0),
             getattr(C, "SPARSE_MIN_RATIO", 0.0)))
    if INDEX.mode == "hash":
        print("!! 警告：当前是「演示模式」的哈希向量，检索结果没有语义能力，评测数字不可用！")
    if len(INDEX.ids) < 100:
        print("!! 提示：语料只有 %d 个片段，Recall@5 会明显偏高，建议先扩充到 300 个以上。" % len(INDEX.ids))
    if missing:
        print("!! 提示：下列标注的片段ID不在索引里（可能被删除或重建过），这些题会被判为未命中：%s" % missing[:20])
    print("=" * 78)

    hits = {s: 0 for _, s in SCHEMES}
    top1 = {s: 0 for _, s in SCHEMES}
    mrr = {s: 0.0 for _, s in SCHEMES}
    cat_stat = {}          # category -> {scheme: [hit, n]}
    timings = []           # 每题的各阶段耗时
    details = []
    rerank_fallback = 0

    # 预热：BM25 第一次调用会初始化 jieba 词典（约 1 秒），先跑一次避免污染耗时统计
    try:
        INDEX.search_sparse("预热", 1)
    except Exception:
        pass

    for i, r in enumerate(rows, start=1):
        q, gold = r["question"], r["gold"]
        t = {}

        t0 = time.perf_counter()
        vec = models_api.embed_query(q, INDEX.dim)
        t["查询向量化"] = (time.perf_counter() - t0) * 1000
        if vec is None:
            print("!! 第%d题向量化失败（嵌入维度与索引不一致），跳过：%s" % (i, q[:20]))
            continue

        t0 = time.perf_counter()
        dense = INDEX.search_dense(vec, k)
        t["向量检索"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sparse = INDEX.search_sparse(q, k)
        t["BM25检索"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = pipeline._rrf(dense, sparse)
        t["RRF融合"] = (time.perf_counter() - t0) * 1000

        ranked = {
            "dense": [e["id"] for e in dense],
            "sparse": [e["id"] for e in sparse],
            "fused": [e["id"] for e in fused],
        }

        # 方案四：候选池（两路头部保底 + 融合补齐）交给重排序
        t0 = time.perf_counter()
        reranked_ids = []
        if use_rerank:
            pool_ids = pipeline._candidate_pool(dense, sparse, fused, m)
            info = stores.get_chunks(pool_ids)
            fmap = {x["id"]: x["score"] for x in fused}
            cand = [{"id": cid, "text": info[cid]["text"], "meta": info[cid]["meta"],
                     "score": fmap.get(cid, 0.0)} for cid in pool_ids if cid in info]
            rr = models_api.rerank_docs(q, cand, n)
            if rr:
                reranked_ids = [doc["id"] for doc, _ in rr]
            else:
                reranked_ids = ranked["fused"][:n]
                rerank_fallback += 1
        t["重排序"] = (time.perf_counter() - t0) * 1000
        ranked["rerank"] = reranked_ids or ranked["fused"][:n]

        row = {"序号": i, "问题": q, "类别": r["category"], "gold": "|".join(map(str, gold))}
        for label, key in SCHEMES:
            hit, rank = score_at_n(ranked[key], gold, n)
            hits[key] += hit
            top1[key] += hit_at_1(ranked[key], gold)
            mrr[key] += (1.0 / rank) if rank else 0.0
            cs = cat_stat.setdefault(r["category"], {kk: [0, 0] for _, kk in SCHEMES})
            cs[key][0] += hit
            cs[key][1] += 1
            row[label + "·前%d" % n] = "|".join(map(str, ranked[key][:n]))
            row[label + "·命中"] = "命中(第%d)" % rank if rank else "未命中"
        details.append(row)
        timings.append(t)

    total = max(len(details), 1)
    print("\n评测完成：有效问题 %d 个\n" % len(details))

    # ---------------- 汇总表
    print("| 检索方案 | Recall@1 | Recall@%d | MRR | 命中题数 |" % n)
    print("|---|---|---|---|---|")
    summary_rows = []
    for label, key in SCHEMES:
        rec = hits[key] / total * 100
        r1 = top1[key] / total * 100
        m_ = mrr[key] / total
        print("| %s | %.1f%% | %.1f%% | %.3f | %d/%d |" % (label, r1, rec, m_, hits[key], len(details)))
        summary_rows.append([label, "%.1f%%" % r1, "%.1f%%" % rec, "%.3f" % m_,
                             "%d/%d" % (hits[key], len(details))])

    if len(cat_stat) > 1:
        print("\n分类型统计（Recall@%d / MRR）：" % n)
        head = "| 类别 | " + " | ".join(l for l, _ in SCHEMES) + " |"
        print(head)
        print("|" + "---|" * (len(SCHEMES) + 1))
        for cat, cs in cat_stat.items():
            cells = []
            for _, key in SCHEMES:
                h, cnt = cs[key]
                cells.append("%.0f%% / %.3f" % (h / max(cnt, 1) * 100, h / max(cnt, 1)))
            print("| %s | %s |" % (cat, " | ".join(cells)))

    # ---------------- 耗时
    print("\n各阶段平均耗时（ms，共 %d 题）：" % len(timings))
    timing_rows = []
    for stage in ("查询向量化", "向量检索", "BM25检索", "RRF融合", "重排序"):
        vals = [t[stage] for t in timings if stage in t]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        mx = max(vals)
        print("  %-8s 平均 %7.1f   最大 %7.1f" % (stage, avg, mx))
        timing_rows.append([stage, "%.1f" % avg, "%.1f" % mx])

    # ---------------- 落盘
    sum_fp = d / "results_summary.csv"
    with sum_fp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["检索方案", "Recall@1", "Recall@%d" % n, "MRR", "命中题数"])
        w.writerows(summary_rows)

    det_fp = d / "results_detail.csv"
    with det_fp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        w.writeheader()
        w.writerows(details)

    tim_fp = d / "timing_summary.csv"
    with tim_fp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["阶段", "平均耗时(ms)", "最大耗时(ms)"])
        w.writerows(timing_rows)

    print("\n已生成：")
    print("  总体结果→ %s" % sum_fp)
    print("  逐题明细（查漏）      → %s" % det_fp)
    print("  耗时统计（抄进 7.4）  → %s" % tim_fp)

    if rerank_fallback:
        print("\n!! 有 %d 题重排序调用失败、退回融合排序，方案四的数字偏保守，"
              "若重排序 Key 正常请重跑一次。" % rerank_fallback)



def do_check():
    """体检：嵌入、索引维度、BM25、重排序是否都正常"""
    INDEX = pipeline.INDEX
    print("=" * 70)
    print("知识库体检")
    print("=" * 70)
    n = len(INDEX.ids) if INDEX.ids else 0
    print("1) 已入库片段：%d 个" % n)
    if INDEX.faiss is None or n == 0:
        print("   !! 索引是空的，先到网页「知识库管理」点一次「重建索引」")
        return
    print("   向量维度：%s   嵌入模式：%s" % (INDEX.dim, INDEX.mode))
    if INDEX.mode == "hash":
        print("   !! 演示模式：云端嵌入不可用，评测数字不可用")

    print("\n2) 嵌入接口：", end="")
    t0 = time.time()
    vec = models_api.embed_query("测试一下嵌入是否可用", INDEX.dim)
    if vec is None:
        print("失败（维度不一致或接口异常）")
    else:
        print("正常，耗时 %.0f ms，维度 %d" % ((time.time() - t0) * 1000, len(vec)))

    print("\n3) BM25 分词与检索：", end="")
    try:
        hits = INDEX.search_sparse("个人信息保护", 3)
        print("正常，命中 %d 条" % len(hits))
    except Exception as e:
        print("失败：%s" % e)

    print("\n4) 重排序接口：", end="")
    t0 = time.time()
    docs = [{"id": 1, "text": "个人信息保护法规定处理个人信息应当遵循合法、正当、必要原则。", "meta": {}},
            {"id": 2, "text": "今天天气晴朗，适合户外运动。", "meta": {}}]
    rr = models_api.rerank_docs("处理个人信息应遵循什么原则", docs, 2)
    if rr:
        print("正常，耗时 %.0f ms，返回 %d 条（%.3f / %.3f）"
              % ((time.time() - t0) * 1000, len(rr), rr[0][1], rr[1][1]))
    else:
        print("未启用或调用失败 → 方案四会回退为融合排序（数字偏保守）")

    print("\n" + "=" * 70)
    print("体检结束：上面 2、4 都正常，就可以放心跑评测。")


def main():
    ap = argparse.ArgumentParser(description="RAG 系统检索效果评测")
    ap.add_argument("--eval", help="评测集 CSV 路径")
    ap.add_argument("--list-chunks", action="store_true", help="导出全部片段，供标注用")
    ap.add_argument("--template", action="store_true", help="生成评测集模板")
    ap.add_argument("--check", action="store_true", help="体检：嵌入/BM25/重排序是否可用")
    ap.add_argument("--k", type=int, default=C.TOP_K, help="单路召回数量，默认取 config.TOP_K")
    ap.add_argument("--m", type=int, default=C.FUSE_TOP_M, help="融合后进入重排序的候选数")
    ap.add_argument("--n", type=int, default=C.RERANK_TOP_N, help="最终截取条数（默认5）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--no-rerank", action="store_true", help="跳过重排序（省接口调用）")
    ap.add_argument("--w-dense", type=float, default=None, help="融合中向量路权重，默认取 config.RRF_W_DENSE")
    ap.add_argument("--w-sparse", type=float, default=None, help="融合中关键词路权重，默认取 config.RRF_W_SPARSE")
    ap.add_argument("--sparse-ratio", type=float, default=None,
                    help="BM25 相对阈值（0 表示不过滤），默认取 config.SPARSE_MIN_RATIO")
    ap.add_argument("--out", help="输出目录，默认 data/eval")
    args = ap.parse_args()

    if args.w_dense is not None:
        C.RRF_W_DENSE = args.w_dense
    if args.w_sparse is not None:
        C.RRF_W_SPARSE = args.w_sparse
    if args.sparse_ratio is not None:
        C.SPARSE_MIN_RATIO = args.sparse_ratio

    d = out_dir(args.out)
    if args.check:
        do_check()
        return
    if args.list_chunks:
        list_chunks(d)
        return
    if args.template:
        make_template(d)
        return
    if not args.eval:
        ap.print_help()
        print("\n先 --check 体检，再 --eval 评测。")
        return
    run_eval(args.eval, args.n, args.k, args.m, not args.no_rerank, args.limit, d)


if __name__ == "__main__":
    main()
