# -*- coding: utf-8 -*-
"""多源数据解析：PDF / Word / Excel / CSV / 数据库 → 统一片段"""
import re
try:
    import pymupdf as fitz          # 新版 PyMuPDF
except ImportError:
    import fitz                     # 旧版 PyMuPDF
import docx
import pandas as pd

ARTICLE_RE = re.compile(r'(?=第[一二三四五六七八九十百零〇\d]{1,6}条)')

def clean_text(t):
    t = t.replace('\u3000', ' ')
    lines = [ln.strip() for ln in t.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def _sliding(text, size, overlap):
    out, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
        start += size - overlap
    return out

def chunk_text(text, size=300, overlap=50):
    """分块：法律条文按条切分，其余按滑窗切分"""
    text = clean_text(text)
    chunks = []
    if len(ARTICLE_RE.split(text)) > 3:
        for seg in ARTICLE_RE.split(text):
            seg = seg.strip()
            if not seg:
                continue
            if len(seg) <= size:
                chunks.append(seg)
            else:
                chunks.extend(_sliding(seg, size, overlap))
    else:
        chunks = _sliding(text, size, overlap)
    return [c for c in chunks if c.strip()]

def parse_pdf(path):
    doc = fitz.open(path)
    recs = []
    for i, page in enumerate(doc):
        recs.append({"text": page.get_text(), "meta": {"page": i + 1}})
    doc.close()
    return recs

def parse_docx(path):
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for tb in d.tables:
        for row in tb.rows:
            parts.append("；".join([c.text.strip() for c in row.cells]))
    return [{"text": "\n".join(parts), "meta": {}}]

def _rows_to_recs(df, sheet):
    recs = []
    cols = [str(c) for c in df.columns]
    for idx, row in df.iterrows():
        parts = [f"{c}：{row[c]}" for c in cols if str(row[c]).strip()]
        txt = f"（工作表：{sheet}）" + "；".join(parts)
        recs.append({"text": txt, "meta": {"sheet": sheet, "row": int(idx) + 2}})
    return recs

def parse_excel(path):
    recs = []
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path).fillna("")
        return _rows_to_recs(df, "CSV")
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet).fillna("")
        recs.extend(_rows_to_recs(df, sheet))
    return recs

def parse_db(db_url, tables=None):
    from sqlalchemy import create_engine, inspect, text as sqlt
    eng = create_engine(db_url)
    insp = inspect(eng)
    names = tables or insp.get_table_names()
    recs = []
    with eng.connect() as conn:
        for tb in names:
            cols = insp.get_columns(tb)
            schema = "；".join([f"{c['name']}（{c['type']}）" for c in cols])
            recs.append({"text": f"数据表 {tb}，包含字段：{schema}", "meta": {"table": tb, "row": 0}})
            rows = conn.execute(sqlt(f'SELECT * FROM "{tb}"')).fetchall()
            for i, row in enumerate(rows):
                txt = "；".join([f"{cols[j]['name']}：{row[j]}" for j in range(len(row))])
                recs.append({"text": f"数据表 {tb} 第{i+1}行：{txt}", "meta": {"table": tb, "row": i + 1}})
    return recs

def parse_file(path):
    """按扩展名分发；返回 (片段列表, 数据源类型)"""
    low = path.lower()
    if low.endswith(".pdf"):
        return parse_pdf(path), "文档(PDF)"
    if low.endswith(".docx"):
        return parse_docx(path), "文档(Word)"
    if low.endswith((".xlsx", ".xls")):
        return parse_excel(path), "表格(Excel)"
    if low.endswith(".csv"):
        return parse_excel(path), "表格(CSV)"
    if low.endswith((".txt", ".md")):
        with open(path, encoding="utf-8") as f:
            return [{"text": f.read(), "meta": {}}], "文档(文本)"
    raise ValueError("暂不支持的文件类型：" + path)