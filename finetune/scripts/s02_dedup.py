# finetune/scripts/s02_dedup.py
"""
数据工程 Step 2：去重
  1. N-gram 精确去重
  2. 语义去重（BGE-M3 向量相似度）
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "raw.jsonl"
DEDUP_PATH = DATA_DIR / "deduped.jsonl"

SEMANTIC_THRESHOLD = 0.92
NGRAM_N = 3


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ngram_dedup(data, n=NGRAM_N):
    seen = set()
    result = []
    for item in data:
        text = item["input"]
        ngrams = frozenset(
            tuple(text[i:i + n]) for i in range(max(len(text) - n + 1, 1))
        )
        if ngrams and ngrams not in seen:
            seen.add(ngrams)
            result.append(item)
    return result


def semantic_dedup(data, model: SentenceTransformer):
    texts = [item["input"] for item in data]
    embeddings = model.encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )

    keep = [True] * len(data)
    for i in range(len(data)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(data)):
            if not keep[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= SEMANTIC_THRESHOLD:
                keep[j] = False

    return [item for i, item in enumerate(data) if keep[i]]


def main():
    data = load_jsonl(RAW_PATH)
    print(f"📂 原始数据：{len(data)} 条")

    data = ngram_dedup(data)
    print(f"🔤 N-gram 去重后：{len(data)} 条")

    print("📦 加载 BGE-M3 模型...")
    model = SentenceTransformer("BAAI/bge-m3", cache_folder="./models")
    data = semantic_dedup(data, model)
    print(f"🧠 语义去重后：{len(data)} 条")

    with open(DEDUP_PATH, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 去重完成 → {DEDUP_PATH}")


if __name__ == "__main__":
    main()