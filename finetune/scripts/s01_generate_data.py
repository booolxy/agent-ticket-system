# finetune/scripts/s01_generate_data.py
"""
数据工程 Step 1：种子扩展
手写种子 → 通过 LLM 生成 15 倍变体 → 落盘 raw.jsonl
"""
import json
import os
from pathlib import Path
from langchain_openai import ChatOpenAI

from app.core.config import settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED_PATH = DATA_DIR / "seeds.jsonl"
RAW_PATH = DATA_DIR / "raw.jsonl"

EXPAND_PROMPT = """你是一个数据增强专家。请参考下面的种子样本，生成 15 个语义相同但表达不同的变体。
要求：
1. 保持 task 和 output 标签不变
2. 只改 input 的表达方式（口语化、简写、错别字、方言）
3. 严格输出 JSON 数组，每个元素包含 task、input、output 三个字段
4. 不要输出任何解释

种子样本：
{seed}
"""


def load_seeds():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def expand_seed(llm, seed: dict, n: int = 15) -> list:
    prompt = EXPAND_PROMPT.format(seed=json.dumps(seed, ensure_ascii=False))
    raw = llm.invoke(prompt).content.strip()

    # 去掉 markdown 代码块标记
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        variants = json.loads(raw)
        # 过滤掉结构不完整的
        return [
            v for v in variants[:n]
            if isinstance(v, dict) and "task" in v and "input" in v and "output" in v
        ]
    except json.JSONDecodeError:
        print(f"   ⚠️ 解析失败，跳过种子：{seed['input'][:20]}")
        return []


def main():
    llm = ChatOpenAI(
        model=settings.LARGE_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.8,
    )

    seeds = load_seeds()
    print(f"📂 加载种子 {len(seeds)} 条")

    all_data = []
    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] 扩展：{seed['input'][:20]}...")
        variants = expand_seed(llm, seed)
        all_data.append(seed)
        all_data.extend(variants)

    # 写入 raw.jsonl
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 扩展完成，共 {len(all_data)} 条数据 → {RAW_PATH}")


if __name__ == "__main__":
    main()