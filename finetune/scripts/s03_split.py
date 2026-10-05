# finetune/scripts/s03_split.py
"""
数据工程 Step 3：8:1:1 划分训练/验证/测试集
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEDUP_PATH = DATA_DIR / "deduped.jsonl"


def main():
    random.seed(42)
    with open(DEDUP_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.shuffle(data)
    n = len(data)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"📁 {name}: {len(subset)} 条 → {path}")


if __name__ == "__main__":
    main()