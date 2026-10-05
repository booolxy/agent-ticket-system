# finetune/scripts/s04_train_qlora.py
"""
QLoRA 微调 Qwen-1.8B
  1. 构造困难负样本
  2. 调用 LLaMA-Factory 训练
"""
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = BASE_DIR / "configs" / "qlora_qwen1.8b.yaml"


def build_hard_negatives(train_path: Path, output_path: Path):
    """
    困难负样本：对每条正样本，混入 1 条"语义相近但标签不同"的样本
    目的：防止模型只学到表面词汇
    """
    with open(train_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    by_task = {}
    for item in data:
        by_task.setdefault(item["task"], []).append(item)

    augmented = list(data)
    for item in data:
        for other_task, pool in by_task.items():
            if other_task == item["task"] or not pool:
                continue
            neg = pool[0]
            augmented.append({
                "task": item["task"],
                "input": neg["input"],
                "output": item["output"],
                "is_hard_negative": True,
            })
            break

    with open(output_path, "w", encoding="utf-8") as f:
        for item in augmented:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 困难负样本构造完成：{len(data)} → {len(augmented)} 条")


def main():
    train_path = DATA_DIR / "train.jsonl"
    augmented_path = DATA_DIR / "train_augmented.jsonl"

    build_hard_negatives(train_path, augmented_path)

    cmd = ["llamafactory-cli", "train", str(CONFIG_PATH)]
    print(f"🚀 开始 QLoRA 微调...\n命令：{' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"✅ 微调完成，LoRA adapter → {BASE_DIR / 'output' / 'qwen1.8b-lora'}")


if __name__ == "__main__":
    main()