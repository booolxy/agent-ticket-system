# finetune/scripts/s05_eval_finetuned.py
"""
微调后四维评估：格式正确率 / 质量分 / 耗时 / Token 成本
对比：基座 vs 微调后
"""
import json
import time
from pathlib import Path

import torch
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_openai import ChatOpenAI

from app.core.config import settings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_PATH = DATA_DIR / "test.jsonl"
BASE_MODEL_PATH = BASE_DIR.parent / "models" / "Qwen" / "Qwen-1.8B"
ADAPTER_PATH = BASE_DIR / "output" / "qwen1.8b-lora"


class IntentOutput(BaseModel):
    """意图识别输出格式校验"""
    intent: str


def load_test(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(base_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def inference(model, tokenizer, prompt, max_new_tokens=16):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def evaluate(model, tokenizer, test_data, judge_llm, name="model"):
    format_pass, quality_scores, latencies, tokens = 0, [], [], 0

    for item in test_data:
        prompt = (
            f"<|im_start|>user\n{item['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        start = time.perf_counter()
        output = inference(model, tokenizer, prompt)
        latencies.append(time.perf_counter() - start)

        # 1. 格式校验
        try:
            IntentOutput(intent=output)
            format_pass += 1
        except ValidationError:
            pass

        # 2. GPT-4 质量分
        judge_prompt = (
            f"用户输入：{item['input']}\n模型输出：{output}\n"
            f"请打分 1-10，只输出数字。"
        )
        try:
            score = float(judge_llm.invoke(judge_prompt).content.strip().split()[0])
            quality_scores.append(min(max(score, 1), 10))
        except Exception:
            quality_scores.append(0)

        # 3. Token 成本
        tokens += int((len(item["input"]) + len(output)) * 1.5)

    return {
        "name": name,
        "format_pass_rate": format_pass / len(test_data),
        "avg_quality": sum(quality_scores) / max(len(quality_scores), 1),
        "avg_latency": sum(latencies) / len(latencies),
        "total_tokens": tokens,
    }


def main():
    test_data = load_test(TEST_PATH)
    print(f"📂 评估样本：{len(test_data)} 条\n")

    judge_llm = ChatOpenAI(
        model="gpt-4",
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0,
    )

    print("📊 评估基座模型...")
    model, tokenizer = load_model(BASE_MODEL_PATH)
    base_report = evaluate(model, tokenizer, test_data, judge_llm, "base")
    del model
    torch.cuda.empty_cache()

    print("📊 评估微调后模型...")
    model, tokenizer = load_model(BASE_MODEL_PATH, ADAPTER_PATH)
    ft_report = evaluate(model, tokenizer, test_data, judge_llm, "finetuned")

    print("\n" + "=" * 60)
    print(f"{'指标':<18}{'基座':<15}{'微调后':<15}")
    print("-" * 60)
    print(f"{'格式正确率':<18}{base_report['format_pass_rate']:.1%}{'':<8}{ft_report['format_pass_rate']:.1%}")
    print(f"{'质量分':<18}{base_report['avg_quality']:.2f}{'':<10}{ft_report['avg_quality']:.2f}")
    print(f"{'平均耗时(s)':<15}{base_report['avg_latency']:.2f}{'':<10}{ft_report['avg_latency']:.2f}")
    print(f"{'总Token':<18}{base_report['total_tokens']}{'':<8}{ft_report['total_tokens']}")
    print("=" * 60)

    report_path = BASE_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"base": base_report, "finetuned": ft_report}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 报告保存至：{report_path}")


if __name__ == "__main__":
    main()