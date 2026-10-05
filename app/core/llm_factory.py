# app/core/llm_factory.py
"""
LLM 工厂：统一管理大模型和小模型的初始化

工程化思想：
  - 单一职责：所有 LLM 从这里产出，不在业务代码里硬编码
  - 可配置：通过 .env 切换不同的 LLM 提供商
  - 双模型路由：大模型处理复杂任务，小模型处理简单任务，降本
"""
from functools import lru_cache
from typing import Optional
from langchain_openai import ChatOpenAI

from app.core.config import settings


# =========================================================
# 大模型（GPT-3.5/4 或国内兼容 API）
# =========================================================
@lru_cache()
def get_large_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    大模型：用于复杂问题处理、多 Agent 协作
    支持任意 OpenAI 兼容协议（GPT、DeepSeek、通义千问、Moonshot 等）
    """
    return ChatOpenAI(
        model=settings.LARGE_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=temperature,
    )


# =========================================================
# 小模型（本地 Qwen-1.8B + LoRA，走本地推理）
# =========================================================
class LocalQwenLLM:
    """
    本地 Qwen-1.8B 推理封装（适配 LangChain ChatModel 接口）
    用于：意图分类、Query 改写、简单问题处理

    生产环境建议用 vLLM 或 TGI 部署成 HTTP 服务，走 OpenAI 兼容协议，
    避免每次调用都在 Python 进程里跑推理。
    """
    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 如果传了 LoRA adapter 路径，加载微调权重
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    def invoke(self, prompt: str, max_new_tokens: int = 256) -> str:
        """同步推理（简单封装，适配 LangChain 调用风格）"""
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()


class LocalLLMResponse:
    """包装成 LangChain 的 response 对象，兼容 .content 属性"""
    def __init__(self, content: str):
        self.content = content


class LocalQwenChatModel:
    """
    适配 LangChain ChatModel 接口的包装
    让 LocalQwenLLM 可以像 ChatOpenAI 一样被 .invoke(prompt).content 调用
    """
    def __init__(self, llm: LocalQwenLLM):
        self._llm = llm

    def invoke(self, prompt: str, **kwargs) -> LocalLLMResponse:
        if hasattr(prompt, "content"):
            prompt = prompt.content
        return LocalLLMResponse(self._llm.invoke(str(prompt)))


@lru_cache()
def get_small_llm() -> LocalQwenChatModel:
    """
    小模型：本地 Qwen-1.8B + LoRA
    首次调用会加载模型（耗时几十秒），之后 lru_cache 复用
    """
    base = LocalQwenLLM(
        model_path=settings.MODEL_QWEN_PATH,
        adapter_path=settings.MODEL_QWEN_LORA_PATH or None,
    )
    return LocalQwenChatModel(base)


# =========================================================
# 对外统一入口
# =========================================================
def get_llm(model_type: str = "large"):
    """
    统一入口：
      - "large"：大模型（GPT-3.5/4）
      - "small"：小模型（Qwen-1.8B 本地）
    """
    if model_type == "small":
        return get_small_llm()
    return get_large_llm()