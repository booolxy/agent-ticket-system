# app/agent/supervisor/router.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 意图标签枚举
IntentType = Literal["refund", "tech", "account", "simple", "unknown"]

# Supervisor 的意图路由 Prompt
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个工单路由主管。请阅读用户问题，只输出一个意图标签，"
     "不要输出任何其他字符。可选标签：\n"
     "- refund（退款、退货、查退款进度）\n"
     "- tech（系统报错、接口异常、页面白屏）\n"
     "- account（重置密码、改绑手机、查余额）\n"
     "- simple（简单问候、闲聊、无需工具）\n"
     "- unknown（无法判断）"),
    ("human", "用户问题：{query}")
])

class SupervisorRouter:
    """
    Supervisor 主管 Agent
    职责：
      1. 意图识别（决定路由到哪个 Worker）
      2. 多意图排序（多标签时按优先级输出）
      3. 不执行业务逻辑，只负责分发
    """
    def __init__(self, llm):
        """
        :param llm: LangChain 兼容的 LLM 实例
                    生产中优先使用微调后的 Qwen-1.8B，降本
        """
        self.llm = llm
        self.chain = ROUTER_PROMPT | self.llm | StrOutputParser()

    def route(self, query: str) -> IntentType:
        """同步路由：返回意图标签"""
        raw = self.chain.invoke({"query": query}).strip().lower()
        # 容错：只取第一个单词
        tag = raw.split()[0] if raw else "unknown"
        valid = {"refund", "tech", "account", "simple", "unknown"}
        return tag if tag in valid else "unknown"

    async def aroute(self, query: str) -> IntentType:
        """异步路由：FastAPI 中推荐使用"""
        raw = (await self.chain.ainvoke({"query": query})).strip().lower()
        tag = raw.split()[0] if raw else "unknown"
        valid = {"refund", "tech", "account", "simple", "unknown"}
        return tag if tag in valid else "unknown"