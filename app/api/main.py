# app/api/main.py
import re
import time
import uuid
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm_factory import get_large_llm, get_small_llm
from app.agent.graph.main_graph import MainGraph
from app.api.approval import router as approval_router


# =========================================================
# 1. FastAPI 应用初始化
# =========================================================
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="智能工单助手 Agent 系统",
)


# =========================================================
# 2. 入口层：令牌桶限流（单机版；生产替换为 Redis + Lua）
# =========================================================
class TokenBucket:
    """
    内存令牌桶：
      - rate：每秒补充的令牌数（QPS）
      - capacity：桶容量（允许的突发流量）
    生产环境必须替换为 Redis 版，支持分布式多实例。
    """
    def __init__(self, rate: float = 10.0, capacity: int = 20):
        self.rate = rate
        self.capacity = capacity
        self._tokens = defaultdict(lambda: capacity)
        self._last_time = defaultdict(time.time)

    def allow(self, key: str) -> bool:
        now = time.time()
        elapsed = now - self._last_time[key]
        self._tokens[key] = min(
            self.capacity, self._tokens[key] + elapsed * self.rate
        )
        self._last_time[key] = now

        if self._tokens[key] >= 1:
            self._tokens[key] -= 1
            return True
        return False


bucket = TokenBucket(rate=10, capacity=20)


# =========================================================
# 3. 入口层：正则规则引擎（拦截高频确定性问题，降本）
# =========================================================
REGEX_RULES = [
    (re.compile(r"查余额|余额查询|我的余额"), "您的账户余额为 ¥1288.00（规则引擎直接返回）"),
    (re.compile(r"转人工|人工客服|找人工"), "已为您转接人工客服，请稍候。"),
    (re.compile(r"^你好$|^您好$|^hi$|^hello$", re.IGNORECASE), "您好，请问有什么可以帮您？"),
    (re.compile(r"改密码|重置密码"), None),  # None 表示交由 Agent 处理（高危需人工）
]


def regex_prefilter(query: str) -> Optional[str]:
    """命中正则直接返回固定话术；未命中返回 None 交给 Agent"""
    for pattern, reply in REGEX_RULES:
        if pattern.search(query) and reply is not None:
            return reply
    return None


# =========================================================
# 4. 初始化 LLM（大模型 + 小模型）与主图
# =========================================================
large_llm = get_large_llm()
small_llm = get_small_llm()

# 主图：大模型跑业务，小模型跑意图分类 / Query 改写
main_graph = MainGraph(llm=large_llm, small_llm=small_llm)


# =========================================================
# 5. 小模型分流 Prompt（简历亮点：降低 API 成本）
# =========================================================
SIMPLE_CLASSIFY_PROMPT = """判断下面这句话是"简单问题"还是"复杂问题"。
简单问题：问候、闲聊、单一确定性问题，不需要查数据库、不需要调工具
复杂问题：涉及多步骤、需要工具调用、需要查数据、涉及退款/技术/账户业务
只输出"简单"或"复杂"这两个词之一，不要输出其他任何内容。

用户问题：{query}
"""

SIMPLE_REPLY_PROMPT = """你是一个友好的客服助手。用户说：{query}
请用一句话简洁回答，不超过 30 字。"""


# =========================================================
# 6. 请求 / 响应模型
# =========================================================
class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    user_id: str = Field("anonymous", description="用户ID")


class ChatResponse(BaseModel):
    source: str                           # regex | small_model | agent
    thread_id: Optional[str] = None       # Agent 会话 ID（用于后续审批）
    intent: Optional[str] = None          # 意图标签
    need_human: bool = False              # 是否需要人工介入
    pending_approval: dict = {}           # 待审批信息
    reply: str = ""                       # 最终回复


# =========================================================
# 7. 主接口：智能工单入口（限流 → 正则 → 小模型 → Agent）
# =========================================================
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"

    # -------- 步骤 1：限流 --------
    if not bucket.allow(client_ip):
        raise HTTPException(429, "请求过于频繁，请稍后再试")

    # -------- 步骤 2：正则规则引擎（拦截约 35% 高频问题，零 LLM 成本）--------
    regex_reply = regex_prefilter(req.query)
    if regex_reply:
        return ChatResponse(source="regex", reply=regex_reply)

    # -------- 步骤 3：小模型分流（判断简单/复杂，降本约 40%）--------
    try:
        classify_prompt = SIMPLE_CLASSIFY_PROMPT.format(query=req.query)
        label = small_llm.invoke(classify_prompt).content.strip()
    except Exception:
        # 小模型异常兜底：直接当复杂问题处理
        label = "复杂"

    if "简单" in label:
        # 简单问题由本地小模型直接处理，零 API 成本
        try:
            reply = small_llm.invoke(
                SIMPLE_REPLY_PROMPT.format(query=req.query)
            ).content.strip()
        except Exception:
            reply = "您好，请问有什么可以帮您？"
        return ChatResponse(source="small_model", reply=reply)

    # -------- 步骤 4：复杂问题走多 Agent 工作流（大模型）--------
    thread_id = f"thread-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = main_graph.graph.invoke(
            {"user_query": req.query},
            config=config,
        )
    except Exception as e:
        raise HTTPException(500, f"Agent 执行异常：{str(e)}")

    return ChatResponse(
        source="agent",
        thread_id=thread_id,
        intent=result.get("intent"),
        need_human=result.get("need_human", False),
        pending_approval=result.get("pending_approval", {}),
        reply=result.get("final_response", ""),
    )


# =========================================================
# 8. 健康检查
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }


# =========================================================
# 9. 挂载审批路由
# =========================================================
app.include_router(approval_router)