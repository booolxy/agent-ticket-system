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
from app.core.security import InputGuard, OutputGuard
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
# 2. 令牌桶限流
# =========================================================
class TokenBucket:
    def __init__(self, rate: float = 10.0, capacity: int = 20):
        self.rate = rate
        self.capacity = capacity
        self._tokens = defaultdict(lambda: capacity)
        self._last_time = defaultdict(time.time)

    def allow(self, key: str) -> bool:
        now = time.time()
        elapsed = now - self._last_time[key]
        self._tokens[key] = min(self.capacity, self._tokens[key] + elapsed * self.rate)
        self._last_time[key] = now
        if self._tokens[key] >= 1:
            self._tokens[key] -= 1
            return True
        return False


bucket = TokenBucket(rate=10, capacity=20)


# =========================================================
# 3. 正则规则引擎（业务分流，不是安全防护）
# =========================================================
REGEX_RULES = [
    (re.compile(r"查余额|余额查询|我的余额"), "您的账户余额为 ¥1288.00（规则引擎直接返回）"),
    (re.compile(r"转人工|人工客服|找人工"), "已为您转接人工客服，请稍候。"),
    (re.compile(r"^你好$|^您好$|^hi$|^hello$", re.IGNORECASE), "您好，请问有什么可以帮您？"),
    (re.compile(r"改密码|重置密码"), None),
]


def regex_prefilter(query: str) -> Optional[str]:
    for pattern, reply in REGEX_RULES:
        if pattern.search(query) and reply is not None:
            return reply
    return None


# =========================================================
# 4. 初始化 LLM 与主图
# =========================================================
large_llm = get_large_llm()
small_llm = get_small_llm()
main_graph = MainGraph(llm=large_llm, small_llm=small_llm)


# =========================================================
# 5. 小模型分流 Prompt
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
    source: str
    thread_id: Optional[str] = None
    intent: Optional[str] = None
    need_human: bool = False
    pending_approval: dict = {}
    reply: str = ""


# =========================================================
# 7. 主接口（含三层安全防护）
# =========================================================
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"

    # -------- 步骤 1：限流 --------
    if not bucket.allow(client_ip):
        raise HTTPException(429, "请求过于频繁，请稍后再试")

    # -------- 步骤 2：【安全层 1】用户输入黑名单拦截 --------
    is_safe, reason = InputGuard.check(req.query)
    if not is_safe:
        return ChatResponse(
            source="blocked",
            reply=f"抱歉，无法处理该请求（{reason}）。如需帮助请联系人工客服。",
        )

    # -------- 步骤 3：正则规则引擎（业务分流）--------
    regex_reply = regex_prefilter(req.query)
    if regex_reply:
        return ChatResponse(source="regex", reply=regex_reply)

    # -------- 步骤 4：小模型分流 --------
    try:
        classify_prompt = SIMPLE_CLASSIFY_PROMPT.format(query=req.query)
        label = small_llm.invoke(classify_prompt).content.strip()
    except Exception:
        label = "复杂"

    if "简单" in label:
        try:
            reply = small_llm.invoke(
                SIMPLE_REPLY_PROMPT.format(query=req.query)
            ).content.strip()
        except Exception:
            reply = "您好，请问有什么可以帮您？"

        # 【安全层 3】输出拦截
        is_safe, reply = OutputGuard.check(reply)
        return ChatResponse(source="small_model", reply=reply)

    # -------- 步骤 5：复杂问题走多 Agent 工作流 --------
    thread_id = f"thread-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = main_graph.graph.invoke({"user_query": req.query}, config=config)
    except Exception as e:
        raise HTTPException(500, f"Agent 执行异常：{str(e)}")

    # 【安全层 3】输出拦截
    final_reply = result.get("final_response", "")
    is_safe, final_reply = OutputGuard.check(final_reply)

    return ChatResponse(
        source="agent",
        thread_id=thread_id,
        intent=result.get("intent"),
        need_human=result.get("need_human", False),
        pending_approval=result.get("pending_approval", {}),
        reply=final_reply,
    )


# =========================================================
# 8. 健康检查
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok", "service": settings.PROJECT_NAME, "version": settings.VERSION}


# =========================================================
# 9. 挂载审批路由
# =========================================================
app.include_router(approval_router)