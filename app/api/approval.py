# app/api/approval.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.core.database import SessionLocal
from app.domain.models import AgentState as AgentStateModel

router = APIRouter(prefix="/api/v1/approval", tags=["approval"])


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    comment: str = ""


class ApprovalResponse(BaseModel):
    status: str
    message: str
    thread_id: str


def _resume_agent(thread_id: str, approved: bool, comment: str):
    """
    后台任务：唤醒被挂起的 LangGraph 流程。
    生产中应改为投递 RedisMQ，由独立 Worker 消费恢复。
    """
    from app.api.main import main_graph
    from langchain_core.runnables import RunnableConfig

    db = SessionLocal()
    try:
        row = db.query(AgentStateModel).filter(
            AgentStateModel.thread_id == thread_id
        ).first()
        if not row:
            return
        state_data = row.state_data or {}
        pending = state_data.get("pending_approval", {})

        # 写回审批结果到 state_data
        if approved:
            state_data["need_human"] = False
            state_data["pending_approval"] = {**pending, "status": "APPROVED", "comment": comment}
        else:
            state_data["need_human"] = False
            state_data["pending_approval"] = {**pending, "status": "REJECTED", "comment": comment}
        row.state_data = state_data
        db.commit()

        # 唤醒 LangGraph：传入 None 表示从 checkpoint 恢复
        if approved:
            config = {"configurable": {"thread_id": thread_id}}
            main_graph.graph.invoke(None, config=config)
    finally:
        db.close()


@router.post("/submit", response_model=ApprovalResponse)
def submit_approval(req: ApprovalRequest, background_tasks: BackgroundTasks):
    """
    客服审批提交：
      - approved=True：唤醒 Agent 继续执行高危操作
      - approved=False：终止流程
    """
    db = SessionLocal()
    try:
        row = db.query(AgentStateModel).filter(
            AgentStateModel.thread_id == req.thread_id
        ).first()
        if not row:
            raise HTTPException(404, "找不到对应的 Agent 会话")

        state_data = row.state_data or {}
        pending = state_data.get("pending_approval", {})
        if not pending:
            raise HTTPException(400, "该会话没有待审批的高危操作")

        # 异步唤醒（避免阻塞客服端请求）
        background_tasks.add_task(_resume_agent, req.thread_id, req.approved, req.comment)

        status = "APPROVED" if req.approved else "REJECTED"
        return ApprovalResponse(
            status=status,
            message=f"审批已{'通过' if req.approved else '拒绝'}，Agent 将{'继续执行' if req.approved else '终止流程'}",
            thread_id=req.thread_id,
        )
    finally:
        db.close()


@router.get("/pending/{thread_id}")
def get_pending(thread_id: str):
    """查询某个会话的待审批状态（供客服后台轮询）"""
    db = SessionLocal()
    try:
        row = db.query(AgentStateModel).filter(
            AgentStateModel.thread_id == thread_id
        ).first()
        if not row:
            raise HTTPException(404, "找不到对应的 Agent 会话")
        state_data = row.state_data or {}
        return {
            "thread_id": thread_id,
            "pending_approval": state_data.get("pending_approval", {}),
            "need_human": state_data.get("need_human", False),
        }
    finally:
        db.close()