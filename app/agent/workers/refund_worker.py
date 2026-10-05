# app/agent/workers/refund_worker.py
import re
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.agent.graph.state import AgentState
from app.agent.tools.registry import tool_registry
from app.rag.engine import RAGEngine

# 全局单例（模型加载耗时长，只初始化一次）
_rag_engine: RAGEngine = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


class RefundWorker:
    """
    退款 Worker：
      - 线性工作流，容错率为零
      - LLM 只负责提取参数和生成话术
      - 高危工具（initiate_refund）必须走 pending_approval
    """
    def __init__(self, llm):
        self.llm = llm
        self.rag = get_rag_engine()
        self.graph = self._build_graph()

    # ==================== 节点定义 ====================
    def node_extract_order(self, state: AgentState) -> Dict[str, Any]:
        """节点1：从用户问题中提取订单号"""
        query = state["user_query"]
        # 先用正则提取，命中直接返回（零 LLM 成本）
        match = re.search(r"(ORD[-\s]?\d+|订单号[:：]?\s*\w+)", query, re.IGNORECASE)
        if match:
            order_id = match.group(0).replace("订单号", "").replace(":", "").replace("：", "").strip()
        else:
            # 正则未命中，用 LLM 提取
            prompt = f"从下面这句话中提取订单号，只输出订单号，没有则输出 NONE：\n{query}"
            order_id = self.llm.invoke(prompt).content.strip()
            if order_id.upper() == "NONE":
                order_id = None

        return {"refund_result": {"order_id": order_id, "stage": "extracted"}}

    def node_query_order(self, state: AgentState) -> Dict[str, Any]:
        """节点2：调用工具查询订单状态"""
        refund = state.get("refund_result", {})
        order_id = refund.get("order_id")
        if not order_id:
            return {"refund_result": {**refund, "error": "未识别到订单号"}}

        tool = tool_registry.get("query_order_status")
        result = tool.func(order_id=order_id)
        return {"refund_result": {**refund, **result, "stage": "queried"}}

    def node_rag_policy(self, state: AgentState) -> Dict[str, Any]:
        """节点3：RAG 检索退款政策（混合检索 + RRF + 精排 + 父子召回）"""
        from app.core.database import SessionLocal
        db = SessionLocal()
        try:
            self.rag.build_bm25_index(db)
            docs = self.rag.retrieve(db, "退款政策 已发货 退款金额")
            policy_text = "\n".join([d["parent_content"] for d in docs]) if docs else "无相关政策"
        finally:
            db.close()

        refund = state.get("refund_result", {})
        return {"refund_result": {**refund, "policy": policy_text, "stage": "policy_retrieved"}}

    def node_check_condition(self, state: AgentState) -> Dict[str, Any]:
        """节点4：判定是否符合退款条件（规则 + LLM）"""
        refund = state.get("refund_result", {})
        status = refund.get("status", "")
        # 规则判定：已发货需退货退款
        if status in ("已发货", "已签收"):
            eligible = True
            note = "已发货订单，符合退货退款条件，退款需扣除运费"
        elif status in ("待支付", "已支付"):
            eligible = True
            note = "未发货订单，符合无理由退款条件"
        elif status == "已退款":
            eligible = False
            note = "订单已退款，无需重复操作"
        else:
            eligible = False
            note = f"当前订单状态（{status}）不符合退款条件"

        return {"refund_result": {**refund, "is_eligible": eligible, "condition_note": note, "stage": "checked"}}

    def node_generate_suggestion(self, state: AgentState) -> Dict[str, Any]:
        """节点5：生成退款建议（LLM 生成话术）"""
        refund = state.get("refund_result", {})
        if not refund.get("is_eligible"):
            return {"refund_result": {**refund, "suggestion": refund.get("condition_note", "")}}

        prompt = (
            f"订单 {refund.get('order_id')} 状态：{refund.get('status')}，"
            f"金额：{refund.get('amount')} 元，退款政策：{refund.get('policy')}。\n"
            f"请用客服口吻生成一段给用户的退款建议（不超过 80 字）。"
        )
        suggestion = self.llm.invoke(prompt).content.strip()
        return {"refund_result": {**refund, "suggestion": suggestion, "stage": "suggested"}}

    def node_pending_approval(self, state: AgentState) -> Dict[str, Any]:
        """节点6：高危操作挂起，等待人工确认（Human-in-the-loop）"""
        refund = state.get("refund_result", {})
        if not refund.get("is_eligible"):
            # 不符合条件，不需要人工审批，直接结束
            return {"need_human": False}

        pending = {
            "worker": "refund",
            "order_id": refund.get("order_id"),
            "amount": refund.get("amount"),
            "reason": refund.get("suggestion", ""),
            "status": "PENDING",
        }
        return {
            "pending_approval": pending,
            "need_human": True,        # ⚠️ 挂起，等人工确认
            "refund_result": {**refund, "stage": "pending_human"},
        }

    def node_execute_refund(self, state: AgentState) -> Dict[str, Any]:
        """节点7：人工确认后，执行退款（高危工具）"""
        refund = state.get("refund_result", {})
        tool = tool_registry.get("initiate_refund")
        result = tool.func(
            order_id=refund.get("order_id"),
            amount=float(refund.get("amount", 0)),
            reason=refund.get("suggestion", "用户申请退款"),
        )
        return {"refund_result": {**refund, **result, "stage": "executed"}}

    # ==================== 图编排 ====================
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("extract_order", self.node_extract_order)
        workflow.add_node("query_order", self.node_query_order)
        workflow.add_node("rag_policy", self.node_rag_policy)
        workflow.add_node("check_condition", self.node_check_condition)
        workflow.add_node("generate_suggestion", self.node_generate_suggestion)
        workflow.add_node("pending_approval", self.node_pending_approval)
        workflow.add_node("execute_refund", self.node_execute_refund)

        # 单向线性流
        workflow.set_entry_point("extract_order")
        workflow.add_edge("extract_order", "query_order")
        workflow.add_edge("query_order", "rag_policy")
        workflow.add_edge("rag_policy", "check_condition")
        workflow.add_edge("check_condition", "generate_suggestion")
        workflow.add_edge("generate_suggestion", "pending_approval")

        # 条件分流：需要人工 → END（等外部唤醒）；不需要 → 直接 END
        workflow.add_conditional_edges(
            "pending_approval",
            lambda s: "execute_refund" if s.get("need_human") is False and s.get("refund_result", {}).get("is_eligible") else END,
            {"execute_refund": "execute_refund", END: END},
        )
        workflow.add_edge("execute_refund", END)

        return workflow.compile()