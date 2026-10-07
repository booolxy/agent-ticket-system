# app/agent/workers/account_worker.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.agent.graph.state import AgentState
from app.agent.tools.registry import tool_registry
from app.core.security import SECURITY_SYSTEM_PROMPT, safe_llm_invoke
from app.rag.engine import RAGEngine


# =========================================================
# RAG 引擎单例
# =========================================================
_rag_engine: RAGEngine = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


class AccountWorker:
    """
    账户 Worker：
      - 分支工作流：手机号可用 → 短信验证；不可用 → 身份证验证
      - RAG 检索验证规则（风控策略常变，不写死代码）
      - reset_password 高危，必须 Human-in-the-loop
      - 所有 LLM 调用经 safe_llm_invoke 注入安全防护
    """

    def __init__(self, llm, small_llm=None):
        self.llm = llm
        self.small_llm = small_llm or llm
        self.rag = get_rag_engine()
        self.graph = self._build_graph()

    # ==================== 节点 1：参数提取 ====================
    def node_extract_params(self, state: AgentState) -> Dict[str, Any]:
        """
        从用户问题中提取 user_id 和账户操作意图。
        当前用规则提取；如需 LLM 提取，用 safe_llm_invoke 保证安全。
        """
        query = state["user_query"]

        # 【安全层 2】若未来改用 LLM 提取，用 safe_llm_invoke 保底：
        # prompt = f"从下面这句话中提取账户操作意图，只输出意图关键词：\n{query}"
        # intent = safe_llm_invoke(self.small_llm, prompt).strip()

        return {
            "account_result": {
                "user_id": "U001",          # 实际应从登录态获取
                "intent": query,
                "stage": "extracted",
            }
        }

    # ==================== 节点 2：检查手机号 ====================
    def node_check_phone(self, state: AgentState) -> Dict[str, Any]:
        """检查手机号是否可用（模拟：固定返回可用；生产应查 DB）"""
        account = state.get("account_result", {})
        phone_available = True
        return {
            "account_result": {
                **account,
                "phone_available": phone_available,
                "stage": "phone_checked",
            }
        }

    # ==================== 节点 3A：短信验证子流程 ====================
    def node_sms_flow(self, state: AgentState) -> Dict[str, Any]:
        """短信验证子流程"""
        account = state.get("account_result", {})
        return {
            "account_result": {
                **account,
                "verify_method": "sms",
                "verified": True,
                "stage": "sms_verified",
            }
        }

    # ==================== 节点 3B：身份证验证子流程 ====================
    def node_id_card_flow(self, state: AgentState) -> Dict[str, Any]:
        """身份证验证子流程（RAG 检索合规手册）"""
        from app.core.database import SessionLocal

        db = SessionLocal()
        try:
            self.rag.build_bm25_index(db)
            docs = self.rag.retrieve(db, "身份证核验 账户安全 重置密码规则")
            rules = "\n".join([d["parent_content"] for d in docs]) if docs else ""
        finally:
            db.close()

        account = state.get("account_result", {})
        return {
            "account_result": {
                **account,
                "verify_method": "id_card",
                "rules": rules,
                "verified": True,
                "stage": "id_verified",
            }
        }

    # ==================== 节点 4：高危操作挂起 ====================
    def node_pending_approval(self, state: AgentState) -> Dict[str, Any]:
        """高危操作挂起，等待人工确认"""
        account = state.get("account_result", {})
        pending = {
            "worker": "account",
            "user_id": account.get("user_id"),
            "action": "reset_password",
            "status": "PENDING",
        }
        return {
            "pending_approval": pending,
            "need_human": True,
            "account_result": {**account, "stage": "pending_human"},
        }

    # ==================== 节点 5：执行重置（高危）====================
    def node_execute_reset(self, state: AgentState) -> Dict[str, Any]:
        """人工确认后执行重置"""
        account = state.get("account_result", {})
        tool = tool_registry.get("reset_password")
        result = tool.func(
            user_id=account.get("user_id", ""),
            new_password="Temp@123456",
        )
        return {
            "account_result": {
                **account,
                **result,
                "stage": "executed",
            }
        }

    # ==================== 图编排 ====================
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("extract_params", self.node_extract_params)
        workflow.add_node("check_phone", self.node_check_phone)
        workflow.add_node("sms_flow", self.node_sms_flow)
        workflow.add_node("id_card_flow", self.node_id_card_flow)
        workflow.add_node("pending_approval", self.node_pending_approval)
        workflow.add_node("execute_reset", self.node_execute_reset)

        workflow.set_entry_point("extract_params")
        workflow.add_edge("extract_params", "check_phone")

        # 分支：手机号可用 → 短信；不可用 → 身份证
        workflow.add_conditional_edges(
            "check_phone",
            lambda s: "sms_flow"
            if s.get("account_result", {}).get("phone_available")
            else "id_card_flow",
            {
                "sms_flow": "sms_flow",
                "id_card_flow": "id_card_flow",
            },
        )
        workflow.add_edge("sms_flow", "pending_approval")
        workflow.add_edge("id_card_flow", "pending_approval")

        # 挂起后由外部人工确认唤醒
        workflow.add_conditional_edges(
            "pending_approval",
            lambda s: "execute_reset"
            if s.get("need_human") is False
            else END,
            {"execute_reset": "execute_reset", END: END},
        )
        workflow.add_edge("execute_reset", END)

        return workflow.compile()