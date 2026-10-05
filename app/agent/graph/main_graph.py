# app/agent/graph/main_graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from app.agent.graph.state import AgentState
from app.agent.supervisor.router import SupervisorRouter
from app.agent.workers.refund_worker import RefundWorker
from app.agent.workers.tech_worker import TechWorker
from app.agent.workers.account_worker import AccountWorker
from app.core.config import settings


class MainGraph:
    """
    主图：Supervisor + 三个 Worker 编排

    工程化要点：
      1. PostgresSaver：状态持久化到 PG，支持崩溃恢复、回滚
      2. 串行路由：Supervisor 单线程分发，同一时刻只有一个 Worker 执行
      3. 单向数据流：每个节点只写自己的命名空间字段
      4. Human-in-the-loop：高危操作挂起在 pending_approval
    """

    def __init__(self, llm):
        self.llm = llm
        self.supervisor = SupervisorRouter(llm)
        self.refund_worker = RefundWorker(llm)
        self.tech_worker = TechWorker(llm)
        self.account_worker = AccountWorker(llm)

        # 挂载 PostgreSQL Checkpointer，实现状态快照与崩溃恢复
        self.checkpointer = PostgresSaver.from_conn_string(settings.DATABASE_URL)
        # 首次使用需初始化 checkpointer 所需的表（幂等操作）
        self.checkpointer.setup()

        self.graph = self._build_graph()

    # ==================== 主图节点 ====================

    def node_supervisor(self, state: AgentState) -> dict:
        """Supervisor 节点：意图路由（不执行业务，只分发）"""
        intent = self.supervisor.route(state["user_query"])
        return {"intent": intent}

    def node_refund(self, state: AgentState) -> dict:
        """
        退款 Worker 节点
        严格单向数据流：只提取 refund_result 命名空间回写主图
        """
        result = self.refund_worker.graph.invoke(state)
        return {
            "refund_result": result.get("refund_result", {}),
            "pending_approval": result.get("pending_approval", {}),
            "need_human": result.get("need_human", False),
        }

    def node_tech(self, state: AgentState) -> dict:
        """
        技术 Worker 节点
        严格单向数据流：只提取 tech_result 命名空间回写主图
        """
        result = self.tech_worker.graph.invoke(state)
        return {"tech_result": result.get("tech_result", {})}

    def node_account(self, state: AgentState) -> dict:
        """
        账户 Worker 节点
        严格单向数据流：只提取 account_result 命名空间回写主图
        """
        result = self.account_worker.graph.invoke(state)
        return {
            "account_result": result.get("account_result", {}),
            "pending_approval": result.get("pending_approval", {}),
            "need_human": result.get("need_human", False),
        }

    def node_aggregate(self, state: AgentState) -> dict:
        """汇总节点：合并所有子结果 → 生成最终回复"""
        intent = state.get("intent")
        parts = []

        if intent == "refund":
            r = state.get("refund_result", {})
            parts.append(
                r.get("suggestion")
                or r.get("message")
                or r.get("condition_note", "已处理您的退款请求")
            )
        elif intent == "tech":
            t = state.get("tech_result", {})
            if t.get("hit"):
                parts.append(t.get("solution", "已为您找到解决方案"))
            else:
                parts.append(f"已为您创建技术工单：{t.get('ticket_id', 'N/A')}")
        elif intent == "account":
            a = state.get("account_result", {})
            parts.append(a.get("message") or "账户请求已处理，等待人工确认")
        elif intent == "simple":
            parts.append("您好，请问有什么可以帮您？")
        else:
            parts.append("抱歉，暂不支持该类型请求，已为您转人工客服。")

        return {"final_response": "\n".join([p for p in parts if p])}

    # ==================== 图编排 ====================

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 注册所有节点
        workflow.add_node("supervisor", self.node_supervisor)
        workflow.add_node("refund_worker", self.node_refund)
        workflow.add_node("tech_worker", self.node_tech)
        workflow.add_node("account_worker", self.node_account)
        workflow.add_node("aggregate", self.node_aggregate)

        # 入口：Supervisor
        workflow.set_entry_point("supervisor")

        # Supervisor 单线程串行路由（同一时刻只有一个 Worker 执行，避免并发写冲突）
        workflow.add_conditional_edges(
            "supervisor",
            lambda s: s.get("intent", "unknown"),
            {
                "refund": "refund_worker",
                "tech": "tech_worker",
                "account": "account_worker",
                "simple": "aggregate",
                "unknown": "aggregate",
            },
        )

        # 每个 Worker 完成后 → 汇总节点
        workflow.add_edge("refund_worker", "aggregate")
        workflow.add_edge("tech_worker", "aggregate")
        workflow.add_edge("account_worker", "aggregate")
        workflow.add_edge("aggregate", END)

        # 挂载 Checkpointer，实现状态持久化
        return workflow.compile(checkpointer=self.checkpointer)