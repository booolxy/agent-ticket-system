# app/agent/graph/main_graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from app.agent.graph.state import AgentState
from app.agent.planner.planner import Planner
from app.agent.supervisor.router import Supervisor
from app.agent.workers.refund_worker import RefundWorker
from app.agent.workers.tech_worker import TechWorker
from app.agent.workers.account_worker import AccountWorker
from app.core.config import settings


class MainGraph:
    """
    主图：Planner → 拓扑排序 → Supervisor 调度 → Worker → Aggregate

    执行流程：
      1. Planner 拆解 DAG + 拓扑排序
      2. Supervisor 找出下一个任务
      3. 根据 intent 路由到对应 Worker
      4. Worker 完成后回到 Supervisor
      5. 所有任务完成 → Aggregate
    """

    def __init__(self, llm, small_llm=None):
        self.llm = llm
        self.small_llm = small_llm or llm

        # 核心组件
        self.planner = Planner(self.llm)              # 拆解 DAG（用大模型）
        self.supervisor = Supervisor()                # 调度器（纯算法）
        self.refund_worker = RefundWorker(self.llm)   # 退款
        self.tech_worker = TechWorker(self.llm, small_llm=self.small_llm)  # 技术
        self.account_worker = AccountWorker(self.llm) # 账户

        self.checkpointer = PostgresSaver.from_conn_string(settings.DATABASE_URL)
        self.checkpointer.setup()

        self.graph = self._build_graph()

    # ==================== 节点 1：Planner ====================
    def node_planner(self, state: AgentState) -> dict:
        """Planner：拆解 DAG + 拓扑排序"""
        result = self.planner.plan(state["user_query"])
        return {
            "plan": result["plan"],
            "execution_order": result["execution_order"],
        }

    # ==================== 节点 2：Supervisor ====================
    def node_supervisor(self, state: AgentState) -> dict:
        """Supervisor：找出下一个要执行的任务"""
        next_id = self.supervisor.next_task(state)
        if next_id is None:
            return {"current_task_id": ""}
        return {"current_task_id": next_id}

    # ==================== 节点 3/4/5：三个 Worker ====================
    def node_refund(self, state: AgentState) -> dict:
        task_id = state.get("current_task_id", "")
        # 把当前子任务的 query 注入到 state，Worker 用这个 query
        task = self.supervisor.get_task_by_id(state, task_id)
        sub_state = {**state, "user_query": task["query"] if task else state["user_query"]}

        result = self.refund_worker.graph.invoke(sub_state)

        return {
            "refund_result": result.get("refund_result", {}),
            "task_results": {task_id: result.get("refund_result", {})},
            "pending_approval": result.get("pending_approval", {}),
            "need_human": result.get("need_human", False),
        }

    def node_tech(self, state: AgentState) -> dict:
        task_id = state.get("current_task_id", "")
        task = self.supervisor.get_task_by_id(state, task_id)
        sub_state = {**state, "user_query": task["query"] if task else state["user_query"]}

        result = self.tech_worker.graph.invoke(sub_state)

        return {
            "tech_result": result.get("tech_result", {}),
            "task_results": {task_id: result.get("tech_result", {})},
        }

    def node_account(self, state: AgentState) -> dict:
        task_id = state.get("current_task_id", "")
        task = self.supervisor.get_task_by_id(state, task_id)
        sub_state = {**state, "user_query": task["query"] if task else state["user_query"]}

        result = self.account_worker.graph.invoke(sub_state)

        return {
            "account_result": result.get("account_result", {}),
            "task_results": {task_id: result.get("account_result", {})},
            "pending_approval": result.get("pending_approval", {}),
            "need_human": result.get("need_human", False),
        }

    # ==================== 节点 6：Aggregate 汇总 ====================
    def node_aggregate(self, state: AgentState) -> dict:
        """汇总所有子任务结果 → 生成最终回复"""
        plan = state.get("plan", [])
        results = state.get("task_results", {})

        parts = []
        for task in plan:
            tid = task["id"]
            intent = task["intent"]
            r = results.get(tid, {})

            if intent == "refund":
                parts.append(r.get("suggestion") or r.get("condition_note", "退款处理完成"))
            elif intent == "tech":
                if r.get("hit"):
                    parts.append(r.get("solution", "技术问题已解决"))
                else:
                    parts.append(f"技术工单已创建：{r.get('ticket_id', 'N/A')}")
            elif intent == "account":
                parts.append(r.get("message") or "账户操作已处理，等待人工确认")
            else:
                parts.append("暂不支持该类型请求")

        final = "\n".join([p for p in parts if p]) or "已处理您的请求"
        return {"final_response": final}

    # ==================== 路由函数 ====================
    def _route_from_supervisor(self, state: AgentState) -> str:
        """根据 current_task_id 路由到对应 Worker，或结束"""
        next_id = state.get("current_task_id", "")
        if not next_id:
            return "aggregate"

        task = self.supervisor.get_task_by_id(state, next_id)
        if not task:
            return "aggregate"

        intent = task["intent"]
        if intent in ("refund", "tech", "account"):
            return intent
        # unknown 类型直接跳过（视为已完成）
        return "aggregate"

    # ==================== 图编排 ====================
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self.node_planner)
        workflow.add_node("supervisor", self.node_supervisor)
        workflow.add_node("refund", self.node_refund)
        workflow.add_node("tech", self.node_tech)
        workflow.add_node("account", self.node_account)
        workflow.add_node("aggregate", self.node_aggregate)

        # 入口：Planner → Supervisor
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "supervisor")

        # Supervisor 根据 current_task_id 分发
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "refund": "refund",
                "tech": "tech",
                "account": "account",
                "aggregate": "aggregate",
            },
        )

        # 每个 Worker 完成后回到 Supervisor（继续调度下一个）
        workflow.add_edge("refund", "supervisor")
        workflow.add_edge("tech", "supervisor")
        workflow.add_edge("account", "supervisor")

        workflow.add_edge("aggregate", END)

        return workflow.compile(checkpointer=self.checkpointer)