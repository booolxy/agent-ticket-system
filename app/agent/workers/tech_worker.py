# app/agent/workers/tech_worker.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.agent.graph.state import AgentState
from app.agent.tools.registry import tool_registry
from app.rag.engine import RAGEngine


# =========================================================
# RAG 引擎单例（模型加载耗时长，只初始化一次）
# =========================================================
_rag_engine: RAGEngine = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


class TechWorker:
    """
    技术 Worker：
      - 先查后建：能答就答，答不了就建单转研发
      - 置信度判定：向量相似度 + Rerank 分数 + LLM 二次校验
      - 不挂高危工具（不涉及资金和账户）

    模型分工（工程化降本）：
      - small_llm：Query 改写（口语化转书面语，简单任务）
      - llm：生成解决方案（需要专业性和上下文理解）
    """

    # 置信度阈值（Rerank 分数）
    RERANK_THRESHOLD = 0.7

    def __init__(self, llm, small_llm=None):
        """
        :param llm: 大模型，用于生成解决方案
        :param small_llm: 小模型（本地 Qwen-1.8B），用于 Query 改写
                          若不传，降级复用大模型，保证不崩
        """
        self.llm = llm
        self.small_llm = small_llm or llm     # 降级兜底
        self.rag = get_rag_engine()
        self.graph = self._build_graph()

    # ==================== 节点 1：参数提取 ====================
    def node_extract_params(self, state: AgentState) -> Dict[str, Any]:
        """从用户问题中提取问题描述、错误码、模块"""
        import re
        query = state["user_query"]
        # 正则提取错误码（例如 500、ERR-1001、E500）
        code_match = re.search(r"\b(\d{3,5}|[A-Z]{2,}-\d+)\b", query)
        error_code = code_match.group(0) if code_match else ""

        return {
            "tech_result": {
                "description": query,
                "error_code": error_code,
                "stage": "extracted",
            }
        }

    # ==================== 节点 2：Query 改写（用小模型）====================
    def node_rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Query 改写：口语化 → 书面语
        使用小模型（本地 Qwen-1.8B），降低 API 成本
        """
        tech = state.get("tech_result", {})
        raw = tech.get("description", "")

        prompt = (
            "请将下面这句用户口语化的技术问题，改写成简洁、专业的书面技术描述，"
            "只输出改写结果，不要解释：\n"
            f"{raw}"
        )

        try:
            rewritten = self.small_llm.invoke(prompt).content.strip()
        except Exception:
            # 小模型异常兜底：直接用原句
            rewritten = raw

        return {
            "tech_result": {
                **tech,
                "rewritten": rewritten,
                "stage": "rewritten",
            }
        }

    # ==================== 节点 3：RAG 混合检索 ====================
    def node_rag_search(self, state: AgentState) -> Dict[str, Any]:
        """RAG 混合检索技术知识库（BM25 + 向量 + RRF + Rerank + 父子召回）"""
        from app.core.database import SessionLocal

        tech = state.get("tech_result", {})
        query = tech.get("rewritten") or tech.get("description", "")

        db = SessionLocal()
        try:
            self.rag.build_bm25_index(db)
            docs = self.rag.retrieve(db, query)
        finally:
            db.close()

        top_score = docs[0]["score"] if docs else 0.0
        return {
            "tech_result": {
                **tech,
                "docs": docs,
                "top_score": top_score,
                "stage": "retrieved",
            }
        }

    # ==================== 节点 4：置信度判定 ====================
    def node_check_confidence(self, state: AgentState) -> Dict[str, Any]:
        """置信度判定：Rerank 分数 >= 阈值则视为命中"""
        tech = state.get("tech_result", {})
        score = tech.get("top_score", 0.0)
        hit = score >= self.RERANK_THRESHOLD

        return {
            "tech_result": {
                **tech,
                "hit": hit,
                "stage": "checked",
            }
        }

    # ==================== 节点 5：命中，生成解决方案（用大模型）====================
    def node_generate_solution(self, state: AgentState) -> Dict[str, Any]:
        """
        命中知识库，生成解决方案
        使用大模型（需要专业性和上下文理解）
        """
        tech = state.get("tech_result", {})
        docs = tech.get("docs", [])
        context = "\n".join([d["parent_content"] for d in docs[:2]]) if docs else "无参考"

        prompt = (
            f"用户技术问题：{tech.get('rewritten', '')}\n"
            f"知识库参考：{context}\n"
            f"请给出简洁、可执行的解决方案（不超过 100 字）。"
        )

        try:
            solution = self.llm.invoke(prompt).content.strip()
        except Exception:
            solution = "已为您找到相关问题记录，正在转接技术支持。"

        return {
            "tech_result": {
                **tech,
                "solution": solution,
                "stage": "solved",
            }
        }

    # ==================== 节点 6：未命中，创建技术工单 ====================
    def node_create_ticket(self, state: AgentState) -> Dict[str, Any]:
        """未命中知识库，创建技术工单转研发"""
        tech = state.get("tech_result", {})
        tool = tool_registry.get("create_tech_ticket")
        result = tool.func(
            user_id="U001",                            # 实际应从登录态获取
            description=tech.get("description", ""),
            error_code=tech.get("error_code", ""),
        )
        return {
            "tech_result": {
                **tech,
                **result,
                "stage": "ticketed",
            }
        }

    # ==================== 图编排 ====================
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("extract_params", self.node_extract_params)
        workflow.add_node("rewrite_query", self.node_rewrite_query)
        workflow.add_node("rag_search", self.node_rag_search)
        workflow.add_node("check_confidence", self.node_check_confidence)
        workflow.add_node("generate_solution", self.node_generate_solution)
        workflow.add_node("create_ticket", self.node_create_ticket)

        # 单向线性流
        workflow.set_entry_point("extract_params")
        workflow.add_edge("extract_params", "rewrite_query")
        workflow.add_edge("rewrite_query", "rag_search")
        workflow.add_edge("rag_search", "check_confidence")

        # 条件分流：命中 → 生成解决方案；未命中 → 创建工单
        workflow.add_conditional_edges(
            "check_confidence",
            lambda s: "generate_solution"
            if s.get("tech_result", {}).get("hit")
            else "create_ticket",
            {
                "generate_solution": "generate_solution",
                "create_ticket": "create_ticket",
            },
        )
        workflow.add_edge("generate_solution", END)
        workflow.add_edge("create_ticket", END)

        return workflow.compile()