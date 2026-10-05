# app/agent/tools/registry.py
from typing import Callable, Dict, Any, List
from pydantic import BaseModel, Field

class ToolSpec(BaseModel):
    """工具元数据规范"""
    name: str
    description: str
    func: Callable
    is_high_risk: bool = False       # 是否高危（高危必须走人工审批）
    requires_sandbox: bool = False   # 是否需要 Docker 沙箱隔离
    args_schema: type = None         # Pydantic 参数模型

class ToolRegistry:
    """
    全局工具注册中心 - 单例模式
    """
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        if spec.name in self._tools:
            raise ValueError(f"工具 {spec.name} 已注册")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"未注册的工具：{name}")
        return self._tools[name]

    def list_all(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def list_by_risk(self, high_risk: bool) -> List[ToolSpec]:
        return [t for t in self._tools.values() if t.is_high_risk == high_risk]


# 全局单例
tool_registry = ToolRegistry()


# ============ 参数模型（Pydantic 强校验）============
class QueryOrderArgs(BaseModel):
    order_id: str = Field(..., description="订单ID，例如 ORD-123")

class InitiateRefundArgs(BaseModel):
    order_id: str
    amount: float
    reason: str

class CreateTechTicketArgs(BaseModel):
    user_id: str
    description: str
    error_code: str = ""

class SearchKnowledgeArgs(BaseModel):
    query: str = Field(..., description="检索关键词")

class ResetPasswordArgs(BaseModel):
    user_id: str
    new_password: str


# ============ 工具函数定义（模拟实现）============
def query_order_status(order_id: str) -> Dict[str, Any]:
    """查询订单状态（模拟：实际应查 PG）"""
    # TODO: 后续换成真实 DB 查询
    return {
        "order_id": order_id,
        "status": "已发货",
        "amount": 199.00,
        "user_id": "U001"
    }

def initiate_refund(order_id: str, amount: float, reason: str) -> Dict[str, Any]:
    """发起退款（高危操作，必须先人工确认）"""
    # 实际生产中这里会调用支付网关 API
    return {
        "success": True,
        "order_id": order_id,
        "amount": amount,
        "message": f"退款 {amount} 元已提交，原因：{reason}"
    }

def create_tech_ticket(user_id: str, description: str, error_code: str = "") -> Dict[str, Any]:
    """创建技术工单（非高危）"""
    import uuid
    return {
        "ticket_id": f"TECH-{uuid.uuid4().hex[:8]}",
        "user_id": user_id,
        "status": "OPEN"
    }

def search_knowledge_base(query: str) -> Dict[str, Any]:
    """RAG 检索（占位，实际在 Agent 中直接调用 RAGEngine）"""
    return {"query": query, "results": []}

def reset_password(user_id: str, new_password: str) -> Dict[str, Any]:
    """重置密码（高危操作）"""
    return {"success": True, "user_id": user_id, "message": "密码已重置"}


# ============ 注册所有工具 ============
def register_all_tools():
    tool_registry.register(ToolSpec(
        name="query_order_status",
        description="查询订单状态、退款进度",
        func=query_order_status,
        is_high_risk=False,
        args_schema=QueryOrderArgs,
    ))
    tool_registry.register(ToolSpec(
        name="initiate_refund",
        description="发起退款（高危，必须人工确认）",
        func=initiate_refund,
        is_high_risk=True,      # ⚠️ 高危
        args_schema=InitiateRefundArgs,
    ))
    tool_registry.register(ToolSpec(
        name="create_tech_ticket",
        description="创建技术工单转研发",
        func=create_tech_ticket,
        is_high_risk=False,
        args_schema=CreateTechTicketArgs,
    ))
    tool_registry.register(ToolSpec(
        name="search_knowledge_base",
        description="RAG 检索技术知识库",
        func=search_knowledge_base,
        is_high_risk=False,
        args_schema=SearchKnowledgeArgs,
    ))
    tool_registry.register(ToolSpec(
        name="reset_password",
        description="重置用户密码（高危，必须人工确认）",
        func=reset_password,
        is_high_risk=True,      # ⚠️ 高危
        args_schema=ResetPasswordArgs,
    ))


# 模块导入时自动注册
register_all_tools()