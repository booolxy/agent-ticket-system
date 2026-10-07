# app/agent/tools/registry.py
from typing import Callable, Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """工具元数据规范"""
    name: str
    description: str
    func: Callable
    is_high_risk: bool = False
    requires_sandbox: bool = False
    args_schema: Optional[type] = None
    allowed_workers: List[str] = []   # 权限：哪些 Worker 可以调用

    class Config:
        arbitrary_types_allowed = True

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        导出 OpenAI function calling 兼容的 JSON Schema
        结构：{"type": "function", "function": {"name", "description", "parameters"}}
        """
        parameters = {"type": "object", "properties": {}, "required": []}
        if self.args_schema and issubclass(self.args_schema, BaseModel):
            schema = self.args_schema.model_json_schema()
            parameters = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class ToolRegistry:
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

    def get_schemas_for_worker(self, worker_name: str) -> List[Dict[str, Any]]:
        """
        按 Worker 权限取出对应的工具 Schema
        - allowed_workers 为空表示公共工具
        - allowed_workers 非空则只对该 Worker 可见
        """
        schemas = []
        for spec in self._tools.values():
            if not spec.allowed_workers or worker_name in spec.allowed_workers:
                schemas.append(spec.to_openai_schema())
        return schemas


tool_registry = ToolRegistry()


# ============ 参数模型 ============
class QueryOrderArgs(BaseModel):
    order_id: str = Field(..., description="订单ID，例如 ORD-123")


class InitiateRefundArgs(BaseModel):
    order_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0, le=100000, description="退款金额（元）")
    reason: str = Field(..., min_length=2, max_length=200)


class CreateTechTicketArgs(BaseModel):
    user_id: str
    description: str = Field(..., min_length=2)
    error_code: str = ""


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(..., min_length=1)


class ResetPasswordArgs(BaseModel):
    user_id: str
    new_password: str = Field(..., min_length=6)


# ============ 工具函数（示例，实际查 DB）============
def query_order_status(order_id: str) -> Dict[str, Any]:
    # 模拟：订单不存在的情况
    if order_id == "ORD-NOT-EXIST":
        return {"success": False, "error": "ORDER_NOT_FOUND", "message": "订单不存在"}
    return {"success": True, "order_id": order_id, "status": "已发货", "amount": 199.00, "user_id": "U001"}


def initiate_refund(order_id: str, amount: float, reason: str) -> Dict[str, Any]:
    return {"success": True, "order_id": order_id, "amount": amount, "message": f"退款 {amount} 元已提交"}


def create_tech_ticket(user_id: str, description: str, error_code: str = "") -> Dict[str, Any]:
    import uuid
    return {"success": True, "ticket_id": f"TECH-{uuid.uuid4().hex[:8]}", "user_id": user_id, "status": "OPEN"}


def search_knowledge_base(query: str) -> Dict[str, Any]:
    return {"success": True, "query": query, "results": []}


def reset_password(user_id: str, new_password: str) -> Dict[str, Any]:
    return {"success": True, "user_id": user_id, "message": "密码已重置"}


# ============ 注册（带权限）============
def register_all_tools():
    tool_registry.register(ToolSpec(
        name="query_order_status",
        description="查询订单状态、退款进度。输入订单ID，返回订单状态和金额",
        func=query_order_status,
        is_high_risk=False,
        args_schema=QueryOrderArgs,
        allowed_workers=["refund"],
    ))
    tool_registry.register(ToolSpec(
        name="initiate_refund",
        description="发起退款。高危操作，必须人工确认后调用",
        func=initiate_refund,
        is_high_risk=True,
        args_schema=InitiateRefundArgs,
        allowed_workers=["refund"],
    ))
    tool_registry.register(ToolSpec(
        name="create_tech_ticket",
        description="创建技术工单转研发。输入用户ID、问题描述、错误码",
        func=create_tech_ticket,
        is_high_risk=False,
        args_schema=CreateTechTicketArgs,
        allowed_workers=["tech"],
    ))
    tool_registry.register(ToolSpec(
        name="search_knowledge_base",
        description="RAG 检索技术知识库。输入查询关键词，返回相关文档",
        func=search_knowledge_base,
        is_high_risk=False,
        args_schema=SearchKnowledgeArgs,
        allowed_workers=["tech", "refund", "account"],
    ))
    tool_registry.register(ToolSpec(
        name="reset_password",
        description="重置用户密码。高危操作，必须人工确认后调用",
        func=reset_password,
        is_high_risk=True,
        args_schema=ResetPasswordArgs,
        allowed_workers=["account"],
    ))


register_all_tools()