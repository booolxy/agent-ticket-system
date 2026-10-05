# app/agent/graph/state.py
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from operator import add

def merge_dict(left: dict, right: dict) -> dict:
    """
    命名空间隔离核心：合并只做浅拷贝覆盖，
    保证 Worker 只能写自己的命名空间字段。
    """
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}


class AgentState(TypedDict):
    """
    全局共享状态 - 严格遵循单向数据流
    """
    # ============ 全局共享字段 ============
    user_query: str                                 # 用户原始问题
    intent: str                                     # Supervisor 路由标签
    messages: Annotated[list, add_messages]         # 对话历史

    # ============ 命名空间隔离：每个 Worker 只写自己的字段 ============
    # 退款 Worker 只写 refund_result
    refund_result: Annotated[Dict[str, Any], merge_dict]
    # 技术 Worker 只写 tech_result
    tech_result: Annotated[Dict[str, Any], merge_dict]
    # 账户 Worker 只写 account_result
    account_result: Annotated[Dict[str, Any], merge_dict]

    # ============ Human-in-the-loop 审批状态 ============
    pending_approval: Annotated[Dict[str, Any], merge_dict]  # 高危操作挂起信息
    need_human: bool                                         # 是否需要人工介入

    # ============ 汇总输出 ============
    final_response: str                             # 最终给用户的回复