# app/agent/graph/state.py
"""
Agent 全局状态定义

工程化要点：
  1. 单向数据流：每个节点只写自己的命名空间字段
  2. 命名空间隔离：refund_result / tech_result / account_result 物理隔离
  3. 自定义 Reducer：merge_dict 保证合并是浅拷贝覆盖，不产生并发写冲突
  4. Planner 字段：plan（DAG）、execution_order（拓扑排序结果）
  5. Supervisor 字段：current_task_id、task_results
  6. Human-in-the-loop：pending_approval、need_human
"""
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages


# =========================================================
# 自定义 Reducer：命名空间隔离核心
# =========================================================
def merge_dict(left: dict, right: dict) -> dict:
    """
    合并字典只做浅拷贝覆盖。
    保证每个 Worker 只能写自己的命名空间字段，不会覆盖其他 Worker 的数据。
    """
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}


# =========================================================
# Planner 输出的子任务结构
# =========================================================
class SubTask(TypedDict):
    """
    Planner 拆解出的单个子任务

    示例：
        {
            "id": "t1",
            "intent": "account",
            "query": "重置密码",
            "depends_on": []
        }
    """
    id: str                    # 任务ID，如 t1, t2
    intent: str                # refund / tech / account / unknown
    query: str                 # 该子任务的用户问题描述
    depends_on: List[str]      # 依赖的前置任务ID列表


# =========================================================
# 全局共享状态
# =========================================================
class AgentState(TypedDict):
    """
    严格遵循单向数据流的全局状态。
    任何节点只允许写自己负责的字段。
    """

    # ============ 全局输入 ============
    user_query: str                                    # 用户原始问题
    messages: Annotated[list, add_messages]            # 对话历史

    # ============ Planner 输出 ============
    plan: List[SubTask]                                # DAG 任务列表
    execution_order: List[str]                         # 拓扑排序后的执行顺序

    # ============ Supervisor 调度状态 ============
    current_task_id: str                               # 当前正在执行的子任务ID
    task_results: Annotated[Dict[str, Any], merge_dict]  # task_id -> 子任务结果

    # ============ Worker 命名空间隔离 ============
    # 退款 Worker 只写 refund_result
    refund_result: Annotated[Dict[str, Any], merge_dict]
    # 技术 Worker 只写 tech_result
    tech_result: Annotated[Dict[str, Any], merge_dict]
    # 账户 Worker 只写 account_result
    account_result: Annotated[Dict[str, Any], merge_dict]

    # ============ Human-in-the-loop ============
    pending_approval: Annotated[Dict[str, Any], merge_dict]  # 高危操作挂起信息
    need_human: bool                                          # 是否需要人工介入

    # ============ 最终输出 ============
    final_response: str                                # 最终给用户的回复