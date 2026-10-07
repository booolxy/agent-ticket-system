# app/agent/planner/planner.py
"""
Planner 节点：意图识别 + 任务拆解

输入：user_query
输出：plan（DAG）、execution_order（拓扑排序结果）
"""
import json
import logging
from typing import Dict, Any, List

from app.core.security import SECURITY_SYSTEM_PROMPT, safe_llm_invoke
from app.agent.planner.topological_sort import topological_sort

logger = logging.getLogger(__name__)


# =========================================================
# Planner Prompt（强制 JSON 输出）
# =========================================================
PLANNER_PROMPT = """
你是一个任务规划专家。请分析用户问题，将其拆解成多个子任务，输出 JSON。

【拆解规则】
1. 每个子任务包含四个字段：id、intent、query、depends_on
2. intent 只能是以下三个之一：
   - refund（退款、退货、查退款进度）
   - tech（系统报错、接口异常、页面白屏）
   - account（重置密码、改绑手机、查余额）
3. depends_on 是依赖的子任务 id 数组，无依赖时为空数组 []
4. 如果用户问题是单一意图，输出只包含一个任务的数组
5. 严格输出 JSON 数组，不要输出任何解释、注释、markdown 标记

【示例 1】无依赖
输入："我要退款，顺便查一下我的余额"
输出：
[
  {{"id": "t1", "intent": "refund", "query": "我要退款", "depends_on": []}},
  {{"id": "t2", "intent": "account", "query": "查一下我的余额", "depends_on": []}}
]

【示例 2】有依赖
输入："帮我重置密码，然后查一下订单 ORD-123 能不能退"
输出：
[
  {{"id": "t1", "intent": "account", "query": "重置密码", "depends_on": []}},
  {{"id": "t2", "intent": "refund", "query": "查订单 ORD-123 能不能退", "depends_on": ["t1"]}}
]

【示例 3】单一意图
输入："订单 ORD-123 我要退款"
输出：
[
  {{"id": "t1", "intent": "refund", "query": "订单 ORD-123 我要退款", "depends_on": []}}
]

用户问题：{query}
"""


def _clean_json(raw: str) -> str:
    """去掉 markdown 代码块标记"""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _fallback_plan(query: str) -> List[Dict[str, Any]]:
    """LLM 解析失败时的兜底：单任务"""
    return [{"id": "t1", "intent": "unknown", "query": query, "depends_on": []}]


class Planner:
    """意图识别 + 任务拆解"""

    def __init__(self, llm):
        self.llm = llm

    def plan(self, query: str) -> Dict[str, Any]:
        """
        :return: {"plan": [...], "execution_order": [...]}
        """
        prompt = PLANNER_PROMPT.format(query=query)

        # 【安全层 2 + 3】safe_llm_invoke 自动注入安全提示 + 输出检查
        raw = safe_llm_invoke(self.llm, prompt)
        cleaned = _clean_json(raw)

        try:
            plan = json.loads(cleaned)
            if not isinstance(plan, list) or not plan:
                raise ValueError("Plan 不是非空数组")
            # 校验每个任务的字段
            for t in plan:
                if not all(k in t for k in ("id", "intent", "query", "depends_on")):
                    raise ValueError(f"任务缺少字段：{t}")
                if t["intent"] not in ("refund", "tech", "account", "unknown"):
                    t["intent"] = "unknown"
                if not isinstance(t["depends_on"], list):
                    t["depends_on"] = []
        except Exception as e:
            logger.warning(f"Planner 解析失败，降级为单任务：{e}\n原始输出：{raw[:200]}")
            plan = _fallback_plan(query)

        # ---------- 拓扑排序 ----------
        try:
            order = topological_sort(plan)
        except ValueError as e:
            logger.error(f"拓扑排序失败（可能成环）：{e}，降级为原始顺序")
            order = [t["id"] for t in plan]

        logger.info(f"Planner 输出：plan={plan}, order={order}")
        return {"plan": plan, "execution_order": order}