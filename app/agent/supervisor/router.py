# app/agent/supervisor/router.py
"""
Supervisor 主管节点：按拓扑排序调度 Worker

职责：
  - 从 execution_order 中找出下一个未执行的子任务
  - 返回 current_task_id 供主图路由
  - 所有子任务完成后，返回空字符串触发 Aggregate
"""
import logging
from typing import Dict, Any, Optional

from app.agent.graph.state import AgentState

logger = logging.getLogger(__name__)


class Supervisor:
    """主管节点：调度器"""

    def __init__(self, llm=None):
        # Supervisor 现在不需要 LLM，纯调度逻辑
        pass

    def next_task(self, state: AgentState) -> Optional[str]:
        """找出下一个未完成的子任务ID"""
        order = state.get("execution_order", [])
        completed = set(state.get("task_results", {}).keys())

        for tid in order:
            if tid not in completed:
                return tid

        return None  # 全部完成

    def get_task_by_id(self, state: AgentState, task_id: str) -> Optional[Dict]:
        """根据 task_id 查找子任务"""
        for task in state.get("plan", []):
            if task["id"] == task_id:
                return task
        return None

    def should_skip(self, state: AgentState, task: Dict) -> bool:
        """
        检查依赖是否都完成（拓扑排序已保证合法顺序，这里是双保险）
        """
        results = state.get("task_results", {})
        for dep in task.get("depends_on", []):
            if dep not in results:
                logger.warning(f"任务 {task['id']} 的依赖 {dep} 未完成，跳过")
                return True
        return False