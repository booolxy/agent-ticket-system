# app/agent/planner/topological_sort.py
"""
拓扑排序（Kahn 算法）
输入：带 depends_on 的 DAG 任务列表
输出：合法的执行顺序
若存在环，抛出 ValueError
"""
from collections import defaultdict, deque
from typing import List, Dict, Any


def topological_sort(tasks: List[Dict[str, Any]]) -> List[str]:
    """
    Kahn 算法：
      1. 统计每个节点的入度
      2. 入度为 0 的入队
      3. 依次出队，将邻居入度 -1，为 0 则入队
      4. 如果输出节点数 < 总数，说明有环
    """
    if not tasks:
        return []

    task_ids = {t["id"] for t in tasks}
    graph: Dict[str, List[str]] = defaultdict(list)
    in_degree: Dict[str, int] = {t["id"]: 0 for t in tasks}

    for task in tasks:
        for dep in task.get("depends_on", []):
            if dep not in task_ids:
                raise ValueError(f"任务 {task['id']} 依赖的 {dep} 不存在")
            graph[dep].append(task["id"])
            in_degree[task["id"]] += 1

    # 入度为 0 的节点入队
    queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
    order: List[str] = []

    while queue:
        tid = queue.popleft()
        order.append(tid)
        for neighbor in graph[tid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(tasks):
        raise ValueError("DAG 中存在环，无法拓扑排序")

    return order