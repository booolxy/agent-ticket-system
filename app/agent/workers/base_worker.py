# app/agent/workers/base_worker.py
"""
BaseWorker：所有 Worker 的基类，封装 ReAct 循环

核心循环：
  LLM(tools) → 解析 tool_calls → 执行 → ToolMessage 喂回 → LLM → ... → 最终答案

关键机制：
  1. 高危工具拦截：不执行，写 pending_approval，直接返回 need_human
  2. 参数校验错误 → reflection 重试（最多 2 次）
  3. 工具执行异常 → 熔断计数（2 次 → 转人工）
  4. 业务逻辑错误（success=False）→ 转人工
  5. ReAct 循环上限 5 次 → 熔断转人工
"""
import json
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from app.agent.tools.registry import tool_registry
from app.agent.tools.executor import tool_executor, build_reflection_message


class BaseWorker:
    MAX_REACT_ITERATION = 5
    MAX_REFLECTION_RETRIES = 2
    WORKER_NAME = "base"

    def __init__(self, llm, system_prompt: str = ""):
        self.llm = llm
        self.system_prompt = system_prompt or (
            f"你是 {self.WORKER_NAME} Worker，负责处理相关业务。可以调用工具来完成任务。"
        )

    def _bind_tools(self):
        """从注册表按权限取出工具 Schema，绑定到 LLM"""
        schemas = tool_registry.get_schemas_for_worker(self.WORKER_NAME)
        return self.llm.bind_tools(schemas) if schemas else self.llm

    def react_loop(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 ReAct 循环
        返回：
          {
            final_answer: str,
            tool_trace: List[Dict],
            failures: Dict[str, int],
            need_human: bool,
            pending_approval: Dict,
          }
        """
        llm_with_tools = self._bind_tools()

        messages: List[Any] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=state["user_query"]),
        ]

        failures: Dict[str, int] = dict(state.get("tool_call_failures", {}) or {})
        tool_trace: List[Dict[str, Any]] = []
        reflection_retries: Dict[str, int] = {}

        for iteration in range(self.MAX_REACT_ITERATION):
            # 1. 调用 LLM
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # 2. 没有工具调用 → 输出最终答案
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return {
                    "final_answer": response.content,
                    "tool_trace": tool_trace,
                    "failures": failures,
                    "need_human": False,
                    "pending_approval": {},
                }

            # 3. 遍历 tool_calls
            for call in tool_calls:
                tool_name = call.get("name")
                raw_args = call.get("args", {}) or {}

                # 3.1 解析 args（LLM 可能返回 string）
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}

                # 3.2 高危工具拦截 → 不执行，挂起等人工
                try:
                    spec = tool_registry.get(tool_name)
                except KeyError:
                    spec = None

                if spec and spec.is_high_risk:
                    tool_trace.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "args": raw_args,
                        "blocked": "HIGH_RISK",
                    })
                    pending = {
                        "worker": self.WORKER_NAME,
                        "tool": tool_name,
                        "args": raw_args,
                        "status": "PENDING",
                        "trace": tool_trace,
                    }
                    return {
                        "final_answer": (
                            f"已为您准备好执行 {tool_name}，等待客服人工确认。"
                            f"参数：{json.dumps(raw_args, ensure_ascii=False)}"
                        ),
                        "tool_trace": tool_trace,
                        "failures": failures,
                        "need_human": True,
                        "pending_approval": pending,
                    }

                # 3.3 执行工具
                result = tool_executor.execute(tool_name, raw_args, failures)
                tool_trace.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "args": raw_args,
                    "result": result.to_dict(),
                })

                # 3.4 处理结果
                if result.success:
                    messages.append(ToolMessage(
                        content=json.dumps(result.result, ensure_ascii=False),
                        tool_call_id=call.get("id", ""),
                    ))
                elif result.need_reflection:
                    # 参数错误 / 工具异常 → 反射重试
                    reflection_retries[tool_name] = reflection_retries.get(tool_name, 0) + 1
                    if reflection_retries[tool_name] > self.MAX_REFLECTION_RETRIES:
                        return {
                            "final_answer": "抱歉，多次尝试后仍无法完成任务，已为您转人工。",
                            "tool_trace": tool_trace,
                            "failures": failures,
                            "need_human": True,
                            "pending_approval": {},
                        }
                    messages.append(ToolMessage(
                        content=build_reflection_message(tool_name, result.error),
                        tool_call_id=call.get("id", ""),
                    ))
                else:
                    # 熔断 或 业务错误 → 转人工
                    return {
                        "final_answer": f"抱歉，处理您的问题时遇到问题：{result.error}。已为您转人工客服。",
                        "tool_trace": tool_trace,
                        "failures": failures,
                        "need_human": True,
                        "pending_approval": {},
                    }

        # 4. 超过 MAX_REACT_ITERATION → 熔断
        return {
            "final_answer": "抱歉，任务处理超时，已为您转人工。",
            "tool_trace": tool_trace,
            "failures": failures,
            "need_human": True,
            "pending_approval": {},
        }