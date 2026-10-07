# app/agent/tools/executor.py
import json
from typing import Dict, Any
from pydantic import ValidationError

from app.agent.tools.registry import tool_registry

MAX_TOOL_FAILURES = 2
MAX_REFLECTION_RETRIES = 2


class ToolExecutionResult:
    def __init__(self, success: bool, result: Any = None, error: str = "",
                 need_reflection: bool = False, need_human: bool = False):
        self.success = success
        self.result = result
        self.error = error
        self.need_reflection = need_reflection
        self.need_human = need_human

    def to_dict(self):
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "need_reflection": self.need_reflection,
            "need_human": self.need_human,
        }


class ToolExecutor:
    def execute(self, tool_name: str, raw_args: Dict[str, Any],
                failures: Dict[str, int]) -> ToolExecutionResult:
        # 0. 熔断检查
        if failures.get(tool_name, 0) >= MAX_TOOL_FAILURES:
            return ToolExecutionResult(
                success=False,
                error=f"工具 {tool_name} 已熔断（连续失败 {MAX_TOOL_FAILURES} 次）",
                need_human=True,
            )

        # 1. 从注册表获取工具
        try:
            spec = tool_registry.get(tool_name)
        except KeyError as e:
            failures[tool_name] = failures.get(tool_name, 0) + 1
            return ToolExecutionResult(success=False, error=str(e), need_human=True)

        # 2. 参数校验
        if spec.args_schema:
            try:
                validated = spec.args_schema(**raw_args)
                args = validated.model_dump()
            except ValidationError as e:
                failures[tool_name] = failures.get(tool_name, 0) + 1
                return ToolExecutionResult(
                    success=False,
                    error=f"参数校验失败：{e.errors()}",
                    need_reflection=True,
                )
        else:
            args = raw_args

        # 3. 执行
        try:
            result = spec.func(**args)
        except Exception as e:
            failures[tool_name] = failures.get(tool_name, 0) + 1
            return ToolExecutionResult(
                success=False,
                error=f"工具执行异常：{type(e).__name__}: {str(e)}",
                need_reflection=True,
            )

        # 4. 业务错误
        if isinstance(result, dict) and result.get("success") is False:
            return ToolExecutionResult(
                success=False,
                result=result,
                error=result.get("message", "业务处理失败"),
                need_human=True,
            )

        # 5. 成功
        failures[tool_name] = 0
        return ToolExecutionResult(success=True, result=result)


tool_executor = ToolExecutor()


def build_reflection_message(tool_name: str, error: str) -> str:
    return (
        f"你上一次调用工具 `{tool_name}` 时出错：{error}。\n"
        f"请重新生成合法的参数 JSON，只输出 JSON，不要解释。"
    )