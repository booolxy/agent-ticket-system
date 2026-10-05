# 智能工单助手 Agent 系统

基于 LangGraph 的多 Agent 协作工单处理系统，支持退款、技术支持、账户操作三大业务场景。

## 架构总览

用户请求 -> 令牌桶限流 -> 正则规则引擎 -> 小模型分流 -> Supervisor 意图路由 -> 三个 Worker -> 汇总节点 -> PostgresSaver -> 返回用户

## 核心技术亮点

| 模块 | 方案 |
|---|---|
| 多 Agent 编排 | LangGraph Supervisor-Worker 模式 |
| 状态管理 | 单向数据流 + 命名空间隔离 |
| 崩溃恢复 | PostgreSQL + PostgresSaver |
| RAG | BM25 + 向量 + RRF + Reranker + 父子召回 |
| 降本 | 正则 + 小模型分流，API 成本下降约 40% |
| 安全 | Human-in-the-loop 二次确认 + Docker 沙箱 |
| 可观测 | 四维评估（格式/质量/耗时/Token） |
| 微调 | QLoRA + Qwen-1.8B + 困难负样本 |

## 快速开始

1. docker-compose up -d
2. pip install -r requirements.txt
3. python -m app.domain.init_db
4. python -m scripts.init_knowledge
5. uvicorn app.api.main:app --reload --port 8000

打开 http://127.0.0.1:8000/docs 查看 Swagger 文档。

## 四维评估

python -m scripts.run_eval

## 目录结构

agent-ticket-system/
- app/          主应用
- evals/        评估数据集与报告
- finetune/     QLoRA 微调工程
- scripts/      初始化脚本
- tests/        测试

## License

MIT
