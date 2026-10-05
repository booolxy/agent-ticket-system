# scripts/test_graph.py
from langchain_openai import ChatOpenAI
from app.agent.graph.main_graph import MainGraph

# 临时用云端 LLM 做验证（后续可换微调后的 Qwen-1.8B）
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key="sk-xxx",       # 换成你自己的 Key
    base_url="https://api.openai.com/v1",
    temperature=0
)

graph = MainGraph(llm)
config = {"configurable": {"thread_id": "test-001"}}
result = graph.graph.invoke(
    {"user_query": "订单 ORD-123 我要退款"},
    config=config,
)

print("意图：", result.get("intent"))
print("退款结果：", result.get("refund_result"))
print("是否需人工：", result.get("need_human"))
print("最终回复：", result.get("final_response"))