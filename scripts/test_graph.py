# scripts/test_graph.py
from app.core.llm_factory import get_large_llm, get_small_llm
from app.agent.graph.main_graph import MainGraph

large_llm = get_large_llm()
small_llm = get_small_llm()
graph = MainGraph(llm=large_llm, small_llm=small_llm)


def run(query: str):
    print(f"\n{'=' * 70}")
    print(f"🔍 用户Query: {query}")
    print(f"{'=' * 70}")

    config = {"configurable": {"thread_id": f"test-{hash(query) % 10000}"}}
    result = graph.graph.invoke({"user_query": query}, config=config)

    # 打印 Planner 的 DAG
    print(f"\n📋 Planner 拆解：")
    for t in result.get("plan", []):
        print(f"  - {t['id']}: [{t['intent']}] {t['query']} | depends_on={t['depends_on']}")

    print(f"\n🔀 拓扑排序执行顺序：{result.get('execution_order', [])}")

    print(f"\n✅ 子任务结果：")
    for tid, r in result.get("task_results", {}).items():
        print(f"  - {tid}: {str(r)[:100]}...")

    print(f"\n💬 最终回复：{result.get('final_response', '')}")
    print(f"🚨 是否需人工：{result.get('need_human', False)}")


if __name__ == "__main__":
    # 测试 1：单意图
    run("订单 ORD-123 我要退款")

    # 测试 2：多意图无依赖
    run("我要退款，顺便查一下我的余额")

    # 测试 3：多意图有依赖（拓扑排序生效）
    run("帮我重置密码，然后查一下订单 ORD-123 能不能退")

    # 测试 4：三意图复杂场景
    run("系统报500错误，顺便帮我退款，再查一下我的账户余额")