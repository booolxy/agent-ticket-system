# scripts/test_rag.py
from app.core.database import SessionLocal
from app.core.llm_factory import get_small_llm
from app.rag.engine import RAGEngine


def main():
    db = SessionLocal()
    engine = RAGEngine()
    engine.build_bm25_index(db)
    small_llm = get_small_llm()

    # 测试用例
    queries = [
        "我要退款，已经发货了怎么办？",      # 测 HyDE + 混合召回
        "页面白屏了",                        # 测 BM25 精确匹配
        "手机号不用了，怎么重置密码",         # 测 small-to-big
    ]

    for q in queries:
        print(f"\n{'=' * 60}")
        print(f"🔍 Query: {q}")
        print(f"{'=' * 60}")

        results = engine.retrieve(db, q, small_llm=small_llm, use_hyde=True, top_k=2)

        for i, r in enumerate(results, 1):
            print(f"\n【Top {i}】")
            print(f"  子块: {r['content'][:60]}...")
            print(f"  父块: {r['parent_content'][:100]}...")
            print(f"  RRF 分: {r['rrf_score']:.4f} | Rerank 分: {r['rerank_score']:.4f}")


if __name__ == "__main__":
    main()