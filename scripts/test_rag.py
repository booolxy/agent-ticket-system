# scripts/test_rag.py
from app.core.database import SessionLocal
from app.rag.engine import RAGEngine

def main():
    db = SessionLocal()
    engine = RAGEngine()
    # 构建 BM25 内存索引
    engine.build_bm25_index(db)

    query = "我要退款，已经发货了怎么办？"
    print(f"\n🔍 用户提问: {query}")
    results = engine.retrieve(db, query)

    print("\n✅ 最终返回结果（父子文档召回）:")
    for r in results:
        print(f"- 子块: {r['content']}")
        print(f"  父块: {r['parent_content']}")
        print(f"  精排得分: {r['score']:.4f}\n")

if __name__ == "__main__":
    main()