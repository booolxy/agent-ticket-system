# scripts/init_knowledge.py
from app.core.database import SessionLocal
from app.domain.models import KnowledgeChunk
from app.rag.engine import RAGEngine


def init_knowledge():
    db = SessionLocal()
    engine = RAGEngine()

    # 模拟真实业务文档（子块 -> 父块）
    docs = [
        {"content": "七天无理由退款规则",
         "parent": "退款政策说明：客户在收到商品后7天内，商品未拆封且不影响二次销售的情况下，可申请无理由退款。"},
        {"content": "已发货订单退款规则",
         "parent": "退款政策说明：已发货订单需先拒收或签收后申请退货退款，退款金额将扣除运费。"},
        {"content": "系统报错 500 排查",
         "parent": "技术支持文档：遇到500错误，先检查后端服务日志，确认数据库连接是否正常。"},
        {"content": "页面白屏处理",
         "parent": "技术支持文档：遇到页面白屏，按F12查看Console报错，通常是前端JS加载失败或接口跨域。"},
        {"content": "重置密码验证流程", "parent": "账户安全手册：重置密码需验证手机号，若手机号不可用，需进行身份证核验。"},
        {"content": "身份证核验规则", "parent": "账户安全手册：身份证核验需上传身份证正反面，系统自动比对公安数据库。"},
    ]

    print("⏳ 正在向量化并写入数据库...")
    for doc in docs:
        # 使用 BGE-M3 生成 1024 维向量
        vector = engine.embed_model.encode(doc["content"])['dense_vecs'].tolist()
        chunk = KnowledgeChunk(
            embedding=vector,
            content=doc["content"],
            parent_content=doc["parent"],
            doc_metadata={"source": "simulated_doc"}
        )
        db.add(chunk)

    db.commit()
    db.close()
    print("✅ 知识库初始化完成！")


if __name__ == "__main__":
    init_knowledge()