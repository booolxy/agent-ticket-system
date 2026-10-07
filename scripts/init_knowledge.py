# scripts/init_knowledge.py
"""
知识库初始化：
  Raw Docs → Chunker 切片 → BGE-M3 向量化 → pgvector 入库
"""
from app.core.database import SessionLocal
from app.domain.models import KnowledgeChunk
from app.rag.chunker import chunker
from app.rag.engine import RAGEngine


# ==================== 原始文档（模拟真实业务文档）====================
RAW_DOCS = [
    {
        "doc_id": "refund_policy",
        "title": "退款政策说明书",
        "content": (
            "第一章 退款总则。"
            "客户在收到商品后7天内，商品未拆封且不影响二次销售的情况下，可申请无理由退款。"
            "已发货订单需先拒收或签收后申请退货退款，退款金额将扣除运费。"
            "退款申请提交后，一般在3-5个工作日内到账。"
            "第二章 特殊商品。"
            "定制类商品、生鲜类商品、虚拟商品一经售出，不支持七天无理由退款。"
            "第三章 退款金额计算。"
            "退款金额 = 订单实付金额 - 运费（若已发货）- 优惠券折算金额。"
            "如果订单使用了优惠券，退款时优惠券不予返还。"
        ),
    },
    {
        "doc_id": "tech_support",
        "title": "技术支持手册",
        "content": (
            "常见问题一：页面白屏。"
            "排查步骤：按F12查看Console报错，通常是前端JS加载失败或接口跨域。"
            "检查浏览器控制台的Network面板，确认接口是否返回200。"
            "常见问题二：系统报500错误。"
            "排查步骤：先检查后端服务日志，确认数据库连接是否正常。"
            "查看应用日志中的Stack Trace，定位具体异常。"
            "常见问题三：接口401未授权。"
            "排查步骤：检查Token是否过期，重新登录获取新Token。"
            "确认请求头中的Authorization字段格式正确。"
        ),
    },
    {
        "doc_id": "account_security",
        "title": "账户安全验证手册",
        "content": (
            "重置密码流程："
            "第一步，验证手机号是否可用。手机号可用则发送短信验证码，"
            "用户输入验证码后即可重置密码。"
            "第二步，手机号不可用时，进入身份证验证子流程。"
            "用户需上传身份证正反面照片，系统自动比对公安数据库。"
            "身份证核验通过后，进入人工审核环节，由客服确认身份后执行重置。"
            "安全提示：所有涉及账户敏感操作必须经过人工二次确认。"
        ),
    },
]


def init_knowledge():
    db = SessionLocal()
    engine = RAGEngine()

    total_chunks = 0
    for doc in RAW_DOCS:
        # 1. 切片（父子块）
        chunks = chunker.split(
            raw_text=doc["content"],
            doc_metadata={"doc_id": doc["doc_id"], "title": doc["title"]},
        )
        print(f"📄 {doc['title']}: 切出 {len(chunks)} 个子块")

        # 2. 批量向量化
        texts = [c.content for c in chunks]
        vectors = engine.embed_model.encode(texts, batch_size=8)["dense_vecs"]

        # 3. 入库
        for chunk, vec in zip(chunks, vectors):
            db.add(KnowledgeChunk(
                embedding=vec.tolist(),
                content=chunk.content,
                parent_content=chunk.parent_content,
                doc_metadata=chunk.metadata,
            ))
            total_chunks += 1

    db.commit()
    db.close()
    print(f"\n✅ 知识库初始化完成，共入库 {total_chunks} 个子块")


if __name__ == "__main__":
    init_knowledge()