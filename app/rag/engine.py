# app/rag/engine.py
from typing import List, Dict
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from sqlalchemy.orm import Session
from app.core.config import settings
from app.domain.models import KnowledgeChunk


class RAGEngine:
    def __init__(self):
        # 1. 加载 BGE-M3 向量模型 (用于向量化)
        self.embed_model = BGEM3FlagModel(settings.MODEL_BGE_M3_PATH, use_fp16=True)
        # 2. 加载 BGE-Reranker 精排模型
        self.reranker = FlagReranker(settings.MODEL_RERANKER_PATH, use_fp16=True)
        # 3. BM25 索引内存缓存
        self.bm25 = None
        self.corpus = []
        self.chunk_map = {}

    def build_bm25_index(self, db: Session):
        """从数据库加载所有子块，构建 BM25 索引"""
        chunks = db.query(KnowledgeChunk).all()
        self.corpus = [c.content for c in chunks]
        self.chunk_map = {c.id: {"parent": c.parent_content, "content": c.content} for c in chunks}
        # 中文按字切分（工程上可替换为 jieba 分词）
        tokenized_corpus = [list(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def vector_search(self, db: Session, query: str, top_k: int = 10) -> List[Dict]:
        """向量检索：使用 pgvector 余弦距离"""
        query_vector = self.embed_model.encode(query)['dense_vecs']
        # 使用 pgvector 的 <=> 操作符（余弦距离）
        results = db.query(KnowledgeChunk).order_by(
            KnowledgeChunk.embedding.cosine_distance(query_vector)
        ).limit(top_k).all()
        return [{"id": r.id, "content": r.content, "parent": r.parent_content} for r in results]

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 关键词检索"""
        if not self.bm25: return []
        tokenized_query = list(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"id": idx, "content": self.corpus[idx], "score": scores[idx]} for idx in top_indices]

    def rrf_fusion(self, vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        """RRF 融合算法（Reciprocal Rank Fusion）"""
        rrf_scores = {}
        # 向量结果权重
        for rank, item in enumerate(vector_results):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0) + 1 / (k + rank + 1)
        # BM25 结果权重
        for rank, item in enumerate(bm25_results):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0) + 1 / (k + rank + 1)

        # 按 RRF 分数降序排列
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        # 找出对应的内容，以便精排
        fused = []
        for idx, score in sorted_items[:10]:
            if idx in self.chunk_map:
                fused.append({"id": idx, "content": self.chunk_map[idx]["content"], "rrf_score": score})
        return fused

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Cross-Encoder 精排"""
        if not candidates: return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.reranker.compute_score(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = scores[i]
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    def retrieve(self, db: Session, query: str) -> List[Dict]:
        """对外暴露的统一检索接口：粗排 + 精排 + 父子召回"""
        # 1. 多路召回
        vec_res = self.vector_search(db, query)
        bm25_res = self.bm25_search(query)

        # 2. RRF 融合
        fused = self.rrf_fusion(vec_res, bm25_res)

        # 3. Cross-Encoder 精排
        reranked = self.rerank(query, fused[:5])  # 取融合后的前5个做精排

        # 4. 父子文档召回：返回父块内容（给 LLM 用）
        final_results = []
        for r in reranked[:3]:
            c_id = r["id"]
            parent_content = self.chunk_map.get(c_id, {}).get("parent", r["content"])
            final_results.append({
                "content": r["content"],
                "parent_content": parent_content,
                "score": r.get("rerank_score", 0)
            })
        return final_results