# app/rag/engine.py
"""
RAG 引擎：完整的五层架构

  第 1 层：查询理解（HyDE 改写）
  第 2 层：多路并行召回（BM25 + 向量）
  第 3 层：RRF 融合 + BGE-Reranker 精排
  第 4 层：Small-to-Big 父块召回
  第 5 层：Redis 缓存加速
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.redis_client import rag_cache, hyde_cache
from app.domain.models import KnowledgeChunk

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        logger.info("加载 BGE-M3 向量模型...")
        self.embed_model = BGEM3FlagModel(settings.MODEL_BGE_M3_PATH, use_fp16=True)

        logger.info("加载 BGE-Reranker 精排模型...")
        self.reranker = FlagReranker(settings.MODEL_RERANKER_PATH, use_fp16=True)

        # BM25 内存索引
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.chunk_ids: List[int] = []
        self.chunk_map: Dict[int, Dict] = {}

    # ==================== 第 1 层：HyDE 查询改写 ====================
    def hyde_rewrite(self, small_llm, query: str) -> str:
        """
        HyDE（Hypothetical Document Embeddings）：
          让小模型生成一个"假想答案"，用假想答案去检索，
          缩小用户口语化 Query 与知识库文档之间的语义鸿沟。
        """
        # 先查缓存
        cached = hyde_cache.get(query)
        if cached:
            return cached

        prompt = (
            "请用一句专业、简洁的话回答下面的问题，只输出答案本身，不要解释：\n"
            f"{query}"
        )
        try:
            hypothetical = small_llm.invoke(prompt).content.strip()
        except Exception:
            hypothetical = query

        hyde_cache.set(query, hypothetical)
        return hypothetical

    # ==================== BM25 索引构建 ====================
    def build_bm25_index(self, db: Session):
        """从 DB 加载所有子块，构建 BM25 内存索引"""
        chunks = db.query(KnowledgeChunk).all()
        self.corpus = [c.content for c in chunks]
        self.chunk_ids = [c.id for c in chunks]
        self.chunk_map = {
            c.id: {"content": c.content, "parent": c.parent_content}
            for c in chunks
        }
        tokenized = [list(c) for c in self.corpus]  # 中文按字切，可换 jieba
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    # ==================== 第 2 层：并行多路召回 ====================
    def _vector_search(self, db: Session, query: str, top_k: int = 10) -> List[Dict]:
        """向量召回"""
        vec = self.embed_model.encode(query)["dense_vecs"]
        rows = (
            db.query(KnowledgeChunk)
            .order_by(KnowledgeChunk.embedding.cosine_distance(vec))
            .limit(top_k)
            .all()
        )
        return [
            {"id": r.id, "content": r.content, "parent": r.parent_content}
            for r in rows
        ]

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 关键词召回"""
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(list(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"id": self.chunk_ids[i], "content": self.corpus[i],
             "parent": self.chunk_map.get(self.chunk_ids[i], {}).get("parent", "")}
            for i in top_idx
        ]

    def parallel_recall(self, db: Session, query: str, top_k: int = 10) -> Dict[str, List[Dict]]:
        """
        并行执行 BM25 和向量召回，降低延迟
        """
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_vec = pool.submit(self._vector_search, db, query, top_k)
            f_bm25 = pool.submit(self._bm25_search, query, top_k)
            return {
                "vector": f_vec.result(),
                "bm25": f_bm25.result(),
            }

    # ==================== 第 3 层：RRF 融合 ====================
    @staticmethod
    def rrf_fusion(
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        RRF（Reciprocal Rank Fusion）：
          score(d) = Σ 1 / (k + rank_i(d))
        """
        # 用 id 作为唯一标识（因为两路召回的 item 结构可能不同）
        scores: Dict[int, float] = {}
        content_map: Dict[int, Dict] = {}

        for rank, item in enumerate(vector_results):
            scores[item["id"]] = scores.get(item["id"], 0) + 1 / (k + rank + 1)
            content_map[item["id"]] = item

        for rank, item in enumerate(bm25_results):
            scores[item["id"]] = scores.get(item["id"], 0) + 1 / (k + rank + 1)
            content_map.setdefault(item["id"], item)

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {**content_map[cid], "rrf_score": s}
            for cid, s in sorted_ids
            if cid in content_map
        ]

    # ==================== 第 3 层：Rerank 精排 ====================
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """Cross-Encoder 精排"""
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.reranker.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    # ==================== 主入口：完整检索流程 ====================
    def retrieve(
        self,
        db: Session,
        query: str,
        small_llm=None,
        use_hyde: bool = True,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        完整检索流程：
          1. HyDE 改写（可选）
          2. Redis 缓存查询
          3. 并行召回（BM25 + 向量）
          4. RRF 融合
          5. Rerank 精排
          6. Small-to-Big 返回父块
        """
        # ---------- 缓存命中 ----------
        cache_key = f"{query}|hyde={use_hyde}"
        cached = rag_cache.get(cache_key)
        if cached:
            logger.info(f"🎯 RAG 缓存命中：{query[:30]}")
            return cached

        # ---------- 1. HyDE 改写 ----------
        search_query = query
        if use_hyde and small_llm:
            search_query = self.hyde_rewrite(small_llm, query)

        # ---------- 2. 并行召回 ----------
        recall = self.parallel_recall(db, search_query, top_k=10)

        # ---------- 3. RRF 融合 ----------
        fused = self.rrf_fusion(recall["vector"], recall["bm25"], top_n=10)

        # ---------- 4. Rerank 精排 ----------
        reranked = self.rerank(query, fused[:5], top_k=top_k)

        # ---------- 5. Small-to-Big：返回父块 ----------
        final = []
        for item in reranked:
            cid = item["id"]
            parent = self.chunk_map.get(cid, {}).get("parent", item["content"])
            final.append({
                "id": cid,
                "content": item["content"],
                "parent_content": parent,
                "rrf_score": item.get("rrf_score", 0.0),
                "rerank_score": item.get("rerank_score", 0.0),
            })

        # ---------- 6. 写缓存 ----------
        rag_cache.set(cache_key, final)
        return final