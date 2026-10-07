# app/rag/chunker.py
"""
文档切片器：实现 Small-to-Big 策略

核心思想：
  - 子块（child chunk）：用于**检索**，小而精准
  - 父块（parent chunk）：用于**喂给 LLM**，大而完整

流程：
  Raw Document
    ↓ 按段落切
  父块（parent，约 500-800 字）
    ↓ 每个父块再细分
  子块（child，约 100-200 字）
"""
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    parent_content: str
    metadata: Dict[str, Any]


class Chunker:
    """文档切片器（父块 + 子块双层）"""

    def __init__(
        self,
        parent_chunk_size: int = 800,
        child_chunk_size: int = 200,
        parent_overlap: int = 100,
        child_overlap: int = 30,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap

    @staticmethod
    def _split_by_separator(text: str, size: int, overlap: int) -> List[str]:
        """按标点切分（中文优先），保留上下文重叠"""
        # 优先在句末切分
        separators = ["。", "！", "？", "\n\n", "\n", "；", "，"]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            # 尝试在分隔符处断开
            if end < len(text):
                for sep in separators:
                    idx = text.rfind(sep, start + size // 2, end + 20)
                    if idx != -1:
                        end = idx + len(sep)
                        break
            chunks.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end
        return [c for c in chunks if c]

    def split(self, raw_text: str, doc_metadata: Dict[str, Any] = None) -> List[Chunk]:
        """切片主流程"""
        doc_metadata = doc_metadata or {}

        # 1. 先切父块
        parent_chunks = self._split_by_separator(
            raw_text, self.parent_chunk_size, self.parent_overlap
        )

        # 2. 每个父块再切子块（子块继承父块内容）
        result: List[Chunk] = []
        for p_idx, parent in enumerate(parent_chunks):
            child_chunks = self._split_by_separator(
                parent, self.child_chunk_size, self.child_overlap
            )
            for c_idx, child in enumerate(child_chunks):
                result.append(Chunk(
                    content=child,
                    parent_content=parent,
                    metadata={
                        **doc_metadata,
                        "parent_index": p_idx,
                        "child_index": c_idx,
                    },
                ))
        return result


# 全局单例
chunker = Chunker()