# app/domain/models.py
from sqlalchemy import Column, String, Integer, DECIMAL, DateTime, JSON, Text, func
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class Order(Base):
    """
    订单表：用于退款 Worker 查订单状态、金额
    """
    __tablename__ = "orders"

    order_id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False)          # 待支付、已支付、已发货、已退款
    amount = Column(DECIMAL(10, 2), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TechTicket(Base):
    """
    技术工单表：用于技术 Worker 创建研发工单
    """
    __tablename__ = "tech_tickets"

    ticket_id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), nullable=False)
    description = Column(Text, nullable=False)
    error_code = Column(String(64), nullable=True)
    status = Column(String(32), default="OPEN")          # OPEN, PROCESSING, CLOSED
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AgentState(Base):
    """
    Agent 状态表：用于 LangGraph Checkpointer 持久化
    体现简历中的：单向数据流、命名空间隔离、崩溃恢复
    """
    __tablename__ = "agent_states"

    thread_id = Column(String(64), primary_key=True, index=True)   # 会话/任务ID
    # 使用 JSONB 存储整个 AgentState 字典，支持索引和快速查询
    state_data = Column(JSON, nullable=False, default=dict)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KnowledgeChunk(Base):
    """
    RAG 知识库表：体现简历中的父子文档召回 + pgvector 向量存储
    """
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # BGE-M3 输出维度是 1024
    embedding = Column(Vector(1024), nullable=True)
    content = Column(Text, nullable=False)               # 子块内容（用于检索）
    parent_content = Column(Text, nullable=True)          # 父块内容（用于喂给 LLM）
    doc_metadata = Column(JSON, default=dict)             # 元数据（文档来源、类型等）