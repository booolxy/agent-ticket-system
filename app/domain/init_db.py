# app/domain/init_db.py
from app.core.database import engine, Base
from app.domain import models   # 导入所有模型，让 Base 知道它们的存在
from sqlalchemy import text


def init_db():
    print("⏳ 正在创建数据库表...")
    # 开启 pgvector 扩展（确保万无一失）
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表创建成功！")


if __name__ == "__main__":
    init_db()