# app/core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# 1. 创建 SQLAlchemy 引擎
# pool_pre_ping=True 保证从连接池拿到的连接是活的，防止 PG 断连报错
# pool_size=10, max_overflow=20 是生产级别参数，控制连接池大小
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False  # 设为 True 可以打印 SQL，调试时用
)

# 2. 创建 SessionLocal 类
# autocommit=False 保证手动提交事务
# autoflush=False 防止误刷新
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. 创建 Base 类
# 所有 ORM 模型（订单表、工单表、Agent状态表）都要继承它
Base = declarative_base()

# 4. FastAPI 依赖注入函数
# 用于在 API 路由中获取数据库会话，请求结束后自动关闭
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()