# app/core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    全局配置管理
    读取 .env 文件，提供类型安全的配置访问
    """
    # 项目基本信息
    PROJECT_NAME: str = "Agent Ticket System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # PostgreSQL 配置
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        """拼接 SQLAlchemy 需要的连接字符串（包含 pgvector）"""
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # 模型配置
    MODEL_BGE_M3_PATH: str = "./models/BAAI/bge-m3"
    MODEL_RERANKER_PATH: str = "./models/BAAI/bge-reranker-large"
    MODEL_QWEN_PATH: str = "./models/Qwen/Qwen-1.8B"

    # 大模型 API 配置
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # 加载 .env 文件
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # 忽略 .env 中多出的额外配置
    )

@lru_cache()
def get_settings() -> Settings:
    """
    使用 lru_cache 实现单例模式
    保证全局只有一个 Settings 实例，节省资源
    """
    return Settings()

# 全局配置实例，直接导入使用
settings = get_settings()
