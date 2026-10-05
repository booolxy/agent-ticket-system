# scripts/download_models.py
import os
from modelscope import snapshot_download

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def download_models():
    print("开始下载 BGE-M3 向量模型（约2GB）...")
    snapshot_download('BAAI/bge-m3', cache_dir=MODEL_DIR)

    print("开始下载 BGE-Reranker 精排模型（约1.3GB）...")
    snapshot_download('BAAI/bge-reranker-large', cache_dir=MODEL_DIR)

    print("✅ 所有模型下载完成！")


if __name__ == "__main__":
    download_models()