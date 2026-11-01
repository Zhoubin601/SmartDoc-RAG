# -*- coding: utf-8 -*-
"""
全局配置文件与环境变量加载
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 获取 Code 目录绝对路径
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env 环境变量文件（优先加载 Code/.env，若无则尝试上一级）
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ============================
# 大模型配置 (OpenAI 兼容接口)
# ============================
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-flash")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# ============================
# 向量与 Embedding 配置
# ============================
DEFAULT_EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
DEFAULT_EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# ============================
# RAG 文档处理与检索配置
# ============================
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))

# ============================
# 本地持久化与缓存路径
# ============================
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
