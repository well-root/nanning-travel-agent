#配置所有需要的东西
from pathlib import Path
#用flie来定位
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"

# Qdrant
QDRANT_URL="http://localhost:6333"
COLLECTION="nanning_docs"

#模型
EMBED_MODEL="text-embedding-v4"
EMBED_DIM=1024
EMBED_BATCH=10 #每批的上限

CHUNK_MAX = 600
CHUNK_OVERLAP = 80
MIN_CHUNK = 10                       #最短子块

#检索
RETRIEVAL_K = 4
SCORE_THRESHOLD = 0.60              #混合检索的阈值


#DocLoader 校验用
REQUIRED_META = ("title", "source", "source_type", "fetched_at")
