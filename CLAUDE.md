# nanning-travel-agent 项目规范

南宁出行助手：LangGraph 编排两个工具——自研 nanning-metro-mcp（换乘查询，stdio 外挂）与本地文档 RAG（票价/优惠/线路/运营时间），配套评测与 A/B 对比。MVP 目标 2026-11-15。

## 目录约定

| 目录 | 内容 | 规则 |
|---|---|---|
| `src/nanning_travel_agent/` | 包代码：`app.py`（CLI 入口）、`agent/`（建图）、`rag/`（ingest 与检索工具）、`mcp/`（MCP 接入） | 业务代码只写在这里 |
| `data/docs/` | 语料 md，每篇必须有 frontmatter：`source` / `source_type` / `fetched_at` | 新增语料前先登记 `data/SOURCES.md` |
| `data/SOURCES.md` | 语料来源与局限性登记表 | 诚信红线的一部分，随语料同步更新 |
| `data/raw/` | 原始存档（官网 HTML、公告图片、门户文章） | **不进索引、不提交 git** |
| `scripts/` | 一次性脚本：数据生成、连通性测试、检索验收 | 可 import 包内模块，但必须能 `uv run python scripts/xxx.py` 一条命令独立跑通 |
| `eval/` | 评测集（queries.jsonl）与判分脚本 | 每条评测含期望工具/期望答案 |
| `docs/` | 学习笔记、自测题答案、截图 | 自由格式 |

## 环境与命令

- 包管理一律 **uv**：装依赖 `uv add <pkg>`；运行一律 `uv run ...`，不用系统 python
- Qdrant：Docker 容器（名 `qdrant`，端口 6333，storage 挂载到 `qdrant_storage/`）
  - 首次创建（镜像已 pull，容器尚未建过）：`docker run -d --name qdrant -p 6333:6333 -v D:/agent/nanning-travel-agent/qdrant_storage:/qdrant/storage qdrant/qdrant`
  - 之后启动：`docker start qdrant`；健康检查：`curl.exe -s http://localhost:6333/readyz`（PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名，必须写 `curl.exe`）
- Embedding（主）：阿里云百炼 `text-embedding-v4`，1024 维，OpenAI 兼容接口（`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`）。**每批最多 10 条**（`chunk_size=10`），且 `check_embedding_ctx_length=False` 绕开 tiktoken
- Embedding（备用/对照）：`BAAI/bge-small-zh-v1.5`（512 维，本地已缓存），仅作离线兜底与 12 月对照实验
- LLM：DeepSeek `deepseek-chat`，OpenAI 兼容 `base_url=https://api.deepseek.com`，**temperature=0**（评测可复现）
- 密钥放根目录 `.env`（`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`；`HF_ENDPOINT` 只在下载本地 bge 时需要），用 python-dotenv 加载；注意 `uv run` **不会自动读 `.env`**，一次性命令要用 `uv run --env-file .env ...`，或在 PowerShell 里先 `$env:HF_ENDPOINT='https://hf-mirror.com'`

## 验证命令

```bash
uv run python scripts/test_mcp_direct.py    # MCP 直调：朝阳广场→火车东站（1号线13站）；新民路→南宁剧场（朝阳广场换乘）
uv run python scripts/test_retrieval.py     # 检索验收：5 条政策 query 全部命中正确文档
uv run python -m nanning_travel_agent.rag.ingest   # 重建索引
uv run python -m nanning_travel_agent.app   # CLI 问答
```

> 站名注意：地铁站名是「**火车东站**」；「南宁东站」是高铁站，不在 metro 数据里，传进去会返回「站名不存在」。这是 metro-mcp 故意保留的别名用例，不是接线错误。

改动后先跑对应验证再继续，不要只改不验。

## 红线

1. API key 不进代码、不进 commit、不进日志
2. `.env`、`data/raw/`、`qdrant_storage/` 不提交 git
3. 来源不明的语料不入库；票价/时刻等数值内容以官方来源为准，转述来源（如维基）必须在 frontmatter 标注 `source_type`
4. `D:\agent\rag_and_langgrah\agentic-rag-for-dummies-main` 仅为阅读参考，**不复制其代码**；差异决策记录在 README 的对照表里
5. 语料 schema（切块结构、collection 命名、metadata 字段）、**embedding 模型或维度**变更：先改本文档，再改代码；换 embedding 必须 `force_recreate` 重建 collection，旧阈值作废重测

## 范围基线

- MVP 内：双工具路由 CLI、15 条四类评测、A/B 两组（MCP-only vs 双工具）、README（含差异决策表与失败分析）
- MVP 外（12 月投递后再加）：HITL 澄清、滚动摘要记忆、RAGAS、并行子图、Web UI、HYBRID 检索
