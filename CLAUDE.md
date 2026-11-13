# nanning-travel-agent 项目规范

南宁出行助手：LangGraph 编排五个工具——自研 nanning-metro-mcp（换乘/站名/线路清单/线路站序，stdio 外挂、独立仓库）与本地文档 RAG（票价/优惠/运营时间/线路站序），配套规则判分与 RAGAS 评测。

## 目录约定

| 目录 | 内容 | 规则 |
|---|---|---|
| `src/nanning_travel_agent/` | `app.py`（CLI 入口）、`ui.py`（Streamlit）、`agent/`（建图、摘要压缩、工具预算）、`rag/`（ingest / retrieval / tools）、`metro_mcp/`（stdio 接入 4 个 MCP 工具） | 业务代码只写在这里 |
| `data/docs/` | 语料 md，每篇必须有 frontmatter：`source` / `source_type` / `fetched_at` | 新增语料前先登记 `data/SOURCES.md` |
| `data/SOURCES.md` | 语料来源与局限性登记表 | 诚信红线的一部分，随语料同步更新 |
| `data/raw/` | 原始存档（官网 HTML、公告图片、门户文章） | **不进索引、不提交 git** |
| `data/memory/` | `checkpoints.sqlite`，会话持久化 | **不提交 git** |
| `scripts/` | 一次性脚本：数据生成、连通性、检索验收、子图冒烟 | 必须能 `uv run python scripts/xxx.py` 一条命令独立跑通 |
| `eval/` | `queries.jsonl`（18 条）、`judge.py`（规则判分）、`ragas_eval.py`、`results/` | 每条评测含期望工具 + 关键词约束 |
| `docs/` | 学习笔记、自测题答案、截图 | 自由格式 |

## 环境与命令

- 包管理一律 **uv**：装依赖 `uv add <pkg>`；运行一律 `uv run ...`，不用系统 python
- Qdrant：Docker 容器（名 `qdrant`，端口 6333，storage 挂载到 `qdrant_storage/`）
  - 首次创建（镜像已 pull，容器尚未建过）：`docker run -d --name qdrant -p 6333:6333 -v D:/agent/nanning-travel-agent/qdrant_storage:/qdrant/storage qdrant/qdrant`
  - 之后启动：`docker start qdrant`；健康检查：`curl.exe -s http://localhost:6333/readyz`（PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名，必须写 `curl.exe`）
- Embedding（主）：阿里云百炼 `text-embedding-v4`，1024 维，OpenAI 兼容接口（`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`）。**每批最多 10 条**（`chunk_size=10`），且 `check_embedding_ctx_length=False` 绕开 tiktoken
- Embedding（备用/对照）：`BAAI/bge-small-zh-v1.5`（512 维，本地已缓存），仅作离线兜底，未接入主链路
- 重排：阿里云 `gte-rerank-v2`（dashscope 文本重排接口），失败自动退回 RRF 顺序
- LLM：DeepSeek `deepseek-flash`，`base_url=https://api.deepseek.com`，**temperature=0.1**（问答与评测统一口径）；RAGAS 的 judge 用 `temperature=0`
- 记忆：`AsyncSqliteSaver` 写 `data/memory/checkpoints.sqlite`；`thread_id` 区分会话，CLI 用 `cli-<USER_ID>`
- 密钥放根目录 `.env`（`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`；`HF_ENDPOINT` 只在下载本地 bge 时需要），用 python-dotenv 加载；注意 `uv run` **不会自动读 `.env`**，一次性命令要用 `uv run --env-file .env ...`
- MCP 服务路径通过环境变量覆盖：`NANNING_METRO_PYTHON`（解释器）、`NANNING_METRO_SERVER`（server.py）；默认指向 `D:\agent\nanning-metro-mcp`
- 其余可调项：`RETRIEVAL_MODE=vector|hybrid`、`RERANK=0|1`、`MAX_TOOL_ROUNDS=6`、`SUMMARY_THRESHOLD=10`

## 验证命令

```bash
uv run python scripts/test_mcp_direct.py    # MCP 直调：朝阳广场→火车东站（1号线13站）；新民路→南宁剧场（朝阳广场换乘）
uv run python scripts/test_retrieval.py     # 检索验收：vector / hybrid / hybrid+rerank 三配置各 5 条全命中
uv run python -m nanning_travel_agent.rag.ingest   # 重建索引
uv run python -m nanning_travel_agent.app   # CLI 问答
uv run streamlit run src/nanning_travel_agent/ui.py   # Web 界面
uv run python eval/judge.py                 # 18 条规则判分，基线 17/18；--ids 只跑指定题
uv run python eval/ragas_eval.py            # RAGAS 两指标，读最新 *_dual.jsonl
uv run python scripts/test_gather.py        # 并行取数子图冒烟（子图未接入主图）
```

> 站名注意：地铁站名是「**火车东站**」；「南宁东站」是高铁站，不在 metro 数据里，传进去会返回「站名不存在」。这是 metro-mcp 故意保留的别名用例，不是接线错误。

改动后先跑对应验证再继续，不要只改不验。改动涉及检索引擎或语料时，`eval/judge.py` 不低于 17/18；e1（跨城站名）是已知失败项，不要为让它通过而放宽期望值。

## 红线

1. API key 不进代码、不进 commit、不进日志
2. `.env`、`data/raw/`、`data/memory/`、`qdrant_storage/` 不提交 git
3. 来源不明的语料不入库；票价/时刻等数值内容以官方来源为准，转述来源（如维基）必须在 frontmatter 标注 `source_type`
4. `D:\agent\rag_and_langgrah\agentic-rag-for-dummies-main` 仅为阅读参考，**不复制其代码**；若借鉴其思路，在提交信息或 `docs/` 里写明出处与差异
5. 语料 schema（切块结构、collection 命名、metadata 字段）、**embedding 模型或维度**变更：先改本文档，再改代码；换 embedding 必须 `force_recreate` 重建 collection，旧阈值作废重测
6. 评测期望值随「数据事实」变化，不随「这次没跑过」变化；改判分口径必须同时更新本文档和 README 的评测说明

## 范围基线

- **已完成（对应 `v0.1`）**：五工具路由（1 本地 RAG + 4 MCP）、混合检索 + 重排、会话持久化 + 滚动摘要、工具预算收口、Streamlit 界面、18 条四类规则判分、RAGAS 两指标、MCP 直调与检索三模式验收
- **未做**：跨会话长期记忆（Store）、RAGAS 口径并入工具上下文、HITL 澄清、MinerU 解析官网公告图片、pytest / CI、与 metro-mcp 合并为 monorepo
- **保留但不在链路里**：`agent/gather.py` 并行取数子图（实测混合题无净收益，接进主图会让规则判分认不出工具，故不接入）；`eval/judge.py --group mcp_only` 的 A/B 分组
