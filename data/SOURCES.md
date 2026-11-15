# 语料来源清单（诚信披露）

抓取日期：2026-11-06（官网公告图片 2026-11-13 补做 OCR）。本文件记录 `data/docs/` 每篇语料的来源、抓取方式与已知局限。

## 文档来源

| 语料文件 | 来源 | 类型 | 方式 |
|---|---|---|---|
| fare_pricing.md | 维基百科「南宁轨道交通」票务章节（引用 2016 年南宁市物价局文件） | 二手 | Tavily 远程抓取文本 |
| tickets_discounts.md | 同上 + 江南区政府门户 t6274323（一码通城小程序） | 混合 | 同上 |
| ticket_usage.md | 同上 + 江南区政府门户 t6288053（末班车规则） | 混合 | 同上 |
| trip_pass.md | 江南区政府门户 t6354363（宁仔计次票，2025-05-07） | 官方转载 | urllib 抓取正文文本 |
| operating_hours.md | 江南区政府门户 t6288053（2025 运营时间通告，2025-02-27） | 官方 | urllib 抓取正文文本 |
| official_ticket_policy.md | nngdjt.com「票务指引」栏目公告图（原文存档 `raw/nngdjt_pwzy.html`） | 官方 | MinerU 远程 API OCR + 人工逐句核对（HTML 表格已手工转 markdown 结构） |
| official_gate_entry.md | nngdjt.com「进出闸须知」公告图（原文存档 `raw/nngdjt_article.html`） | 官方 | 同上 |
| official_gate_exit.md | 同上 | 官方 | 同上 |
| line_1.md ~ line_5.md | `nanning-metro-mcp/data/nanning_metro.json`（高德静态线路数据） | 数据文件 | `scripts/build_line_docs.py` 生成，可复现；换乘站由站名跨线匹配自动计算 |

## 已知局限

1. **官网公告图片已入库，但存在内容重叠**：nngdjt.com「乘车须知/票务指引」正文为 PNG 图片（`raw/official_images/`），2026-11-13 用 MinerU 远程 API 解析、人工逐句核对后入库 3 篇（`official_ticket_policy` / `official_gate_entry` / `official_gate_exit`）。其中票价与优惠条款与 `fare_pricing.md`、`tickets_discounts.md`（维基二手）**内容重叠**——检索 top-1 会在这两份等价文档间波动，故 `scripts/test_retrieval.py` 对票价/优惠类 query 接受两份中的任一份。数值核对结果一致，官方公告为一手来源，冲突时以官方公告为准。
2. **此前 OCR 失败经历（保留记录）**：2026-11-06 用 DeepSeek-OCR（luma MCP）尝试 4 次，章节标题可读但正文小字出现幻觉复读，未入语料；后改用 MinerU 远程 API 才得到可用文本。这条记录说明图片公告的 OCR 结果必须人工核对，不能直接信任。
3. **无逐站首末班车时刻**：官网「首末班车时刻表」为 Vue SPA + 未公开接口，未能抓取；运营时间文档只含全网级时间（6:30–23:00，周五六 24:00）。
4. **维基为二手来源**：票价政策文本（2016）与优惠细则引用自维基，其自身引用官网车票须知与新闻报道；现已用官网公告（`official_ticket_policy.md`）交叉核对，数值一致。
5. **时效性**：票价政策制定于 2016-01-28；运营时间通告有效期至 2025-12-31；优惠与计次票细则可能已调整。所有文档 frontmatter 均标注 `fetched_at`。

## 原始材料存档

- `raw/nngdjt_home.html` — 官网首页（导航来源）
- `raw/nngdjt_article.html` / `raw/nngdjt_pwzy.html` — 官网「进闸须知」「票务须知」文章页（正文为图片）
- `raw/nngdjt_subway.html` — 官网地铁服务 SPA 页面（692KB，Vue 渲染）
- `raw/official_images/` — 官网公告图片 3 张（已于 2026-11-13 完成 OCR 入库）
- `raw/mineru_out/` — MinerU OCR 原始输出（`pwzy_0_clean.md` 为人工转换后的 markdown 结构），不入库、不提交
- `raw/nnjn/*.html` — 政府门户 3 篇文章页
