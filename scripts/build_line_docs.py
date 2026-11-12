"""从高德静态线路数据生成 5 条线路的 markdown 文档。

数据源: nanning-metro-metro_mcp/data/nanning_metro.json (与 MCP 工具同源，保证一致性)
输出:   data/docs/line_{1..5}.md

用法: python scripts/build_line_docs.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(r"D:\agent\nanning-metro-mcp\data\nanning_metro.json")
OUT = ROOT / "data" / "docs"
FETCHED_AT = "2026-11-06"

def main():
    metro = json.loads(DATA.read_text(encoding="utf-8"))
    lines = {ln["ln"]: [s["n"] for s in ln["st"]] for ln in metro["l"]}

    # 换乘站: 站名出现在 >=2 条线路
    count = Counter(name for sts in lines.values() for name in set(sts))
    transfers = {name for name, c in count.items() if c >= 2}

    OUT.mkdir(parents=True, exist_ok=True)
    for name, stations in sorted(lines.items()):
        num = name.replace("号线", "")
        transfer_in_line = [s for s in stations if s in transfers]
        body = "\n".join(f"{i}. {s}" + ("（换乘站）" if s in transfers else "")
                         for i, s in enumerate(stations, 1))
        doc = f"""---
title: 南宁轨道交通{name}介绍
source: nanning-metro-metro_mcp/data/nanning_metro.json（高德地图静态线路数据）
source_type: 静态线路数据（非官方文档，与自研 MCP 工具同源）
fetched_at: {FETCHED_AT}
---

# 南宁轨道交通{name}

- 起终点：{stations[0]} ↔ {stations[-1]}
- 车站数量：{len(stations)} 座
- 换乘站：{"、".join(transfer_in_line) if transfer_in_line else "无"}
- 注：线路走向为数据文件中的站序，不含实时运营信息（首末班车、行车间隔请查询官方渠道）。

## 站点顺序

{body}
"""
        out = OUT / f"line_{num}.md"
        out.write_text(doc, encoding="utf-8")
        print(f"wrote {out.name}: {len(stations)} stations, {len(transfer_in_line)} transfers")

    print("all transfers:", "、".join(sorted(transfers)))

if __name__ == "__main__":
    main()
