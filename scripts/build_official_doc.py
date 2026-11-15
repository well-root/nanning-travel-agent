"""把人工核对过的 OCR 文本转成 data/docs/ 下的正式语料（带 frontmatter + 基础校验）。"""

import argparse
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"

TEMPLATE = """---
title: {title}
source: {source}
source_type: {source_type}
fetched_at: {fetched_at}
---

# {title}

{body}
"""


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)      # 去图片引用
    text = re.sub(r"<[^>]+>", "", text)                   # 去残留 HTML 标签
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="核对后的 markdown")
    parser.add_argument("--slug", required=True, help="输出文件名，如 official_ticket_policy")
    parser.add_argument("--title", required=True)
    parser.add_argument("--source", required=True, help="原始页面 URL 或存档位置")
    parser.add_argument("--source-type", required=True, help="如：官方公告（图片 OCR + 人工核对）")
    parser.add_argument("--fetched-at", default=date.today().isoformat())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"没有这个文件：{src}")
    target = DOCS_DIR / f"{args.slug}.md"
    if target.exists() and not args.force:
        raise SystemExit(f"{target.name} 已存在，确认覆盖加 --force")

    body = clean(src.read_text(encoding="utf-8"))

    # 宁可拒绝，也不要让可疑内容进索引
    if len(body) < 120:
        raise SystemExit(f"正文只有 {len(body)} 字，太短，先补全或换 OCR 档位重解析")
    if "TODO" in body or "待核对" in body:
        raise SystemExit("正文里还有 TODO/待核对，先清理")
    if "<table" in body.lower() or "<td" in body.lower():
        raise SystemExit("正文里还有 HTML 表格，先转成 markdown 结构再入库")
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if lines and (len(lines) - len(set(lines))) / len(lines) > 0.3:
        raise SystemExit("重复行比例过高，疑似 OCR 复读，先人工重写")

    target.write_text(
        TEMPLATE.format(
            title=args.title,
            source=args.source,
            source_type=args.source_type,
            fetched_at=args.fetched_at,
            body=body,
        ),
        encoding="utf-8",
    )
    print(f"写入 {target.relative_to(ROOT)}（正文 {len(body)} 字，{len(lines)} 个非空行）")
    print("记得在 data/SOURCES.md 登记来源，再跑 ingest")


if __name__ == "__main__":
    main()
