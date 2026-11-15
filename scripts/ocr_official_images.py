"""用 MinerU 远程 API 把官网公告图片 OCR 成 markdown，落到 data/raw/ 供人工核对。"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from mineru import MinerUApiParser

IMG_DIR = ROOT / "data" / "raw" / "official_images"
OUT_DIR = ROOT / "data" / "raw" / "mineru_out"


def main() -> None:
    api_key = os.environ.get("MINERU_API_KEY")
    if not api_key:
        raise SystemExit("缺 MINERU_API_KEY，先在 .env 里加一行")

    images = sorted(IMG_DIR.glob("*.png"))
    parser = MinerUApiParser(api_key=api_key, tier="standard", ocr_mode="auto")
    print("额度:", parser.get_usage())

    results = parser.parse_batch([str(p) for p in images])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for image, result in zip(images, results):
        text = result.markdown()
        (OUT_DIR / f"{image.stem}.md").write_text(text, encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        dup = len(lines) - len(set(lines))
        print(f"{image.name}  字符 {len(text)}  非空行 {len(lines)}  重复行 {dup}")


if __name__ == "__main__":
    main()