"""
run.py - Command line interface cho Module 1 Pipeline
Nhận file .txt, xuất file .json

Cách dùng:
  python src/run.py --input data/input.txt
  python src/run.py --input data/input.txt --output data/output.json
  python src/run.py --input data/input.txt --entity data/raw/wikidata5m_entity.txt
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Đảm bảo import được các module trong cùng thư mục src/
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import Module1Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Module 1 — NER + Entity Linking pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Đường dẫn file .txt chứa tin tức / bài báo / claim cần xử lý",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Đường dẫn file .json để lưu kết quả.\n"
            "Nếu không truyền, tự động tạo tên file dựa trên tên file input.\n"
            "Ví dụ: news.txt → news_output.json"
        ),
    )
    parser.add_argument(
        "--entity", "-e",
        default=os.path.join("data", "raw", "wikidata5m_entity.txt"),
        help=(
            "Đường dẫn file entity Wikidata5M-RE.\n"
            "Mặc định: data/raw/wikidata5m_entity.txt"
        ),
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="Tên spaCy model. Mặc định: en_core_web_sm",
    )
    return parser.parse_args()


def resolve_output_path(input_path: str, output_arg: str) -> str:
    """
    Nếu --output không được truyền, tự tạo tên output dựa trên input.
    Ví dụ: data/news.txt → data/news_output.json
    """
    if output_arg:
        return output_arg

    base = os.path.splitext(input_path)[0]  # bỏ đuôi .txt
    return base + "_output.json"


def main():
    args = parse_args()

    # Kiểm tra file input tồn tại
    if not os.path.exists(args.input):
        print(f"[ERROR] Không tìm thấy file input: {args.input}")
        sys.exit(1)

    # Kiểm tra file entity tồn tại
    if not os.path.exists(args.entity):
        print(f"[ERROR] Không tìm thấy file entity: {args.entity}")
        print("Hãy đảm bảo file wikidata5m_entity.txt nằm trong data/raw/")
        sys.exit(1)

    # Xác định đường dẫn output
    output_path = resolve_output_path(args.input, args.output)

    # Tạo thư mục output nếu chưa có
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Đọc file input
    print(f"[INFO] Đọc file input: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print("[ERROR] File input rỗng.")
        sys.exit(1)

    print(f"[INFO] Độ dài text: {len(text)} ký tự\n")

    # Khởi tạo và chạy pipeline
    pipeline = Module1Pipeline(
        entity_file_path=args.entity,
        spacy_model=args.spacy_model,
    )

    print("[INFO] Đang xử lý...\n")
    result = pipeline.run(text)

    # Thêm metadata vào output
    output = {
        "metadata": {
            "input_file": os.path.abspath(args.input),
            "entity_file": os.path.abspath(args.entity),
            "spacy_model": args.spacy_model,
            "processed_at": datetime.now().isoformat(),
        },
        "stats": result["stats"],
        "sentences": result["sentences"],
        "entities": result["entities"],
    }

    # Ghi file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # In tóm tắt kết quả
    print("=" * 55)
    print("DONE")
    print(f"  Sentences  : {result['stats']['num_sentences']}")
    print(f"  Entities   : {result['stats']['num_entities_raw']} raw "
          f"→ {result['stats']['num_entities_filtered']} filtered "
          f"→ {result['stats']['num_entities_linked']} linked")
    print(f"  Not found  : {result['stats']['num_entities_not_found']}")
    print(f"  Output     : {output_path}")
    print("=" * 55)

    # In bảng entity ra terminal
    if result["entities"]:
        print(f"\n{'Text':<25} {'Type':<10} {'QID':<12} {'Score':>5}  Method")
        print("-" * 65)
        for ent in result["entities"]:
            qid = ent["qid"] or "NOT FOUND"
            print(
                f"{ent['text']:<25} {ent['label']:<10} "
                f"{qid:<12} {ent['link_score']:>5}  {ent['link_method']}"
            )


if __name__ == "__main__":
    main()