"""
Module 1, Bước 1.4: Entity Linking
- Load file Wikidata5M-RE_entity.txt thành dict {label -> QID}
- Tầng 1: Exact match (nhanh, ưu tiên trước)
- Tầng 2: Fuzzy match dùng rapidfuzz (khi không khớp tuyệt đối)
"""

import os
from typing import Dict, List, Optional, Tuple
from rapidfuzz import process, fuzz


# Ngưỡng độ tương đồng tối thiểu để chấp nhận fuzzy match (0-100)
# 85 là điểm cân bằng tốt: đủ chặt để tránh false positive,
# đủ lỏng để bắt được các biến thể tên nhỏ ("US" vs "U.S.")
FUZZY_THRESHOLD = 85

# Số lượng candidate trả về từ rapidfuzz trước khi chọn tốt nhất
FUZZY_TOP_K = 5


def load_entity_dict(entity_file_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Đọc file entity của Wikidata5M-RE, trả về:
    - label_to_qid: dict {label_lowercase -> QID}  (dùng cho exact match)
    - all_labels: list tất cả label gốc               (dùng cho fuzzy match)

    Định dạng file: mỗi dòng là "QID\tlabel"
    Ví dụ: Q22686\tDonald Trump
    """
    if not os.path.exists(entity_file_path):
        raise FileNotFoundError(
            f"Không tìm thấy file entity: {entity_file_path}\n"
            f"Hãy chắc chắn file đã được copy vào thư mục data/raw/"
        )

    label_to_qid: Dict[str, str] = {}
    all_labels: List[str] = []

    print(f"Đang load entity dict từ: {entity_file_path}")
    print("(File lớn, có thể mất vài giây...)")

    with open(entity_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            qid = parts[0].strip()
            if not qid:
                continue

            # parts[1] là label chính, parts[2:] là các alias
            # Map TẤT CẢ alias về cùng QID
            aliases = [p.strip() for p in parts[1:] if p.strip()]
            for alias in aliases:
                # Nếu alias đã có QID khác (trùng tên), giữ nguyên cái đầu tiên
                if alias.lower() not in label_to_qid:
                    label_to_qid[alias.lower()] = qid

            # all_labels chỉ lưu label chính (parts[1]) cho fuzzy match
            # tránh list quá lớn với hàng triệu alias lạ
            if aliases:
                all_labels.append(aliases[0])

    print(f"Đã load {len(label_to_qid):,} label/alias -> QID mappings.")
    print(f"Số entity chính: {len(all_labels):,}\n")
    return label_to_qid, all_labels


def exact_match(text: str, label_to_qid: Dict[str, str]) -> Optional[str]:
    """
    Tìm QID bằng so khớp chính xác (không phân biệt hoa thường).
    Trả về QID nếu tìm thấy, None nếu không.
    """
    return label_to_qid.get(text.lower().strip())


def fuzzy_match(
    text: str,
    all_labels: List[str],
    label_to_qid: Dict[str, str],
    threshold: int = FUZZY_THRESHOLD,
    top_k: int = FUZZY_TOP_K,
) -> Optional[Dict]:
    """
    Tìm QID bằng so khớp mờ (fuzzy matching) dùng rapidfuzz.
    Trả về dict gồm QID, label khớp, score — hoặc None nếu không đủ ngưỡng.
    """
    # process.extractOne trả về (match_label, score, index)
    results = process.extract(
        text,
        all_labels,
        scorer=fuzz.WRatio,   # WRatio tốt với tên người/tổ chức có thứ tự từ linh hoạt
        limit=top_k,
    )

    if not results:
        return None

    # Lấy kết quả tốt nhất
    best_label, best_score, _ = results[0]

    if best_score < threshold:
        return None

    qid = label_to_qid.get(best_label.lower())
    if not qid:
        return None

    return {
        "qid": qid,
        "matched_label": best_label,
        "score": best_score,
        "method": "fuzzy",
    }


def link_entity(
    text: str,
    label_to_qid: Dict[str, str],
    all_labels: List[str],
) -> Dict:
    """
    Hàm chính: nhận text thực thể, trả về kết quả linking.

    Thử exact match trước (nhanh O(1)), nếu không có mới fuzzy match.

    Trả về dict:
    {
        "text": "Donald Trump",
        "qid": "Q22686",          # None nếu không tìm thấy
        "matched_label": "Donald Trump",
        "score": 100,
        "method": "exact" / "fuzzy" / "not_found"
    }
    """
    text = text.strip()

    # Tầng 1: Exact match
    qid = exact_match(text, label_to_qid)
    if qid:
        return {
            "text": text,
            "qid": qid,
            "matched_label": text,
            "score": 100,
            "method": "exact",
        }

    # Tầng 2: Fuzzy match
    fuzzy_result = fuzzy_match(text, all_labels, label_to_qid)
    if fuzzy_result:
        return {
            "text": text,
            **fuzzy_result,
        }

    # Không tìm thấy
    return {
        "text": text,
        "qid": None,
        "matched_label": None,
        "score": 0,
        "method": "not_found",
    }


def link_entities(
    entities: List[Dict],
    label_to_qid: Dict[str, str],
    all_labels: List[str],
) -> List[Dict]:
    """
    Link toàn bộ danh sách entity (output của filtering.py).
    Trả về danh sách entity đã được gắn thêm QID.
    """
    results = []

    for ent in entities:
        link_result = link_entity(ent["text"], label_to_qid, all_labels)

        # Merge thông tin NER + linking thành 1 dict
        results.append({
            **ent,                                    # text, label, sentence_idx, ...
            "qid": link_result["qid"],
            "matched_label": link_result["matched_label"],
            "link_score": link_result["score"],
            "link_method": link_result["method"],
        })

    return results


if __name__ == "__main__":
    import sys
    import os

    # Xác định đường dẫn file entity tương đối từ thư mục gốc project
    # (chạy từ thư mục fakenews-kg-ner/)
    entity_file = os.path.join("data", "raw", "wikidata5m_entity.txt")

    # Load dict
    label_to_qid, all_labels = load_entity_dict(entity_file)

    # Test với một số entity mẫu
    test_entities = [
        {"text": "Donald Trump", "label": "PERSON", "sentence_idx": 0},
        {"text": "the United States", "label": "GPE",    "sentence_idx": 0},
        {"text": "France",           "label": "GPE",    "sentence_idx": 0},
        {"text": "Washington",       "label": "GPE",    "sentence_idx": 1},
        {"text": "Apple Inc.",       "label": "ORG",    "sentence_idx": 2},
    ]

    results = link_entities(test_entities, label_to_qid, all_labels)

    print(f"{'Entity':<25} {'QID':<12} {'Matched Label':<25} {'Score':>5}  Method")
    print("-" * 85)
    for r in results:
        qid = r["qid"] or "NOT FOUND"
        matched = r["matched_label"] or "-"
        print(f"{r['text']:<25} {qid:<12} {matched:<25} {r['link_score']:>5}  {r['link_method']}")