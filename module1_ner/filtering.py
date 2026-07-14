"""
Module 1, Bước 1.3: Lọc thực thể nhiễu
- Loại bỏ thực thể quá ngắn
- Loại bỏ thực thể chỉ là số, ký tự đặc biệt
- Loại bỏ thực thể chung chung không có giá trị KG linking
- Loại bỏ thực thể trùng lặp trong cùng câu
"""

import re
from typing import List, Dict

# Danh sách từ chung chung, không nên đưa vào KG linking
# (thêm dần khi chạy thực tế và thấy false positive)
STOPWORDS_ENTITIES = {
    "the", "a", "an", "this", "that", "these", "those",
    "government", "official", "officials", "authorities",
    "people", "person", "man", "woman", "country", "city",
    "company", "organization", "group", "party",
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def is_valid_entity(entity: Dict) -> bool:
    """
    Kiểm tra một thực thể có hợp lệ để đưa vào entity linking hay không.
    Trả về True nếu hợp lệ, False nếu cần lọc bỏ.
    """
    text = entity["text"].strip()

    # 1. Quá ngắn (dưới 2 ký tự)
    if len(text) < 2:
        return False

    # 2. Chỉ toàn số hoặc ký tự đặc biệt (không có chữ cái)
    if not re.search(r"[a-zA-Z]", text):
        return False

    # 3. Nằm trong danh sách từ chung chung
    if text.lower() in STOPWORDS_ENTITIES:
        return False

    # 4. Chỉ có 1 từ VÀ viết thường hoàn toàn (ít khả năng là tên riêng)
    # Ví dụ: "yesterday", "meeting", "record"
    words = text.split()
    if len(words) == 1 and text.islower():
        return False

    if entity["label"] == "NORP" and len(words) == 1:
        return False
    
    return True


def deduplicate(entities: List[Dict]) -> List[Dict]:
    """
    Loại bỏ thực thể trùng lặp theo text (không phân biệt câu).
    Cùng entity dù xuất hiện ở nhiều câu, chỉ cần linking 1 lần.
    Giữ lại lần xuất hiện đầu tiên (sentence_idx nhỏ nhất).
    """
    seen = set()
    result = []

    for ent in entities:
        # Key: chỉ theo text chuẩn hóa, không quan tâm câu nào
        key = ent["text"].lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(ent)

    return result


def filter_entities(entities: List[Dict]) -> List[Dict]:
    """
    Hàm tổng hợp: nhận danh sách thực thể từ NER,
    trả về danh sách đã lọc sạch và bỏ trùng lặp.
    """
    # Bước 1: lọc theo tiêu chí hợp lệ
    valid = [ent for ent in entities if is_valid_entity(ent)]

    # Bước 2: loại bỏ trùng lặp
    unique = deduplicate(valid)

    return unique


if __name__ == "__main__":
    from preprocessing import preprocess
    from ner import EntityExtractor

    sample = """
    Donald Trump met with leaders from the United States and France yesterday.
    The meeting took place in Washington. Apple Inc. announced record profits.
    Donald Trump also met with the French government officials.
    """

    sentences = preprocess(sample)
    extractor = EntityExtractor()
    raw_entities = extractor.extract_from_sentences(sentences)
    filtered = filter_entities(raw_entities)

    print(f"Thực thể trước lọc: {len(raw_entities)}")
    print(f"Thực thể sau lọc:   {len(filtered)}\n")
    for ent in filtered:
        print(f"  [{ent['label']:10s}] {ent['text']}  (câu {ent['sentence_idx']})")