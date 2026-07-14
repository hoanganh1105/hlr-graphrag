"""
Module 1, Bước 1.1: Tiền xử lý văn bản
- Làm sạch text (bỏ ký tự thừa, chuẩn hóa khoảng trắng)
- Tách câu (sentence splitting)
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Làm sạch văn bản đầu vào:
    - Bỏ thẻ HTML nếu có
    - Chuẩn hóa khoảng trắng thừa
    - Bỏ ký tự xuống dòng dư thừa
    """
    if not text:
        return ""

    # Bỏ thẻ HTML đơn giản (vd <p>, <br>, <div>...)
    text = re.sub(r"<[^>]+>", " ", text)

    # Chuẩn hóa nhiều khoảng trắng/newline thành 1 khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Tách văn bản thành danh sách câu.

    Đây là bản tách câu đơn giản dựa trên dấu câu (regex).
    Đủ dùng cho bước thử nghiệm ban đầu; nếu cần độ chính xác cao hơn
    (xử lý viết tắt, số thập phân, v.v.), nên thay bằng spaCy sentencizer
    (sẽ tận dụng lại pipeline spaCy đã có ở ner.py).
    """
    if not text:
        return []

    # Tách theo . ! ? theo sau bởi khoảng trắng + chữ hoa
    # (tránh tách nhầm tại các từ viết tắt như "Mr." hay số thập phân)
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = sentence_endings.split(text)

    # Lọc câu rỗng hoặc quá ngắn (nhiễu)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    return sentences


def preprocess(text: str) -> List[str]:
    """
    Hàm tổng hợp: nhận text thô, trả về danh sách câu đã làm sạch.
    """
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    return sentences


if __name__ == "__main__":
    # Test nhanh
    sample = """
    Donald Trump met with leaders from the United States and France yesterday.
    The meeting took place in Washington. <p>It was a historic event.</p>
    """
    result = preprocess(sample)
    for i, s in enumerate(result):
        print(f"[{i}] {s}")