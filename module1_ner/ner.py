"""
Module 1, Bước 1.2: Named Entity Recognition (NER)
- Dùng spaCy để trích xuất thực thể (người, tổ chức, địa điểm, v.v.) từ câu
"""

from typing import List, Dict
import spacy

# Các loại thực thể spaCy mà ta quan tâm cho bài toán fake news / KG linking.
# Tham khảo: https://spacy.io/models/en#en_core_web_sm-labels
RELEVANT_ENTITY_LABELS = {
    "PERSON",    # tên người
    "ORG",       # tổ chức, công ty
    "GPE",       # quốc gia, thành phố, địa phương (Geo-Political Entity)
    "LOC",       # địa điểm không phải GPE (sông, núi...)
    "NORP",      # quốc tịch, nhóm tôn giáo, chính trị
    "EVENT",     # sự kiện (chiến tranh, thế vận hội...)
    "FAC",       # công trình, cơ sở hạ tầng
    "PRODUCT",   # sản phẩm
    "WORK_OF_ART",  # tác phẩm (sách, phim...)
    "LAW",       # văn bản pháp luật
}


class EntityExtractor:
    """
    Bọc quanh spaCy pipeline để tái sử dụng (load model 1 lần, dùng nhiều lần),
    tránh việc load lại model tốn thời gian mỗi câu.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(
                f"Chưa tải model spaCy '{model_name}'. "
                f"Chạy lệnh: python -m spacy download {model_name}"
            )

    def extract_from_sentence(self, sentence: str, sentence_idx: int = 0) -> List[Dict]:
        """
        Trích thực thể từ MỘT câu.

        Trả về danh sách dict, mỗi dict có dạng:
        {
            "text": "Donald Trump",
            "label": "PERSON",
            "start_char": 0,
            "end_char": 13,
            "sentence_idx": 0
        }
        """
        doc = self.nlp(sentence)
        entities = []

        for ent in doc.ents:
            if ent.label_ in RELEVANT_ENTITY_LABELS:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "sentence_idx": sentence_idx,
                })

        return entities

    def extract_from_sentences(self, sentences: List[str]) -> List[Dict]:
        """
        Trích thực thể từ NHIỀU câu (danh sách câu đã qua preprocessing.py).
        Dùng nlp.pipe() để xử lý theo batch, nhanh hơn gọi từng câu riêng lẻ.
        """
        all_entities = []

        # nlp.pipe trả về generator các Doc object theo đúng thứ tự đầu vào
        for sentence_idx, doc in enumerate(self.nlp.pipe(sentences)):
            for ent in doc.ents:
                if ent.label_ in RELEVANT_ENTITY_LABELS:
                    all_entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "sentence_idx": sentence_idx,
                    })

        return all_entities


if __name__ == "__main__":
    from preprocessing import preprocess

    sample = """
    Donald Trump met with leaders from the United States and France yesterday.
    The meeting took place in Washington. Apple Inc. announced record profits.
    """

    sentences = preprocess(sample)
    extractor = EntityExtractor()
    entities = extractor.extract_from_sentences(sentences)

    print(f"Số câu: {len(sentences)}")
    print(f"Số thực thể tìm được: {len(entities)}\n")
    for ent in entities:
        print(f"  [{ent['label']:10s}] {ent['text']}  (câu {ent['sentence_idx']})")