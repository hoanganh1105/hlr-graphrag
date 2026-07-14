"""
Module 1 - Pipeline end-to-end
Nhận: text thô (tin tức / bài báo / claim)
Trả về: danh sách entity đã được gắn QID, sẵn sàng giao cho Module 2 (Retriever)

Luồng:
  text -> preprocess -> NER -> filter -> entity_linking -> output JSON
"""

import json
import os
from typing import List, Dict

from preprocessing import preprocess
from ner import EntityExtractor
from filtering import filter_entities
from entity_linking import load_entity_dict, link_entities


class Module1Pipeline:
    """
    Pipeline Module 1 hoàn chỉnh.
    Load tài nguyên (spaCy model, entity dict) 1 lần khi khởi tạo,
    tái sử dụng cho nhiều lần chạy mà không cần load lại.
    """

    def __init__(self, entity_file_path: str, spacy_model: str = "en_core_web_sm"):
        print("=== Khởi tạo Module 1 Pipeline ===")

        # Load spaCy NER model
        print(f"[1/2] Load spaCy model: {spacy_model}")
        self.extractor = EntityExtractor(spacy_model)

        # Load Wikidata entity dict
        print(f"[2/2] Load entity dict: {entity_file_path}")
        self.label_to_qid, self.all_labels = load_entity_dict(entity_file_path)

        print("=== Pipeline sẵn sàng ===\n")

    def run(self, text: str) -> Dict:
        """
        Chạy toàn bộ pipeline trên một đoạn text.

        Trả về dict chuẩn để giao cho Module 2:
        {
            "input_text": "...",
            "sentences": [...],
            "entities": [
                {
                    "text": "Donald Trump",
                    "label": "PERSON",
                    "sentence_idx": 0,
                    "qid": "Q22686",
                    "matched_label": "Donald Trump",
                    "link_score": 100,
                    "link_method": "exact"
                },
                ...
            ],
            "stats": {
                "num_sentences": 3,
                "num_entities_raw": 7,
                "num_entities_filtered": 5,
                "num_entities_linked": 4,
                "num_entities_not_found": 1
            }
        }
        """
        # Bước 1.1: Tiền xử lý
        sentences = preprocess(text)

        # Bước 1.2: NER
        raw_entities = self.extractor.extract_from_sentences(sentences)

        # Bước 1.3: Lọc nhiễu
        filtered_entities = filter_entities(raw_entities)

        # Bước 1.4: Entity Linking
        linked_entities = link_entities(
            filtered_entities,
            self.label_to_qid,
            self.all_labels,
        )

        # Thống kê
        num_linked = sum(1 for e in linked_entities if e["qid"] is not None)
        num_not_found = sum(1 for e in linked_entities if e["qid"] is None)

        return {
            "input_text": text.strip(),
            "sentences": sentences,
            "entities": linked_entities,
            "stats": {
                "num_sentences": len(sentences),
                "num_entities_raw": len(raw_entities),
                "num_entities_filtered": len(filtered_entities),
                "num_entities_linked": num_linked,
                "num_entities_not_found": num_not_found,
            },
        }

    def run_and_print(self, text: str) -> Dict:
        """Chạy pipeline và in kết quả dễ đọc ra terminal."""
        result = self.run(text)

        print("=" * 60)
        print("INPUT:")
        print(f"  {result['input_text'][:100]}...")
        print()

        print(f"SENTENCES ({result['stats']['num_sentences']}):")
        for i, s in enumerate(result["sentences"]):
            print(f"  [{i}] {s}")
        print()

        print(f"ENTITIES (raw={result['stats']['num_entities_raw']}, "
              f"filtered={result['stats']['num_entities_filtered']}, "
              f"linked={result['stats']['num_entities_linked']}, "
              f"not_found={result['stats']['num_entities_not_found']}):")

        print(f"  {'Text':<25} {'Type':<10} {'QID':<12} {'Score':>5}  Method")
        print("  " + "-" * 68)
        for ent in result["entities"]:
            qid = ent["qid"] or "NOT FOUND"
            print(
                f"  {ent['text']:<25} {ent['label']:<10} "
                f"{qid:<12} {ent['link_score']:>5}  {ent['link_method']}"
            )
        print("=" * 60)

        return result


if __name__ == "__main__":
    # Đường dẫn file entity (chạy từ thư mục gốc fakenews-kg-ner/)
    entity_file = os.path.join("data", "raw", "wikidata5m_entity.txt")

    # Khởi tạo pipeline (load 1 lần)
    pipeline = Module1Pipeline(entity_file_path=entity_file)

    # Test với tin tức mẫu
    sample_news = """
    Donald Trump met with Emmanuel Macron at the White House yesterday.
    The two leaders discussed trade relations between the United States and France.
    Apple Inc. and Google were mentioned as key players in the technology dispute.
    The meeting in Washington was described as productive by both sides.
    """

    result = pipeline.run_and_print(sample_news)

    # Xuất JSON (đây là format chuẩn giao cho Module 2)
    print("\nJSON OUTPUT (giao cho Module 2):")
    print(json.dumps(result["entities"], indent=2, ensure_ascii=False))