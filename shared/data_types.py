"""
shared/data_types.py
Định nghĩa schema dữ liệu dùng chung giữa các module trong pipeline HLR-GraphRAG.

Flow:
  Module 1 (NER)  →  Module1Output
  Module 2 (Retriever)  →  Module2Output
  Module 3 (HRM)  →  Module3Output
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# ============================================================
# Các kiểu dữ liệu cơ bản (dùng chung)
# ============================================================

@dataclass
class Entity:
    """Một thực thể đã được NER + Entity Linking."""
    text: str               # Tên entity trong câu ("Donald Trump")
    label: str              # Loại entity theo spaCy ("PERSON", "GPE", "ORG"...)
    sentence_idx: int       # Câu thứ mấy (0-indexed)
    qid: Optional[str]      # QID trong Wikidata ("Q22686"), None nếu không tìm thấy
    matched_label: Optional[str]   # Label khớp trong Wikidata ("Donald Trump")
    link_score: float       # Độ tin cậy của việc linking (0-100)
    link_method: str        # "exact", "fuzzy", hoặc "not_found"


@dataclass
class KGTriple:
    """Một triple tìm được trong Wikidata5M-RE."""
    subject_qid: str        # QID của subject  ("Q22686")
    subject_label: str      # Label của subject ("Donald Trump")
    relation_id: str        # ID của relation   ("P27")
    relation_label: str     # Label của relation ("country of citizenship")
    object_qid: str         # QID của object    ("Q30")
    object_label: str       # Label của object  ("United States")
    hop: int = 1            # Số hop từ entity gốc (1 = trực tiếp, 2 = qua entity trung gian)


@dataclass
class EvidencePassage:
    """Một đoạn văn bản Wikipedia lấy từ Milvus."""
    text: str               # Nội dung đoạn văn
    pseudo_question: str    # Câu hỏi dùng để query ra đoạn này
    subject_entity: str     # Entity chủ thể của câu hỏi
    object_entity: str      # Entity đối tượng của câu hỏi
    score: float            # Độ tương đồng vector (0-1)
    rank: int               # Thứ hạng trong top-S kết quả


# ============================================================
# Output của từng Module
# ============================================================

@dataclass
class Module1Output:
    """
    Output của Module 1 (NER + Entity Linking).
    Giao cho Module 2 (Retriever).
    """
    input_text: str                     # Text gốc của claim/bài báo
    sentences: List[str]                # Danh sách câu sau preprocessing
    entities: List[Entity]              # Danh sách entity đã linked
    stats: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_pipeline_output(cls, raw: dict) -> "Module1Output":
        """
        Convert output dict từ pipeline.py của Module 1
        thành Module1Output object.
        """
        entities = [Entity(**e) for e in raw["entities"]]
        return cls(
            input_text=raw["input_text"],
            sentences=raw["sentences"],
            entities=entities,
            stats=raw.get("stats", {}),
        )


@dataclass
class Module2Output:
    """
    Output của Module 2 (KG Retriever + Evidence Retriever).
    Giao cho Module 3 (HRM Reasoning).
    """
    input_text: str                         # Text gốc của claim
    entities: List[Entity]                  # Entity từ Module 1 (pass-through)
    subgraph: List[KGTriple]                # Triple tìm được trong Wikidata5M-RE
    evidence_passages: List[EvidencePassage]  # Đoạn Wikipedia từ Milvus
    kg_coverage: float = 0.0               # % entity pairs có cạnh trong KG (0-1)
    stats: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


@dataclass
class Module3Output:
    """
    Output của Module 3 (HRM Reasoning).
    Kết quả cuối cùng của pipeline.
    """
    input_text: str             # Text gốc của claim
    verdict: str                # "REAL" hoặc "FAKE"
    confidence: float           # Điểm tin cậy (0.0 - 1.0)
    explanation: str            # Giải thích lý do (từ evidence + subgraph)
    supporting_triples: List[KGTriple]      # Triple hỗ trợ kết luận
    supporting_passages: List[EvidencePassage]  # Passage hỗ trợ kết luận
    stats: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ============================================================
# Helper: load Module1Output từ file JSON
# (dùng khi Module 1 chạy local, Module 2 chạy trên Colab)
# ============================================================

def load_module1_output(json_path: str) -> Module1Output:
    """
    Load kết quả Module 1 từ file JSON (output của src/run.py).
    Dùng ở đầu Module 2 để nhận input từ Module 1.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # File output của run.py có thêm "metadata" wrapper
    # Cần extract đúng phần "entities" và "sentences"
    entities = [Entity(**e) for e in raw["entities"]]
    return Module1Output(
        input_text=raw.get("input_text", ""),
        sentences=raw.get("sentences", []),
        entities=entities,
        stats=raw.get("stats", {}),
    )


if __name__ == "__main__":
    # Test: tạo thử một Module1Output mẫu và in ra JSON
    sample = Module1Output(
        input_text="Donald Trump met with Emmanuel Macron at the White House.",
        sentences=["Donald Trump met with Emmanuel Macron at the White House."],
        entities=[
            Entity(
                text="Donald Trump",
                label="PERSON",
                sentence_idx=0,
                qid="Q22686",
                matched_label="Donald Trump",
                link_score=100,
                link_method="exact",
            ),
            Entity(
                text="Emmanuel Macron",
                label="PERSON",
                sentence_idx=0,
                qid="Q3052772",
                matched_label="Emmanuel Macron",
                link_score=100,
                link_method="exact",
            ),
            Entity(
                text="the White House",
                label="FAC",
                sentence_idx=0,
                qid="Q35525",
                matched_label="the White House",
                link_score=100,
                link_method="exact",
            ),
        ],
        stats={"num_sentences": 1, "num_entities_linked": 3},
    )

    print(sample.to_json())