"""
module2_retriever/evidence_retriever.py
Bước 2.3: Query Milvus lấy Wikipedia evidence passages

Nhận pseudo-question từ question_generator.py,
query vào Milvus (đã build sẵn từ WikiRAG),
trả về top-S đoạn văn bản Wikipedia liên quan nhất.
"""

import os
import sys
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from data_types import EvidencePassage, KGTriple


# Model embedding — phải dùng đúng model đã dùng khi build Milvus index
# (theo paper WikiRAG: paraphrase-multilingual-MiniLM-L12-v2)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Số passage lấy về mặc định (giống WikiRAG TOPS=5)
DEFAULT_TOP_S = 5


class EvidenceRetriever:
    """
    Kết nối Milvus, embed câu hỏi, lấy evidence passages.
    Load model embedding 1 lần, tái sử dụng nhiều lần.
    """

    def __init__(
        self,
        milvus_uri: str = "/content/milvus_demo.db",
        collection_name: str = "wikipedia",
        top_s: int = DEFAULT_TOP_S,
        device: str = "cpu",
    ):
        """
        milvus_uri      : đường dẫn file .db hoặc "http://localhost:19530"
        collection_name : tên collection trong Milvus (mặc định "wikipedia")
        top_s           : số passage lấy về mỗi query
        device          : "cpu" hoặc "cuda" cho embedding model
        """
        self.top_s = top_s
        self.collection_name = collection_name

        print(f"[Evidence] Load embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

        print(f"[Evidence] Kết nối Milvus: {milvus_uri}")
        self.client = MilvusClient(uri=milvus_uri)
        print(f"[Evidence] Sẵn sàng.\n")

    def _embed(self, text: str) -> List[float]:
        """Embed một câu text thành vector."""
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def query(
        self,
        question: str,
        subject_entity: str = "",
        object_entity: str = "",
        triple: Optional[KGTriple] = None,
        rank_offset: int = 0,
    ) -> List[EvidencePassage]:
        """
        Query Milvus với một câu hỏi, trả về top-S EvidencePassage.

        question       : câu hỏi sinh từ QuestionGenerator
        subject_entity : tên entity chủ thể (để ghi vào metadata)
        object_entity  : tên entity đối tượng (để ghi vào metadata)
        triple         : KGTriple tương ứng (optional, để lấy label)
        rank_offset    : offset cho rank (nếu gộp nhiều query)
        """
        if triple:
            subject_entity = triple.subject_label
            object_entity  = triple.object_label

        # Embed câu hỏi
        query_vector = self._embed(question)

        # Query Milvus
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=self.top_s,
                output_fields=["text"],  # lấy về trường text của passage
            )
        except Exception as e:
            print(f"[Evidence] Milvus query error: {e}")
            return []

        # Parse kết quả
        passages = []
        hits = results[0] if results else []

        for rank, hit in enumerate(hits):
            text = hit.get("entity", {}).get("text", "")
            score = hit.get("distance", 0.0)

            if not text:
                continue

            passages.append(
                EvidencePassage(
                    text=text,
                    pseudo_question=question,
                    subject_entity=subject_entity,
                    object_entity=object_entity,
                    score=float(score),
                    rank=rank_offset + rank + 1,
                )
            )

        return passages

    def query_batch(
        self,
        question_triples: List[dict],
    ) -> List[EvidencePassage]:
        """
        Query Milvus cho nhiều câu hỏi cùng lúc.

        question_triples: list dict từ QuestionGenerator.generate_from_subgraph()
          [{"triple": KGTriple, "question": str}, ...]

        Trả về tất cả passage ghép lại, đã sắp xếp theo score giảm dần.
        """
        all_passages = []
        rank_offset = 0

        for item in question_triples:
            question = item["question"]
            triple   = item.get("triple")

            print(f"  [Milvus] Query: {question[:70]}...")
            passages = self.query(
                question=question,
                triple=triple,
                rank_offset=rank_offset,
            )
            all_passages.extend(passages)
            rank_offset += len(passages)

        # Sắp xếp lại theo score giảm dần
        all_passages.sort(key=lambda p: p.score, reverse=True)

        # Cập nhật lại rank
        for i, p in enumerate(all_passages):
            p.rank = i + 1

        print(f"  [Milvus] Tổng passages lấy được: {len(all_passages)}\n")
        return all_passages


if __name__ == "__main__":
    # Test trên Colab — cần Milvus đã được build sẵn
    retriever = EvidenceRetriever(
        milvus_uri="/content/milvus_demo.db",
        collection_name="wikipedia",
        top_s=5,
    )

    # Test query đơn giản
    passages = retriever.query(
        question="Is Donald Trump a citizen of the United States?",
        subject_entity="Donald Trump",
        object_entity="United States",
    )

    print(f"\nKết quả ({len(passages)} passages):")
    for p in passages:
        print(f"\n[Rank {p.rank} | Score {p.score:.4f}]")
        print(f"Q: {p.pseudo_question}")
        print(f"P: {p.text[:200]}...")