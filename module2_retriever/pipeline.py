"""
module2_retriever/pipeline.py
Pipeline end-to-end Module 2

Nhận: JSON output từ Module 1 (file .json hoặc dict)
Trả về: Module2Output (subgraph + evidence passages)
        sẵn sàng giao cho Module 3 (HRM Reasoning)
"""

import os
import sys
import json
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from data_types import (
    Module1Output,
    Module2Output,
    load_module1_output,
)

from kg_retriever import KGRetriever
from evidence_retriever import EvidenceRetriever


class Module2Pipeline:
    """
    Pipeline Module 2 hoàn chỉnh.
    Load tài nguyên 1 lần, tái sử dụng cho nhiều claim.
    """

    def __init__(
    self,
    triple_files: list,                    # list các file triple cần gộp
    entity_file: str   = "/content/data/Wikidata5M-RE_entity.txt",
    relation_file: str = "/content/data/Wikidata5M-RE_relation.txt",
    # Milvus
    milvus_uri: str        = "/content/milvus_demo.db",
    collection_name: str   = "wikipedia",
    top_s: int             = 5,
    # Groq
    groq_api_key: Optional[str] = None,
    groq_model: str             = "openai/gpt-oss-20b",
    # Khác
    device: str = "cpu",
    ):
        print("=" * 55)
        print("Khởi tạo Module 2 Pipeline")
        print("=" * 55)

        # Bước 2.1: KG Retriever
        self.kg = KGRetriever(triple_files, entity_file, relation_file)

        # Bước 2.2: Question Generator
        self.qgen = QuestionGenerator(
            api_key=groq_api_key,
            model=groq_model,
        )

        # Bước 2.3: Evidence Retriever
        self.evidence = EvidenceRetriever(
            milvus_uri=milvus_uri,
            collection_name=collection_name,
            top_s=top_s,
            device=device,
        )

        print("=" * 55)
        print("Module 2 sẵn sàng.\n")

    def run(self, module1_output: Module1Output) -> Module2Output:
        """
        Chạy toàn bộ Module 2 trên output của Module 1.
        """
        entities   = module1_output.entities
        input_text = module1_output.input_text

        print(f"[Module 2] Xử lý: {input_text[:80]}...")
        print(f"[Module 2] Số entity nhận được: {len(entities)}\n")

        # --- Bước 2.1: Tìm subgraph trong KG bằng QID ---
        print("--- Bước 2.1: KG Retrieval ---")
        subgraph = self.kg.retrieve_subgraph(entities)
        coverage = self.kg.compute_kg_coverage(entities, subgraph)
        print(f"KG Coverage: {coverage:.1%}")

        # --- Bước 2.2: Query Milvus bằng label entity ---
        # Không sinh câu hỏi, query thẳng bằng tên entity
        print("\n--- Bước 2.2: Evidence Retrieval (Milvus) ---")
        all_passages = []

        # Query theo từng entity có QID
        linked = [e for e in entities if e.qid is not None]
        for ent in linked:
            query_text = ent.text  # dùng thẳng tên entity làm query
            print(f"  [Milvus] Query entity: {query_text}")
            passages = self.evidence.query(
                question=query_text,
                subject_entity=ent.text,
                object_entity="",
            )
            all_passages.extend(passages)

        # Query thêm bằng triple label nếu có subgraph
        if subgraph:
            for triple in subgraph:
                # Dùng "subject relation object" làm query
                query_text = f"{triple.subject_label} {triple.relation_label} {triple.object_label}"
                print(f"  [Milvus] Query triple: {query_text[:60]}...")
                passages = self.evidence.query(
                    question=query_text,
                    subject_entity=triple.subject_label,
                    object_entity=triple.object_label,
                )
                all_passages.extend(passages)

        # Sắp xếp theo score, bỏ trùng lặp
        seen_texts = set()
        unique_passages = []
        for p in sorted(all_passages, key=lambda x: x.score, reverse=True):
            if p.text not in seen_texts:
                seen_texts.add(p.text)
                unique_passages.append(p)

        # Cập nhật rank
        for i, p in enumerate(unique_passages):
            p.rank = i + 1

        print(f"  [Milvus] Tổng passages unique: {len(unique_passages)}")

        # --- Tổng hợp output ---
        stats = {
            "num_entities": len(entities),
            "num_linked_entities": len(linked),
            "num_triples_found": len(subgraph),
            "kg_coverage": round(coverage, 4),
            "num_passages": len(unique_passages),
        }

        print("\n=== Module 2 Done ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        return Module2Output(
            input_text=input_text,
            entities=entities,
            subgraph=subgraph,
            evidence_passages=unique_passages,
            kg_coverage=coverage,
            stats=stats,
        )

    def run_from_json(self, json_path: str) -> Module2Output:
        """
        Shortcut: load Module1Output từ file JSON rồi chạy pipeline.
        Dùng khi Module 1 chạy local, Module 2 chạy trên Colab.
        """
        print(f"[Module 2] Load Module 1 output từ: {json_path}")
        m1_output = load_module1_output(json_path)
        return self.run(m1_output)

    def run_and_save(
        self,
        json_input_path: str,
        json_output_path: str,
    ) -> Module2Output:
        """
        Load JSON input → chạy pipeline → lưu JSON output.
        """
        result = self.run_from_json(json_input_path)

        with open(json_output_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())

        print(f"\n✅ Đã lưu output tại: {json_output_path}")
        return result


if __name__ == "__main__":
    import os

    # Set API key
    os.environ["GROQ_API_KEY"] = os.environ.get(
        "GROQ_API_KEY", "your_groq_key_here"
    )

    # Khởi tạo pipeline
    pipeline = Module2Pipeline()

    # Chạy từ JSON output của Module 1
    # (file này được tạo bởi: python module1_ner/src/run.py --input news.txt)
    INPUT_JSON  = "/content/module1_output.json"
    OUTPUT_JSON = "/content/module2_output.json"

    if os.path.exists(INPUT_JSON):
        result = pipeline.run_and_save(INPUT_JSON, OUTPUT_JSON)

        # In tóm tắt subgraph
        print("\n--- Subgraph ---")
        for t in result.subgraph:
            print(f"  {t.subject_label} --[{t.relation_label}]--> {t.object_label}")

        # In vài passage đầu
        print("\n--- Top 3 Evidence Passages ---")
        for p in result.evidence_passages[:3]:
            print(f"\n[Score {p.score:.4f}] Q: {p.pseudo_question}")
            print(f"  {p.text[:200]}...")
    else:
        print(f"[ERROR] Không tìm thấy file input: {INPUT_JSON}")
        print("Chạy Module 1 trước: python module1_ner/src/run.py --input <file.txt>")