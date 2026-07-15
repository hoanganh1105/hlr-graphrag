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
from question_generator import QuestionGenerator
from evidence_retriever import EvidenceRetriever


class Module2Pipeline:
    """
    Pipeline Module 2 hoàn chỉnh.
    Load tài nguyên 1 lần, tái sử dụng cho nhiều claim.
    """

    def __init__(
        self,
        # Đường dẫn file KG (trên Colab sau khi symlink)
        triple_file: str   = "/content/WikiRAG/data/Wikidata5M-RE_transductive_train.txt",
        entity_file: str   = "/content/WikiRAG/data/Wikidata5M-RE_entity.txt",
        relation_file: str = "/content/WikiRAG/data/Wikidata5M-RE_relation.txt",
        # Milvus
        milvus_uri: str         = "/content/milvus_demo.db",
        milvus_collection: str  = "wikipedia",
        top_s: int              = 5,
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
        self.kg = KGRetriever(triple_file, entity_file, relation_file)

        # Bước 2.2: Question Generator
        self.qgen = QuestionGenerator(
            api_key=groq_api_key,
            model=groq_model,
        )

        # Bước 2.3: Evidence Retriever
        self.evidence = EvidenceRetriever(
            milvus_uri=milvus_uri,
            collection_name=milvus_collection,
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

        # --- Bước 2.1: Tìm subgraph trong KG ---
        print("--- Bước 2.1: KG Retrieval ---")
        subgraph = self.kg.retrieve_subgraph(entities)
        coverage = self.kg.compute_kg_coverage(entities, subgraph)
        print(f"KG Coverage: {coverage:.1%}")

        # --- Bước 2.2: Sinh pseudo-question ---
        print("\n--- Bước 2.2: Sinh pseudo-question ---")
        entity_label_map = {e.qid: e.text for e in entities if e.qid}

        if subgraph:
            # Có subgraph: sinh question từ triple
            question_triples = self.qgen.generate_from_subgraph(
                subgraph, entity_label_map
            )
        else:
            # Không có subgraph: sinh question từ cặp entity + claim gốc
            print("[QGen] Không có triple trong KG, dùng entity pairs...")
            from itertools import combinations
            linked = [e for e in entities if e.qid is not None]
            question_triples = []
            for ent_a, ent_b in combinations(linked, 2):
                question = self.qgen.from_entity_pair(ent_a, ent_b, input_text)
                question_triples.append({
                    "triple": None,
                    "question": question,
                    "entity_a": ent_a.text,
                    "entity_b": ent_b.text,
                })
                print(f"  [Q] {question}")

        # --- Bước 2.3: Lấy evidence từ Milvus ---
        print("\n--- Bước 2.3: Evidence Retrieval (Milvus) ---")
        evidence_passages = self.evidence.query_batch(question_triples)

        # --- Tổng hợp output ---
        stats = {
            "num_entities": len(entities),
            "num_linked_entities": sum(1 for e in entities if e.qid),
            "num_triples_found": len(subgraph),
            "kg_coverage": round(coverage, 4),
            "num_questions": len(question_triples),
            "num_passages": len(evidence_passages),
        }

        print("\n=== Module 2 Done ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        return Module2Output(
            input_text=input_text,
            entities=entities,
            subgraph=subgraph,
            evidence_passages=evidence_passages,
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