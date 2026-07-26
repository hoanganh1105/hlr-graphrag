"""
module2_retriever/kg_retriever.py
Bước 2.1: Tra subgraph trong Wikidata5M-RE

Nhận danh sách entity (QID) từ Module 1,
tìm tất cả triple kết nối các entity đó trong KG,
trả về subgraph dạng List[KGTriple].
"""

import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional

# Thêm đường dẫn shared/ vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from data_types import Entity, KGTriple


class KGRetriever:
    """
    Load Wikidata5M-RE vào RAM một lần,
    tái sử dụng để tra subgraph nhanh cho nhiều claim.
    """

    def __init__(
        self,
        triple_files: list,   # đổi từ triple_file (str) sang triple_files (list)
        entity_file: str,
        relation_file: str,
    ):
        print("=== Khởi tạo KG Retriever ===")
        print("[1/3] Load entity labels...")
        self.entity_label = self._load_labels(entity_file)
        print("[2/3] Load relation labels...")
        self.relation_label = self._load_labels(relation_file)
        print("[3/3] Load triple index (gộp tất cả split)...")
        self.index = self._load_triple_index(triple_files)
        print(f"\n✅ KG Retriever sẵn sàng:")
        print(f"   Entities  : {len(self.entity_label):,}")
        print(f"   Relations : {len(self.relation_label):,}")
        print(f"   Subjects  : {len(self.index):,}\n")

    # ----------------------------------------------------------
    # Load helpers
    # ----------------------------------------------------------

    def _load_labels(self, filepath: str) -> Dict[str, str]:
        """
        Đọc file entity hoặc relation, trả về dict {QID -> label_tốt_nhất}.
        
        File có format: QID \t label1 \t alias1 \t alias2 ...
        Ưu tiên label viết hoa chữ cái đầu, không phải dạng code (iso, Q...)
        """
        label_map = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                qid = parts[0].strip()
                if not qid:
                    continue

                # Lấy tất cả label/alias (bỏ qua cột QID đầu tiên)
                candidates = [p.strip() for p in parts[1:] if p.strip()]
                if not candidates:
                    continue

                # Chọn label tốt nhất theo thứ tự ưu tiên:
                # 1. Label có chữ hoa đầu, không phải mã code (iso, Q...), không quá dài
                best = None
                for c in candidates:
                    # Bỏ qua dạng mã code (chữ thường hết, có dấu :, bắt đầu bằng Q+số)
                    if c.startswith('Q') and c[1:].isdigit():
                        continue
                    if ':' in c:
                        continue
                    if c.islower() and len(c.split()) == 1:
                        continue
                    # Ưu tiên cái có chữ hoa đầu
                    if c[0].isupper():
                        best = c
                        break

                # Fallback: dùng label đầu tiên nếu không tìm được cái tốt
                label_map[qid] = best if best else candidates[0]

        return label_map

    def _load_triple_index(self, filepaths: list) -> dict:
        from collections import defaultdict
        index = defaultdict(list)
        total = 0
        for filepath in filepaths:
            if not os.path.exists(filepath):
                print(f"  [SKIP] Không tìm thấy: {filepath}")
                continue
            print(f"  Loading: {os.path.basename(filepath)}")
            count = 0
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        subj, rel, obj = parts
                        index[subj.strip()].append((rel.strip(), obj.strip()))
                        count += 1
            print(f"  → {count:,} triples")
            total += count
        print(f"  Tổng: {total:,} triples")
        return dict(index)

    # ----------------------------------------------------------
    # Core retrieval
    # ----------------------------------------------------------

    def get_neighbors(
        self, qid: str
    ) -> List[Tuple[str, str]]:
        """
        Trả về tất cả (relation_id, object_qid) của một entity.
        """
        return self.index.get(qid, [])

    def find_direct_triple(
        self,
        subject_qid: str,
        object_qid: str,
    ) -> List[KGTriple]:
        """
        Tìm triple TRỰC TIẾP nối subject → object (1-hop).
        Một cặp entity có thể có nhiều relation.
        """
        triples = []
        neighbors = self.get_neighbors(subject_qid)

        for rel_id, obj_qid in neighbors:
            if obj_qid == object_qid:
                triples.append(
                    KGTriple(
                        subject_qid=subject_qid,
                        subject_label=self.entity_label.get(subject_qid, subject_qid),
                        relation_id=rel_id,
                        relation_label=self.relation_label.get(rel_id, rel_id),
                        object_qid=object_qid,
                        object_label=self.entity_label.get(object_qid, object_qid),
                        hop=1,
                    )
                )

        return triples

    def find_2hop_triple(
        self,
        subject_qid: str,
        object_qid: str,
        max_intermediates: int = 3,
    ) -> List[KGTriple]:
        """
        Tìm path 2-hop: subject → intermediate → object.
        Giới hạn max_intermediates để tránh bùng nổ tổ hợp.
        Chỉ gọi khi 1-hop không tìm được kết quả.
        """
        triples = []
        neighbors_subj = self.get_neighbors(subject_qid)

        count = 0
        for rel1_id, inter_qid in neighbors_subj:
            if count >= max_intermediates:
                break

            neighbors_inter = self.get_neighbors(inter_qid)
            for rel2_id, obj_qid in neighbors_inter:
                if obj_qid == object_qid:
                    # Tạo 2 triple đại diện cho path
                    triples.append(
                        KGTriple(
                            subject_qid=subject_qid,
                            subject_label=self.entity_label.get(subject_qid, subject_qid),
                            relation_id=rel1_id,
                            relation_label=self.relation_label.get(rel1_id, rel1_id),
                            object_qid=inter_qid,
                            object_label=self.entity_label.get(inter_qid, inter_qid),
                            hop=2,
                        )
                    )
                    triples.append(
                        KGTriple(
                            subject_qid=inter_qid,
                            subject_label=self.entity_label.get(inter_qid, inter_qid),
                            relation_id=rel2_id,
                            relation_label=self.relation_label.get(rel2_id, rel2_id),
                            object_qid=object_qid,
                            object_label=self.entity_label.get(object_qid, object_qid),
                            hop=2,
                        )
                    )
                    count += 1
                    break  # Chỉ lấy 1 path trung gian mỗi intermediate

        return triples

    def retrieve_subgraph(
        self,
        entities: List[Entity],
        try_2hop: bool = True,
    ) -> List[KGTriple]:
        """
        Hàm chính: nhận danh sách entity từ Module 1,
        tìm tất cả triple kết nối các entity đó.

        Với mỗi cặp entity có QID hợp lệ:
          - Thử 1-hop trước
          - Nếu không có, thử 2-hop (nếu try_2hop=True)

        Trả về list KGTriple không trùng lặp.
        """
        # Lọc chỉ lấy entity đã linked thành công
        linked = [e for e in entities if e.qid is not None]

        if len(linked) < 2:
            print("[KG] Không đủ entity để tìm subgraph (cần ít nhất 2).")
            return []

        subgraph: List[KGTriple] = []
        seen: Set[Tuple] = set()  # tránh triple trùng

        # Tạo tất cả cặp entity (không thứ tự)
        pairs = list(combinations(linked, 2))
        print(f"[KG] Kiểm tra {len(pairs)} cặp entity...")

        for ent_a, ent_b in pairs:
            qid_a, qid_b = ent_a.qid, ent_b.qid

            # Thử A → B
            triples = self.find_direct_triple(qid_a, qid_b)

            # Thử B → A nếu A → B không có
            if not triples:
                triples = self.find_direct_triple(qid_b, qid_a)

            # Thử 2-hop nếu vẫn chưa có
            if not triples and try_2hop:
                triples = self.find_2hop_triple(qid_a, qid_b)
                if not triples:
                    triples = self.find_2hop_triple(qid_b, qid_a)

            # Thêm vào subgraph, tránh trùng
            for t in triples:
                key = (t.subject_qid, t.relation_id, t.object_qid)
                if key not in seen:
                    seen.add(key)
                    subgraph.append(t)

        print(f"[KG] Tìm được {len(subgraph)} triple trong subgraph.\n")
        return subgraph

    def compute_kg_coverage(
        self,
        entities: List[Entity],
        subgraph: List[KGTriple],
    ) -> float:
        """
        Tính % cặp entity có ít nhất 1 triple kết nối trong KG.
        Dùng để đánh giá mức độ KG cover được claim.
        """
        linked = [e for e in entities if e.qid is not None]
        if len(linked) < 2:
            return 0.0

        total_pairs = len(list(combinations(linked, 2)))
        if total_pairs == 0:
            return 0.0

        # Tìm các cặp đã có triple
        connected_pairs: Set[Tuple[str, str]] = set()
        for t in subgraph:
            pair = tuple(sorted([t.subject_qid, t.object_qid]))
            connected_pairs.add(pair)

        return len(connected_pairs) / total_pairs


if __name__ == "__main__":
    import os

    # Đường dẫn file (chạy từ thư mục gốc hlr-graphrag/)
    TRIPLE_FILE   = "data/raw/Wikidata5M-RE_transductive_train.txt"
    ENTITY_FILE   = "data/raw/wikidata5m_entity.txt"
    RELATION_FILE = "data/raw/wikidata5m_relation.txt"

    # Kiểm tra file tồn tại
    for f in [TRIPLE_FILE, ENTITY_FILE, RELATION_FILE]:
        if not os.path.exists(f):
            print(f"[ERROR] Không tìm thấy file: {f}")
            print("Chạy file này trên Colab, đảm bảo đã symlink data/ từ Drive.")
            exit(1)

    # Khởi tạo retriever
    retriever = KGRetriever(TRIPLE_FILE, ENTITY_FILE, RELATION_FILE)

    # Test với entity mẫu từ Module 1
    test_entities = [
        Entity("Donald Trump",    "PERSON", 0, "Q22686",  "Donald Trump",    100, "exact"),
        Entity("Emmanuel Macron", "PERSON", 0, "Q3052772","Emmanuel Macron", 100, "exact"),
        Entity("United States",   "GPE",    1, "Q30",     "United States",   100, "exact"),
        Entity("France",          "GPE",    1, "Q142",    "France",          100, "exact"),
        Entity("Apple Inc.",      "ORG",    2, "Q312",    "Apple Inc.",      100, "exact"),
    ]

    subgraph = retriever.retrieve_subgraph(test_entities)
    coverage = retriever.compute_kg_coverage(test_entities, subgraph)

    print(f"KG Coverage: {coverage:.1%}\n")
    print("Subgraph tìm được:")
    print(f"{'Subject':<25} {'Relation':<35} {'Object':<25} Hop")
    print("-" * 95)
    for t in subgraph:
        print(
            f"{t.subject_label:<25} {t.relation_label:<35} "
            f"{t.object_label:<25} {t.hop}"
        )