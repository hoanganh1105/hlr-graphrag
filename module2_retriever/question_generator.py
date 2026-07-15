"""
module2_retriever/question_generator.py
Bước 2.2: Sinh pseudo-question từ KGTriple hoặc cặp entity

Dùng LLM (Groq) để sinh câu hỏi xác minh ngắn gọn,
tương tự Question Prompt trong WikiRAG (Table 9 paper).
"""

import os
import sys
from typing import List, Optional
from groq import Groq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from data_types import Entity, KGTriple


# Prompt sinh câu hỏi — giữ sát với WikiRAG Question Prompt
QUESTION_PROMPT_TEMPLATE = """Given the subject '{subject}', the relation '{relation}', and the object '{object}', generate a short yes/no question to verify whether this triple is true.

Subject description: {subject_desc}
Object description: {object_desc}

Instructions:
- The question must be specific, clear, and short (one sentence).
- The question must mention both the subject and object.
- Do NOT include extra explanation, just the question.

Question:"""


ENTITY_PAIR_PROMPT_TEMPLATE = """Given two entities '{entity_a}' and '{entity_b}', generate a short yes/no question to verify whether they are related in the context of: "{claim}"

Instructions:
- The question must be specific and short (one sentence).
- The question must mention both entities.
- Do NOT include extra explanation, just the question.

Question:"""


class QuestionGenerator:
    """
    Sinh pseudo-question từ KGTriple hoặc cặp entity
    dùng để query Milvus ở bước tiếp theo.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b"):
        self.model = model
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Cần GROQ_API_KEY. Set bằng: os.environ['GROQ_API_KEY'] = '...'"
            )
        self.client = Groq(api_key=api_key)

    def _call_llm(self, prompt: str) -> str:
        """Gọi Groq API, trả về text response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.3,  # thấp để câu hỏi ổn định, ít biến thể
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[QuestionGen] LLM error: {e}")
            return ""

    def from_triple(
        self,
        triple: KGTriple,
        subject_desc: str = "",
        object_desc: str = "",
    ) -> str:
        """
        Sinh câu hỏi từ một KGTriple.
        Ví dụ: (Donald Trump, P27, United States)
        → "Is Donald Trump a citizen of the United States?"
        """
        prompt = QUESTION_PROMPT_TEMPLATE.format(
            subject=triple.subject_label,
            relation=triple.relation_label,
            object=triple.object_label,
            subject_desc=subject_desc or triple.subject_label,
            object_desc=object_desc or triple.object_label,
        )
        question = self._call_llm(prompt)

        # Fallback nếu LLM thất bại
        if not question:
            question = (
                f"Is {triple.subject_label} related to "
                f"{triple.object_label} via {triple.relation_label}?"
            )

        return question

    def from_entity_pair(
        self,
        entity_a: Entity,
        entity_b: Entity,
        claim: str = "",
    ) -> str:
        """
        Sinh câu hỏi từ cặp entity khi không tìm được triple trong KG.
        Dùng claim gốc làm context.
        """
        prompt = ENTITY_PAIR_PROMPT_TEMPLATE.format(
            entity_a=entity_a.text,
            entity_b=entity_b.text,
            claim=claim or f"{entity_a.text} and {entity_b.text}",
        )
        question = self._call_llm(prompt)

        # Fallback
        if not question:
            question = f"Are {entity_a.text} and {entity_b.text} related?"

        return question

    def generate_from_subgraph(
        self,
        subgraph: List[KGTriple],
        entity_label_map: dict,
    ) -> List[dict]:
        """
        Sinh câu hỏi cho toàn bộ subgraph.
        Trả về list dict: {triple, question}
        """
        results = []
        for triple in subgraph:
            question = self.from_triple(triple)
            results.append({
                "triple": triple,
                "question": question,
            })
            print(f"  [Q] {question}")
        return results


if __name__ == "__main__":
    import os
    os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "your_key_here")

    gen = QuestionGenerator()

    # Test với triple mẫu
    test_triple = KGTriple(
        subject_qid="Q22686",
        subject_label="Donald Trump",
        relation_id="P27",
        relation_label="country of citizenship",
        object_qid="Q30",
        object_label="United States of America",
        hop=1,
    )

    question = gen.from_triple(test_triple)
    print(f"Triple  : {test_triple.subject_label} --[{test_triple.relation_label}]--> {test_triple.object_label}")
    print(f"Question: {question}")