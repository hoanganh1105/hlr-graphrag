# fakenews-kg-ner

**Module 1** of the HLR-GraphRAG fake news detection pipeline.

Takes raw news text as input, extracts named entities, and maps them to Wikidata QIDs — ready to hand off to Module 2 (KG Retriever).

```
raw text → preprocess → NER → filter → entity linking → JSON output
```

---

## Requirements

- Python 3.9+
- Wikidata5M-RE entity file (`wikidata5m_entity.txt`) in `data/raw/`

---

## Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/fakenews-kg-ner.git
cd fakenews-kg-ner

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Data Setup

Place the Wikidata5M-RE entity file in the following path:

```
data/raw/wikidata5m_entity.txt
```

This file is available from the [WikiRAG project](https://github.com/colab-nyuad/WikiRAG).  
Expected format per line: `QID \t label \t alias1 \t alias2 \t ...`

---

## Usage

**Run the full pipeline:**

```bash
python src/pipeline.py
```

**Use as a module in your own code:**

```python
from src.pipeline import Module1Pipeline

pipeline = Module1Pipeline(entity_file_path="data/raw/wikidata5m_entity.txt")

result = pipeline.run("Donald Trump met with Emmanuel Macron at the White House.")
print(result["entities"])
```

**Output format (JSON):**

```json
[
  {
    "text": "Donald Trump",
    "label": "PERSON",
    "sentence_idx": 0,
    "qid": "Q22686",
    "matched_label": "Donald Trump",
    "link_score": 100,
    "link_method": "exact"
  }
]
```

`link_method` is one of `exact`, `fuzzy`, or `not_found`.

---

## Project Structure

```
fakenews-kg-ner/
├── data/
│   └── raw/                        # wikidata5m_entity.txt (not tracked by Git)
├── src/
│   ├── preprocessing.py            # Step 1.1 — clean text, split sentences
│   ├── ner.py                      # Step 1.2 — spaCy NER
│   ├── filtering.py                # Step 1.3 — filter noise, deduplicate
│   ├── entity_linking.py           # Step 1.4 — map text to Wikidata QID
│   └── pipeline.py                 # End-to-end Module 1
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## Part of HLR-GraphRAG

This module feeds into the larger **HLR-GraphRAG** pipeline for fake news detection:

```
[Module 1: NER + Linking]  →  [Module 2: KG Retriever]  →  [HRM Reasoning]  →  [LLM Verifier]
         ↑ you are here
```