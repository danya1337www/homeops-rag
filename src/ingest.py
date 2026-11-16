from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml 

@dataclass
class Document: 
    id: str
    text: str
    metadata: dict[str,Any]
    source: str

def load_markdown(path: Path) -> Document:
    raw_text = path.read_text(encoding="utf-8")

    parts = raw_text.split("---", 2)

    if len(parts) != 3 or parts[0].strip(): 
        raise ValueError(f"В файле{path} отсутствует YAML front matter")

    metadata_text = parts[1]
    document_text = parts[2].strip()

    metadata = yaml.safe_load(metadata_text)

    return Document(
        id=metadata["document_id"],
        text=document_text,
        metadata=metadata, 
        source=str(path),
    )

def load_corpus(directory: Path) -> list[Document]: 
    documents = []

    for path in sorted(directory.glob("*.md")): 
        document = load_markdown(path)
        documents.append(document)

    return documents


if __name__ == "__main__":
    corpus_path = Path("data/raw")
    documents = load_corpus(corpus_path)

    print(f"Загружено документов: {len(documents)}")

    for document in documents:
        print(
            f"- {document.id}: "
            f"{len(document.text)} символов, "
            f"модель={document.metadata.get('model', 'не указана')}"
        )

