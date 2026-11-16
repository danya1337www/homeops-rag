from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

from src.ingest import Document, load_corpus


SECTION_PATTERN = re.compile(
    r"^##\s+(.+?)\s*$",
    flags=re.MULTILINE,
)

ERROR_START_PATTERN = re.compile(
    r"(?=Код\s+\*\*[A-Z]+\d+\*\*)"
)

ERROR_CODE_PATTERN = re.compile(
    r"Код\s+\*\*([A-Z]+\d+)\*\*"
)


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    embedding_text: str
    metadata: dict[str, Any]


def extract_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_PATTERN.finditer(text))
    sections = []

    for index, match in enumerate(matches):
        section_title = match.group(1)

        content_start = match.end()

        if index + 1 < len(matches):
            content_end = matches[index + 1].start()
        else:
            content_end = len(text)

        section_text = text[content_start:content_end].strip()

        if section_text:
            sections.append((section_title, section_text))

    return sections


def split_section(section_title: str, text: str) -> list[str]:
    if section_title.casefold() != "коды ошибок":
        return [text]

    parts = ERROR_START_PATTERN.split(text)

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def build_embedding_text(
    document: Document,
    section_title: str,
    chunk_text: str,
) -> str:
    title = document.metadata["title"]
    model = document.metadata.get("model")
    version = document.metadata.get("version")

    context_lines = [
        f"Документ: {title}",
        f"Раздел: {section_title}",
    ]

    if model:
        context_lines.append(f"Модель: {model}")

    if version:
        context_lines.append(f"Версия: {version}")

    context_lines.append("")
    context_lines.append(chunk_text)

    return "\n".join(context_lines)


def create_chunk(
    document: Document,
    section_title: str,
    chunk_text: str,
    chunk_index: int,
) -> Chunk:
    content_hash = sha256(
        chunk_text.encode("utf-8")
    ).hexdigest()[:12]

    chunk_metadata = {
        **document.metadata,
        "section": section_title,
        "chunk_index": chunk_index,
    }

    error_match = ERROR_CODE_PATTERN.search(chunk_text)

    if error_match:
        chunk_metadata["error_code"] = error_match.group(1)

    return Chunk(
        id=f"{document.id}:{content_hash}",
        document_id=document.id,
        text=chunk_text,
        embedding_text=build_embedding_text(
            document=document,
            section_title=section_title,
            chunk_text=chunk_text,
        ),
        metadata=chunk_metadata,
    )


def chunk_document(document: Document) -> list[Chunk]:
    chunks = []
    chunk_index = 0

    for section_title, section_text in extract_sections(document.text):
        parts = split_section(section_title, section_text)

        for part in parts:
            chunk = create_chunk(
                document=document,
                section_title=section_title,
                chunk_text=part,
                chunk_index=chunk_index,
            )

            chunks.append(chunk)
            chunk_index += 1

    return chunks


def chunk_corpus(documents: list[Document]) -> list[Chunk]:
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks


if __name__ == "__main__":
    documents = load_corpus(Path("data/raw"))
    chunks = chunk_corpus(documents)

    print(f"Документов: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks:
        error_code = chunk.metadata.get("error_code", "-")

        print(
            f"{chunk.id} | "
            f"section={chunk.metadata['section']} | "
            f"model={chunk.metadata.get('model', '-')} | "
            f"error={error_code} | "
            f"chars={len(chunk.text)}"
        )

    print("\nChunks с E15:")

    for chunk in chunks:
        if chunk.metadata.get("error_code") == "E15":
            print(
                chunk.metadata.get("model"),
                chunk.text[:100],
            )
                
