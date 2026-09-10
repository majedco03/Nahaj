"""Course-material ingestion and persistent Chroma storage."""

from __future__ import annotations

import hashlib
from math import sqrt
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree
from zipfile import ZipFile

from .app_config import settings

try:  # Chroma is installed with the API requirements in deployments.
    import chromadb
except ImportError:  # Keep the rest of the API importable before dependencies are installed.
    chromadb = None  # type: ignore[assignment]


class LocalHashEmbeddingFunction:
    """Small deterministic embedding function with no model download or network use."""

    dimensions = 256

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in input:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"\w+", str(text).casefold()):
                digest = int.from_bytes(
                    hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
                    "big",
                )
                index = (digest >> 1) % self.dimensions
                vector[index] += -1.0 if digest & 1 else 1.0
            magnitude = sqrt(sum(value * value for value in vector)) or 1.0
            embeddings.append([value / magnitude for value in vector])
        return embeddings

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    @staticmethod
    def name() -> str:
        return "nahaj-local-hash"

    @staticmethod
    def build_from_config(_config: dict[str, Any]) -> "LocalHashEmbeddingFunction":
        return LocalHashEmbeddingFunction()

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def validate_config(_config: dict[str, Any]) -> None:
        return None

    def max_tokens(self) -> int:
        return 8192

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]


class CourseIngestor:
    """Chunk one uploaded course document and upsert it into that course's collection."""

    _HEADING_RE = re.compile(r"^(?:#{1,6}\s+|\d+(?:\.\d+)*[.)]?\s+)")
    _EXAM_COLLECTIONS = {"exam", "exams", "old_exam", "old_exams", "past_exam", "past_exams"}

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        client: Any | None = None,
        *,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        embedding_function: Any | None = None,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
        self.persist_dir = Path(
            persist_dir
            or os.getenv("CHROMA_DIR", "")
            or (settings.upload_dir / ".chroma")
        ).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or self._create_client()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_function = embedding_function or LocalHashEmbeddingFunction()

    def _create_client(self) -> Any:
        if chromadb is None:
            raise RuntimeError("chromadb is required for RAG ingestion")
        return chromadb.PersistentClient(path=str(self.persist_dir))

    @staticmethod
    def collection_name(course_id: Any) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "-", str(course_id)).strip("-_") or "course"
        return f"course-{safe_id}"

    def collection(self, course_id: Any) -> Any:
        options = {"name": self.collection_name(course_id), "metadata": {"hnsw:space": "cosine"}}
        if self.embedding_function is not None:
            options["embedding_function"] = self.embedding_function
        try:
            return self.client.get_or_create_collection(**options)
        except ValueError as exc:
            if "embedding function conflict" not in str(exc).casefold():
                raise
            existing = self.client.get_collection(name=options["name"])
            if existing.count():
                raise
            self.client.delete_collection(name=options["name"])
            return self.client.create_collection(**options)

    def ingest_document(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Index one document. Calling this again for the same document is idempotent."""

        if self._is_exam(document):
            return {"collection": None, "chunks": 0, "skipped": True}
        course_id = document.get("course_id")
        if not course_id:
            raise ValueError("course_id is required for ingestion")
        document_id = str(document.get("id") or document.get("document_id") or document.get("filename") or "document")
        text = self._read_document(document)
        sections = self._normalise_sections(document.get("sections")) if document.get("sections") else self._sectionise(text)
        chunks = [
            (section_title, piece)
            for section_title, section_text in sections
            for piece in self._chunk_section(section_title, section_text)
        ]
        collection = self.collection(course_id)
        # Remove old chunks first so a retry also handles a changed chunk count.
        collection.delete(where={"document_id": document_id})
        if not chunks:
            return {"collection": self.collection_name(course_id), "chunks": 0, "skipped": False}

        filename = str(document.get("filename") or "Course material")
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "course_id": str(course_id),
                "filename": filename,
                "section": section_title,
                "collection": str(document.get("collection") or "course_material"),
                "chunk_index": index,
            }
            for index, (section_title, _) in enumerate(chunks)
        ]
        collection.upsert(
            ids=ids,
            documents=[piece for _, piece in chunks],
            metadatas=metadatas,
        )
        return {"collection": self.collection_name(course_id), "chunks": len(chunks), "skipped": False}

    def remove_document(self, course_id: Any, document_id: Any) -> None:
        """Remove all indexed chunks belonging to one document."""

        try:
            options = {"name": self.collection_name(course_id)}
            if self.embedding_function is not None:
                options["embedding_function"] = self.embedding_function
            self.client.get_collection(**options).delete(where={"document_id": str(document_id)})
        except Exception:
            # A missing collection/document is already in the desired state.
            return

    def _is_exam(self, document: Mapping[str, Any]) -> bool:
        collection = str(document.get("collection", "")).strip().lower().replace("-", "_").replace(" ", "_")
        return bool(document.get("is_exam")) or collection in self._EXAM_COLLECTIONS

    def _read_document(self, document: Mapping[str, Any]) -> str:
        for key in ("text", "content", "body", "raw_text"):
            value = document.get(key)
            if value:
                return "\n".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)

        path_value = document.get("path") or document.get("stored_path") or document.get("file_path")
        if not path_value:
            return ""
        path = Path(str(path_value))
        if not path.is_file():
            return ""
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown", ".html", ".htm"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader

                return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            except Exception:
                return ""
        if suffix in {".docx", ".pptx"}:
            return self._read_office_xml(path, suffix)
        return ""

    @staticmethod
    def _read_office_xml(path: Path, suffix: str) -> str:
        try:
            with ZipFile(path) as archive:
                names = archive.namelist()
                if suffix == ".docx":
                    names = [name for name in names if name == "word/document.xml"]
                else:
                    names = [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
                    names.sort(key=lambda name: int(re.search(r"(\d+)", name).group(1)))
                paragraphs: list[str] = []
                for name in names:
                    root = ElementTree.fromstring(archive.read(name))
                    words = [
                        element.text.strip()
                        for element in root.iter()
                        if element.tag.rsplit("}", 1)[-1] in {"t", "instrText"}
                        and element.text
                        and element.text.strip()
                    ]
                    if words:
                        paragraphs.append(" ".join(words))
                return "\n\n".join(paragraphs)
        except Exception:
            return ""

    def _sectionise(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        title = "General"
        lines: list[str] = []

        def flush() -> None:
            body = "\n".join(lines).strip()
            if body:
                sections.append((title, body))

        for line in str(text).splitlines():
            stripped = line.strip()
            if self._is_heading(stripped):
                flush()
                title = self._HEADING_RE.sub("", stripped).strip() or stripped
                lines.clear()
            else:
                lines.append(stripped)
        flush()
        return sections

    @classmethod
    def _is_heading(cls, line: str) -> bool:
        if not line:
            return False
        if cls._HEADING_RE.match(line):
            return True
        words = line.split()
        return len(line) <= 100 and 1 <= len(words) <= 12 and line.isupper()

    @staticmethod
    def _normalise_sections(sections: Iterable[Any]) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for section in sections:
            if isinstance(section, Mapping):
                title = str(section.get("title") or section.get("heading") or "General")
                text = str(section.get("text") or section.get("content") or "")
            else:
                title, text = "General", str(section)
            if text.strip() or title != "General":
                result.append((title, text))
        return result

    def _chunk_section(self, title: str, text: str) -> Iterable[str]:
        words = " ".join(text.split()).split()
        if title != "General":
            words = [title, *words]
        if not words:
            return
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + self.chunk_size]).strip()
            if len(piece.split()) >= 8:
                yield piece
            if start + self.chunk_size >= len(words):
                break
