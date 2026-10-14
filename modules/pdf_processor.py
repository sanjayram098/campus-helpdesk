"""PDF text extraction and chunking for the College Helpdesk RAG system.

This module is responsible for:
  * Validating uploaded PDF files (extension, size, readability).
  * Extracting text page by page using PyMuPDF.
  * Splitting extracted text into overlapping chunks.
  * Returning structured chunk records with stable document IDs and metadata.

It is intentionally defensive: a single bad PDF must never crash the whole
application. Scanned (image-only) PDFs yield no extractable text and are
reported as such (OCR is out of scope but can be added later).
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pymupdf as fitz

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_PDF_SIZE_MB


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PageText:
    """Text extracted from a single PDF page."""

    page_number: int  # 1-based page number
    text: str


@dataclass
class DocumentChunk:
    """A single chunk of text ready to be embedded and stored."""

    doc_id: str          # stable hash of the original file bytes
    filename: str        # original file name
    page_number: int     # 1-based page number this chunk came from
    chunk_index: int     # position of this chunk within the page
    text: str            # the chunk text
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Result of processing one PDF document."""

    doc_id: str
    filename: str
    success: bool
    chunks: List[DocumentChunk] = field(default_factory=list)
    error: Optional[str] = None
    page_count: int = 0
    is_scanned: bool = False  # True if pages exist but no text was extracted


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_pdf(filename: str, file_bytes: bytes) -> Optional[str]:
    """Return an error message if the file is invalid, else None."""
    if not filename.lower().endswith(".pdf"):
        return f"File '{filename}' is not a PDF (extension must be .pdf)."

    if not file_bytes:
        return f"File '{filename}' is empty."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        return (
            f"File '{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{MAX_PDF_SIZE_MB} MB limit."
        )

    # PDF magic number check (very light corruption guard).
    if not file_bytes[:5].startswith(b"%PDF-"):
        return f"File '{filename}' does not look like a valid PDF (bad header)."

    return None


# ---------------------------------------------------------------------------
# Stable document ID
# ---------------------------------------------------------------------------
def compute_doc_id(file_bytes: bytes) -> str:
    """Return a stable SHA-256 hash of the file bytes.

    Using a content hash means identical files get the same ID, which lets us
    detect duplicates and skip re-indexing unchanged documents.
    """
    return hashlib.sha256(file_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_pages(file_bytes: bytes) -> List[PageText]:
    """Extract text from every page of a PDF.

    Raises:
        fitz.FileDataError: if the PDF is corrupted or unreadable.
        RuntimeError: if the PDF is encrypted and cannot be decrypted.
    """
    pages: List[PageText] = []
    doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    try:
        if doc.is_encrypted:
            # Try an empty password; many "encrypted" PDFs are just restricted.
            if not doc.authenticate(""):
                raise RuntimeError(
                    "This PDF is password-protected and could not be opened. "
                    "Please upload an unencrypted copy."
                )

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            pages.append(PageText(page_number=page_index + 1, text=text))
    finally:
        doc.close()

    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split a long string into overlapping chunks.

    The split is character-based with a configurable overlap so that sentences
    cut near a boundary are still captured by the neighbouring chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and < chunk_size")

    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
        # Guard against infinite loops when overlap is large relative to text.
        if start >= n:
            break
    return chunks


# ---------------------------------------------------------------------------
# High-level processing
# ---------------------------------------------------------------------------
def process_pdf(
    source: Union[str, Path, bytes],
    filename: Optional[str] = None,
) -> ProcessedDocument:
    """Process a single PDF from a file path or raw bytes.

    Parameters
    ----------
    source : str | Path | bytes
        Either a path to a PDF file or the raw bytes of a PDF.
    filename : str, optional
        Original filename (used only for display/metadata). Required when
        ``source`` is bytes; inferred from the path otherwise.

    Returns
    -------
    ProcessedDocument
        Always returns a result object; check ``.success`` and ``.error``.
    """
    # Resolve bytes and filename.
    if isinstance(source, (str, Path)):
        path = Path(source)
        file_bytes = path.read_bytes()
        if filename is None:
            filename = path.name
    elif isinstance(source, (bytes, bytearray)):
        file_bytes = bytes(source)
        if not filename:
            filename = "document.pdf"
    else:
        return ProcessedDocument(
            doc_id="",
            filename=filename or "unknown",
            success=False,
            error="Unsupported source type. Provide a file path or bytes.",
        )

    doc_id = compute_doc_id(file_bytes)

    # Validate.
    err = _validate_pdf(filename, file_bytes)
    if err:
        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            success=False,
            error=err,
        )

    # Extract text.
    try:
        pages = extract_pages(file_bytes)
    except RuntimeError as exc:
        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            success=False,
            error=str(exc),
        )
    except fitz.FileDataError as exc:
        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            success=False,
            error=f"Corrupted or unreadable PDF: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - last-resort guard
        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            success=False,
            error=f"Unexpected error while reading PDF: {exc}",
        )

    if not pages:
        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            success=True,
            page_count=0,
            error="The PDF contains no pages.",
        )

    # Build chunks, preserving filename + page number metadata.
    chunks: List[DocumentChunk] = []
    total_text = 0
    for page in pages:
        page_chunks = split_text_into_chunks(page.text)
        total_text += len(page.text)
        for idx, chunk_text in enumerate(page_chunks):
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page.page_number,
                    chunk_index=idx,
                    text=chunk_text,
                    metadata={
                        "filename": filename,
                        "page_number": page.page_number,
                        "doc_id": doc_id,
                    },
                )
            )

    # A scanned PDF has pages but little or no extractable text.
    is_scanned = bool(pages) and total_text == 0

    return ProcessedDocument(
        doc_id=doc_id,
        filename=filename,
        success=True,
        chunks=chunks,
        page_count=len(pages),
        is_scanned=is_scanned,
        error=(
            "No text could be extracted. This is likely a scanned (image-only) "
            "PDF. OCR would be required to read its contents."
            if is_scanned and not chunks
            else None
        ),
    )


def process_pdfs(
    sources: Iterable[Union[str, Path, bytes]],
    filenames: Optional[Iterable[str]] = None,
) -> List[ProcessedDocument]:
    """Process a batch of PDFs. One failure does not stop the others."""
    results: List[ProcessedDocument] = []
    names = list(filenames) if filenames is not None else []
    for i, src in enumerate(sources):
        fname = names[i] if i < len(names) else None
        try:
            results.append(process_pdf(src, filename=fname))
        except Exception as exc:  # noqa: BLE001 - safety net
            results.append(
                ProcessedDocument(
                    doc_id="",
                    filename=fname or f"document_{i}.pdf",
                    success=False,
                    error=f"Unexpected failure: {exc}",
                )
            )
    return results
