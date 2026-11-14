
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader
from docx import Document


def read_uploaded_file(uploaded_file):
    """
    Read text from an uploaded PDF, DOCX, TXT or Markdown file.

    Supports Streamlit UploadedFile objects.
    Returns extracted text as a string.
    """

    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    # ---------------------------------
    # PDF FILE
    # ---------------------------------

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))

        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        text = "\n".join(pages)

    # ---------------------------------
    # WORD DOCUMENT
    # ---------------------------------

    elif filename.endswith(".docx"):
        document = Document(BytesIO(file_bytes))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

    # ---------------------------------
    # TEXT AND MARKDOWN FILES
    # ---------------------------------

    elif filename.endswith((".txt", ".md")):
        text = file_bytes.decode(
            "utf-8",
            errors="replace"
        )

    else:
        raise ValueError(
            "Unsupported file format. "
            "Please upload PDF, DOCX, TXT or MD."
        )

    if not text.strip():
        raise ValueError(
            "No readable text was found in this document. "
            "If this is a scanned PDF, OCR may be required."
        )

    return text.strip()
