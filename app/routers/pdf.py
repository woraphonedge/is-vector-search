import io
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pypdf import PdfReader, PdfWriter

from app.routers.document_extraction_pg import _find_uploaded_file_path

router = APIRouter()


def _get_pdf_path(file_md5_checksum: str) -> Path:
    path = _find_uploaded_file_path(file_md5_checksum)
    if not path:
        raise HTTPException(status_code=404, detail="file not found")

    if path.suffix.lower() == ".pdf":
        return path

    # If multiple artifacts exist with the same checksum prefix (e.g. .json, .pdf),
    # prefer a real PDF if present.
    pdf_candidate = path.with_suffix(".pdf")
    if pdf_candidate.exists() and pdf_candidate.is_file():
        return pdf_candidate

    raise HTTPException(status_code=400, detail="file is not a pdf")


@router.get("/original")
def get_pdf_original(file_md5_checksum: str = Query(...)):
    path = _get_pdf_path(file_md5_checksum)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.get("/page")
def get_pdf_page(
    file_md5_checksum: str = Query(...),
    page_number: int = Query(..., ge=0),
):
    try:
        path = _get_pdf_path(file_md5_checksum)

        reader = PdfReader(str(path))
        total = len(reader.pages)
        page_index = page_number
        if page_index < 0 or page_index >= total:
            raise HTTPException(status_code=400, detail="invalid page index")

        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])

        buffer = io.BytesIO()
        writer.write(buffer)
        buffer.seek(0)

        filename = f"{path.stem}_page_{page_number + 1}.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
