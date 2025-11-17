import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Base class for document metadata models."""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_by_alias=True,
        extra="forbid",
        from_attributes=True,
    )

    id: int | None = None
    file_md5_checksum: str = Field(alias="fileMd5Checksum")
    file_name: str = Field(alias="fileName")
    type: str
    category: str
    tags: List[str]
    parsed_at: datetime.datetime = Field(alias="parsedAt")
    published_date: datetime.datetime | None = Field(
        default=None, alias="publishedDate"
    )
    user_id: str | None = Field(default=None, alias="userId")
    product_type: str | None = Field(default=None, alias="productType")
    service_type: str | None = Field(default=None, alias="serviceType")
    json_file_path: str = Field(alias="jsonFilePath")
    # New status fields for file and JSON lifecycle
    file_status: str | None = Field(default=None, alias="fileStatus")
    json_status: str | None = Field(default=None, alias="jsonStatus")
    created_at: datetime.datetime | None = Field(default=None, alias="createdAt")
    updated_at: datetime.datetime | None = Field(default=None, alias="updatedAt")


class ChatRequest(BaseModel):
    query: str
    file_reference: str  # md5 checksum


class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]


class ExtractionResponse(BaseModel):
    message: str
    file_reference: str  # This will be the md5 checksum
    parsed_json: Dict[str, Any]
