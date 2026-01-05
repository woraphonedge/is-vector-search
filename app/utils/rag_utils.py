import re
from collections import defaultdict
from typing import Literal

import tiktoken
from langchain_core.documents import Document

tokenizer = tiktoken.get_encoding("o200k_base")
TOKEN_LIMIT = 4096


def prepare_document(
    file_name: str,
    md5_checksum: str,
    parsed_json: dict,
    publishedDate: str,
    chunk_type: Literal["page", "chunk"] = "page",
):
    """
    Prepare document for vectorization.

    Args:
        file_name: The name of the file to be processed.
        parsed_json: The parsed JSON data to be processed.
        chunk_type: The type of chunk to be processed. Can be "page" or "chunk".
    """
    base_name = re.sub(r"\W+", "_", file_name.rsplit(".", 1)[0]).lower()

    # Step 1: Pre-process all individual chunks to ensure none exceed TOKEN_LIMIT.
    # Support both the legacy agentic-doc schema and the new landingai_ade ParseResponse schema.
    pre_processed_chunks = []
    for item in parsed_json.get("chunks", []):
        # --- Normalize schema fields ---
        # source_chunk_type: legacy uses "chunk_type"; ADE uses "type" (e.g. "chunkText", "chunkMarginalia").
        source_chunk_type = item.get("chunk_type") or item.get("type", "")

        # text content: legacy uses "text"; ADE uses "markdown".
        text = item.get("text") or item.get("markdown", "")

        # grounding/page: legacy uses a list of groundings; ADE uses a single object.
        grounding = item.get("grounding")
        page_no = None
        if isinstance(grounding, list) and grounding:
            page_no = grounding[0].get("page")
        elif isinstance(grounding, dict):
            page_no = grounding.get("page")

        # chunk id: legacy uses "chunk_id"; ADE uses "id".
        original_chunk_id = item.get("chunk_id") or item.get("id")

        # Skip if we don't have essential fields
        if not text or page_no is None or original_chunk_id is None:
            continue

        # Skip marginalia / non-primary text chunks
        if source_chunk_type:
            lowered = str(source_chunk_type).lower()
            if "marginalia" in lowered:
                continue

        if len(text) < 20:
            continue
        tokens = tokenizer.encode(text, allowed_special={"<space>"})

        if len(tokens) > TOKEN_LIMIT:
            for i, chunk_start in enumerate(range(0, len(tokens), TOKEN_LIMIT)):
                token_chunk = tokens[chunk_start : chunk_start + TOKEN_LIMIT]
                split_content = tokenizer.decode(token_chunk)
                pre_processed_chunks.append(
                    {
                        "page_content": split_content,
                        "page_number": page_no,
                        "chunk_id": f"{original_chunk_id}_part_{i+1}",
                    }
                )
        else:
            pre_processed_chunks.append(
                {
                    "page_content": text,
                    "page_number": page_no,
                    "chunk_id": original_chunk_id,
                }
            )

    # Step 2: Combine the pre-processed chunks based on the desired chunk_type.
    final_chunks = []
    if chunk_type == "page":
        pages = defaultdict(list)
        for chunk in pre_processed_chunks:
            pages[chunk["page_number"]].append(chunk)

        for page_no, chunks_on_page in sorted(pages.items()):
            if not chunks_on_page:
                continue

            current_texts = []
            current_token_count = 0
            page_part_counter = 1

            for chunk in chunks_on_page:
                chunk_text = chunk["page_content"]
                chunk_tokens = tokenizer.encode(chunk_text, allowed_special={"<space>"})

                if (
                    current_token_count + len(chunk_tokens) > TOKEN_LIMIT
                    and current_texts
                ):
                    page_content = "\n\n".join(current_texts)
                    chunk_id = f"{base_name}_page_{page_no}_part_{page_part_counter}"
                    final_chunks.append(
                        {
                            "page_content": page_content,
                            "page_number": page_no,
                            "chunk_id": chunk_id,
                            "published_date": publishedDate,
                        }
                    )
                    page_part_counter += 1
                    current_texts = [chunk_text]
                    current_token_count = len(chunk_tokens)
                else:
                    current_texts.append(chunk_text)
                    current_token_count += len(chunk_tokens)

            if current_texts:
                page_content = "\n\n".join(current_texts)
                if page_part_counter > 1:
                    chunk_id = f"{base_name}_page_{page_no}_part_{page_part_counter}"
                else:
                    chunk_id = f"{base_name}_page_{page_no}"
                final_chunks.append(
                    {
                        "page_content": page_content,
                        "page_number": page_no,
                        "chunk_id": chunk_id,
                        "published_date": publishedDate,
                    }
                )

    elif chunk_type == "chunk":
        final_chunks = pre_processed_chunks
    else:
        raise ValueError(f"Invalid chunk_type: {chunk_type}")

    # 3) Build Document objects
    documents = []
    for item in final_chunks:
        page_content = item["page_content"]

        metadata = {
            "file_name": file_name,
            "file_md5_checksum": md5_checksum,
            "published_date": publishedDate,
            "page_number": item["page_number"],
            "chunk_type": chunk_type,
            "chunk_id": item["chunk_id"],
            "token_count": len(
                tokenizer.encode(page_content, allowed_special={"<space>"})
            ),
        }

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    return documents
