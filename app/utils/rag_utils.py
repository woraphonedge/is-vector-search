import re
from collections import defaultdict
from typing import Any, Dict, Literal

import tiktoken
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import app.utils.supabase_db as sdb
from app.models import ChatRequest

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


def process_and_insert_documents(
    parsed_json: Dict[str, Any], file_name: str, md5_checksum: str, publishedDate: str
):
    documents = prepare_document(file_name, md5_checksum, parsed_json, publishedDate)
    # check if any document have token over 4096
    for doc in documents:
        if doc.metadata["token_count"] > 4096:
            print(
                f"Document {doc.metadata['chunk_id']} has token count over 4096, skipping."
            )
    if documents:
        # batch insert documents up to 20
        for i in range(0, len(documents), 20):
            document_chunk = documents[i : i + 20]
            sdb.vector_store.add_documents(
                documents=document_chunk,
                ids=[doc.metadata["chunk_id"] for doc in document_chunk],
            )
        print(
            f"Successfully added {len(documents)} document chunks to the vector store."
        )


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


async def get_rag_response(chat_request: ChatRequest) -> Dict[str, Any]:
    """
    Process a chat request by retrieving relevant documents and generating a response.

    This function performs RAG (Retrieval-Augmented Generation) by:
    1. Using semantic search to retrieve relevant documents
    2. Combining retrieved documents with the user query
    3. Generating a response using an LLM

    Args:
        chat_request: The ChatRequest object containing the user's query

    Returns:
        Dict[str, Any]: A dictionary containing the generated response and metadata
    """
    # https://docs.trychroma.com/docs/querying-collections/full-text-search
    # Retriever for page-level chunks
    page_retriever = sdb.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "filter": {
                "$and": [
                    {"chunk_type": {"$eq": "page"}},
                    {"token_count": {"$gt": 20}},
                    {"chunk_id": {"$ne": "mock-chunk-id-1"}},
                ]
            },
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.8,
            "score_threshold": 0.9,
        },
    )

    # Retriever for other chunks (non-page)
    chunk_retriever = sdb.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "filter": {
                "$and": [
                    {"chunk_type": {"$ne": "page"}},
                    {"token_count": {"$gt": 20}},
                    {"chunk_id": {"$ne": "mock-chunk-id-1"}},
                ]
            },
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5,
            "score_threshold": 0.9,
        },
    )

    # Retrieve documents from both
    page_docs = await page_retriever.ainvoke(chat_request.query)
    chunk_docs = await chunk_retriever.ainvoke(chat_request.query)

    # Combine and remove duplicates
    all_docs = page_docs + chunk_docs
    unique_docs_dict = {doc.metadata["chunk_id"]: doc for doc in all_docs}
    unique_docs = list(unique_docs_dict.values())

    system_prompt = """
    You are a financial consultant for question-answering tasks.
    - If you don't know the answer, say that you don't know.
    - Keep the answer in markdown format, concise and use bulleted-point format if needed.
    - If needed, mention name entity for clarification.
    - Only assist with financial, product, or internal business questions.
    Do not provide general lifestyle or non-financial advice.
    - Use the following pieces of retrieved context to answer
    the question.
    \n\n
    {context}
    \n\n
    Do not speculate or generate information beyond what is retrieved.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Format docs to context string explicitly, then run a simple chain: prompt -> llm -> str parser
    formatted_context = _format_docs(unique_docs)

    qa_chain = prompt | sdb.llm | StrOutputParser()

    answer = await qa_chain.ainvoke(
        {
            "input": chat_request.query,
            "context": formatted_context,
        }
    )

    return {"answer": answer, "context": unique_docs}
