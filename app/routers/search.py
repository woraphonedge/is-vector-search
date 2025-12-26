import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Mock Gemini API URL (replace with actual URL when available)
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
)


# Request model
class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


@router.post("/search")
async def search_files(request: SearchRequest):
    """
    Search files using the Gemini API
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY environment variable not set"
        )

    headers = {
        "Content-Type": "application/json",
    }

    # Prepare the request payload for Gemini API
    payload = {
        "contents": [
            {"parts": [{"text": f"Search for files related to: {request.query}"}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 800,
            "temperature": 0.9,
            "topP": 1,
            "topK": 40,
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={api_key}",
                headers=headers,
                json=payload,
                timeout=30.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error from Gemini API: {response.text}",
                )

            # Process and return the response
            result = response.json()
            return {
                "query": request.query,
                "results": result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No results found"),
                "status": "success",
            }

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500, detail=f"Error connecting to Gemini API: {str(e)}"
        ) from e
