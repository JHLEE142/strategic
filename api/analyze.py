from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from backend.logic import analyze_accounts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze(prompt: str = Form(""), file: UploadFile = File(...)) -> Dict[str, Any]:
    if file.content_type not in {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }:
        raise HTTPException(status_code=400, detail="Excel 파일만 업로드할 수 있습니다.")

    file_bytes = await file.read()
    try:
        payload = analyze_accounts(prompt, file_bytes)
    except Exception as error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(error)) from error

    return payload


handler = Mangum(app)

