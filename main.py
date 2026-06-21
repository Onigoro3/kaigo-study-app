import os
import json
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("APIキーが設定されていません。.envファイルを確認してください。")

client = genai.Client(api_key=API_KEY)

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/analyze")
async def analyze_images(
    files: List[UploadFile] = File(...),
    mode: str = Form(...),
    num_questions: int = Form(3) # 【新規追加】画面から問題数を受け取る
):
    try:
        image_parts = []
        for file in files:
            data = await file.read()
            image_parts.append(
                types.Part.from_bytes(data=data, mime_type=file.content_type)
            )
        
        # 【変更】f文字列を使用し、問題数をAIに直接指示。JSONの括弧は {{ }} でエスケープしています。
        if mode == "workbook":
            prompt = f"""
            画像内の問題を読み取り、デジタルで解ける形式（選択式や穴埋め）に変換してください。
            写真の中に「図、表、グラフ」が含まれている場合は、その内容もテキストとして詳しく説明し、問題を解くための手がかりとして含めてください。
            正解と、詳細な理由・解説（根拠となる法律や算定要件など）を含めてください。
            以下のJSON形式で出力してください。
            {{
                "type": "workbook",
                "questions": [
                    {{
                        "question_text": "問題文（図表の説明含む）",
                        "options": ["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
                        "answer": "正解の選択肢",
                        "explanation": "なぜこの答えになるのかの詳しい理由と解説"
                    }}
                ]
            }}
            """
        else:
            prompt = f"""
            1. 画像の文章を抽出し、テストに出やすい重要語句を <mark class='ai-mark'>タグで囲んでください。
            2. 写真の中に「図、表、グラフ」が含まれている場合は無視せず、その図解が何を表しているのか詳細にテキスト化（可能ならMarkdownの表形式で）して抽出テキストに含めてください。
            3. その内容から、確認のためのオリジナル問題を{num_questions}問作成してください。正解と詳細な解説を含めてください。
            以下のJSON形式で出力してください。
            {{
                "type": "textbook",
                "extracted_text": "抽出されたテキストと図表の解説（<mark class='ai-mark'>重要語句</mark>）",
                "generated_questions": [
                    {{
                        "question_text": "作成した問題文",
                        "options": ["選択肢1", "選択肢2", "選択肢3"],
                        "answer": "正解",
                        "explanation": "なぜこの答えになるのかの詳しい理由と解説"
                    }}
                ]
            }}
            """

        contents = [prompt] + image_parts

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )
        )
        
        return JSONResponse(content=json.loads(response.text))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)