import os
import json
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
async def analyze_image(
    file: UploadFile = File(...),
    mode: str = Form(...)
):
    try:
        image_data = await file.read()
        
        # プロンプト（指示）もAIが処理しやすいように短く・明確に最適化しました
        if mode == "workbook":
            prompt = """
            画像内の問題を読み取り、デジタルで解ける形式（選択式や穴埋め）に変換してください。
            正解と、詳細な理由・解説（根拠となる法律や算定要件など）を含めてください。
            以下のJSON形式で出力してください。
            {
                "type": "workbook",
                "questions": [
                    {
                        "question_text": "問題文",
                        "options": ["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
                        "answer": "正解の選択肢",
                        "explanation": "なぜこの答えになるのかの詳しい理由と解説"
                    }
                ]
            }
            """
        else:
            prompt = """
            1. 画像の文章を抽出し、テストに出やすい重要語句を <mark class='ai-mark'>タグで囲んでください。
            2. その内容から、確認のためのオリジナル問題を3問作成してください。正解と詳細な解説を含めてください。
            以下のJSON形式で出力してください。
            {
                "type": "textbook",
                "extracted_text": "抽出されたテキスト（<mark class='ai-mark'>重要語句</mark>）",
                "generated_questions": [
                    {
                        "question_text": "作成した問題文",
                        "options": ["選択肢1", "選択肢2", "選択肢3"],
                        "answer": "正解",
                        "explanation": "なぜこの答えになるのかの詳しい理由と解説"
                    }
                ]
            }
            """

        # 【重要】AIを「高速・高精度モード」に設定
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_data,
                    mime_type=file.content_type,
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", # AIにJSON形式での出力を強制（超高速化）
                temperature=0.2, # ランダム性を下げて、教科書に忠実な正確な回答を出させる
            )
        )
        
        # JSONモードをオンにしたため、不要な文字を削る処理が不要になりました
        return JSONResponse(content=json.loads(response.text))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)