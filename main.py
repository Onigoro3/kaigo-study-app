import os
import json
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
# 【新規】構造化データ（設計図）を作成するためのライブラリ
from pydantic import BaseModel
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

# ==========================================
# 🌟 【最重要】AIの出力を100%綺麗なJSONにするための設計図定義
# ==========================================
class QuizQuestion(BaseModel):
    question_text: str
    options: List[str]
    answer: str
    explanation: str

class StudyAnalyzeResponse(BaseModel):
    type: str
    extracted_text: str
    generated_questions: List[QuizQuestion]

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    mode: str = Form(...),
    num_questions: int = Form(3)
):
    try:
        image_data = await file.read()
        image_part = types.Part.from_bytes(data=image_data, mime_type=file.content_type)
        
        if mode == "workbook":
            prompt = f"""
            画像内の問題を読み取り、デジタルで解ける形式（選択式や穴埋め）に変換して指定のデータ構造に格納してください。
            写真の中に「表」や「グラフ」が含まれている場合は、HTMLの <table> タグを使用して見やすい表形式で解説（explanation）に組み込んでください。
            
            ※extracted_text は空文字（""）にしてください。
            """
        else:
            prompt = f"""
            添付された教科書の画像を解析し、以下の指示に従って指定のデータ構造に格納してください。
            1. 画像の文章を抽出し、テストに出やすい重要語句を <mark class='ai-mark'>タグで囲んで extracted_text に格納してください。
            2. 写真の中に「表」や「グラフ」が含まれている場合、絶対に無視せず、HTMLの <table> タグを使用したシンプルな表（装飾属性なし）として抽出し、extracted_text 内に組み込んでください。
            3. その内容から、確認のためのオリジナル問題を{num_questions}問作成し、generated_questions に格納してください。
            """

        # 🌟 response_schema を指定することで、AIは100%パース可能な綺麗なデータしか出せなくなります
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StudyAnalyzeResponse, # これがパースエラーを根本解決する魔法の設定です
                temperature=0.2,
                max_output_tokens=8192,
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                ]
            )
        )
        
        raw_text = response.text
        if not raw_text:
            return JSONResponse(content={"error": "AIがテキストを生成できませんでした。別の画像をお試しください。"}, status_code=500)

        # システムが成形を保証しているため、安全にパースして即座に画面へ返却できます
        parsed_data = json.loads(raw_text)
        return JSONResponse(content=parsed_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)