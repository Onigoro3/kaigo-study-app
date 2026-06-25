import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from supabase import create_client, Client

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

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

class QuizQuestion(BaseModel):
    question_text: str
    options: List[str]
    answer: str
    explanation: str

class StudyAnalyzeResponse(BaseModel):
    type: str
    extracted_text: str
    generated_questions: List[QuizQuestion]
    target_form: Optional[str] = None  # 実技問題で指定された様式を保持

class SaveDataRequest(BaseModel):
    data: Dict[str, Any]

class GenerateQuizRequest(BaseModel):
    text: str
    num_questions: int

class QuizOnlyResponse(BaseModel):
    generated_questions: List[QuizQuestion]

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.get("/api/load-data")
async def load_data():
    if not supabase_client:
        return JSONResponse(content={"error": "Supabase設定が見つかりません。"}, status_code=500)
    try:
        response = supabase_client.table("study_apps").select("data").eq("id", 1).execute()
        if response.data and len(response.data) > 0:
            return JSONResponse(content=response.data[0]["data"])
        return JSONResponse(content={})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/save-data")
async def save_data(req: SaveDataRequest):
    if not supabase_client:
        return JSONResponse(content={"error": "Supabase設定が見つかりません。"}, status_code=500)
    try:
        supabase_client.table("study_apps").upsert({"id": 1, "data": req.data}).execute()
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

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
        elif mode == "practice":
            prompt = f"""
            画像は介護事務の実技問題（請求実技問題など）です。
            問題文および提示されている条件（保険者、被保険者、事業所、提供サービスなど）をすべて読み取り、HTMLで見やすい形式（改行や <ul>, <table> タグを適宜使用）に整形して extracted_text に格納してください。
            また、問題文中で指定されている様式（例：「様式第二」「様式第三」など）を判別し、以下のいずれかの文字列を target_form に設定してください。
            "yoshiki-2", "yoshiki-3", "yoshiki-4", "yoshiki-5", "yoshiki-6", "yoshiki-7", "yoshiki-8", "yoshiki-9", "yoshiki-10", "yoshiki-11"
            判別できない場合や指定がない場合は null にしてください。
            generated_questions は空のリスト [] とし、typeは "practice" としてください。
            """
        else:
            prompt = f"""
            添付された教科書の画像を解析し、以下の指示に従って指定のデータ構造に格納してください。
            1. 画像の文章を抽出し、テストに出やすい重要語句を <mark class='ai-mark'>タグで囲んで extracted_text に格納してください。
            2. 写真の中に「表」や「グラフ」が含まれている場合、絶対に無視せず、HTMLの <table> タグを使用したシンプルな表（装飾属性なし）として抽出し、extracted_text 内に組み込んでください。
            3. その内容から、確認のためのオリジナル問題を{num_questions}問作成し、generated_questions に格納してください。
            """

        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StudyAnalyzeResponse,
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
            return JSONResponse(content={"error": "AIがテキストを生成できませんでした。"}, status_code=500)

        parsed_data = json.loads(raw_text)
        return JSONResponse(content=parsed_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/generate-quiz")
async def generate_quiz_from_text(req: GenerateQuizRequest):
    try:
        prompt = f"""
        以下の学習テキストの内容に基づいて、学習内容を復習・確認するためのオリジナル問題を {req.num_questions} 問作成してください。
        必ず指定されたデータ構造（generated_questions）に格納して返答してください。

        【学習テキスト】
        {req.text}
        """

        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuizOnlyResponse,
                temperature=0.3,
                max_output_tokens=8192,
            )
        )
        
        raw_text = response.text
        if not raw_text:
            return JSONResponse(content={"error": "AIがテキストを生成できませんでした。"}, status_code=500)

        parsed_data = json.loads(raw_text)
        return JSONResponse(content=parsed_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)