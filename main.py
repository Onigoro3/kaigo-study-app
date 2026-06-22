import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
# 【新規】Supabaseのインポート
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

# 【新規】Supabaseクライアントの初期化設定
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

# 【新規】データ受取用の形
class SaveDataRequest(BaseModel):
    data: Dict[str, Any]

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# 【新規】Supabaseから全端末共通のデータを読み込む
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

# 【新規】全端末共通のデータをSupabaseへ保存・上書きする
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

if __name__ == "__main__":
    # Renderが指定するポート番号を自動で取得する（無ければ8000を使う）
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)