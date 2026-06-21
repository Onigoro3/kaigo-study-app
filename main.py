import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
# 【修正】FileResponse を追加してHTMLを返せるようにしました
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

# 【新規追加】トップページ（URLそのまま）にアクセスされたら index.html を表示する
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
        
        if mode == "workbook":
            prompt = """
            あなたは優秀な介護請求事務の講師です。添付された画像は「問題集」のページです。
            以下のJSON形式でデータを出力してください。
            
            1. 画像内の問題を読み取り、デジタルで解ける形式（選択式や穴埋め）に変換してください。
            2. 正解の提示とともに、「なぜその答えになるのか（根拠となる法律や算定要件）」の詳細な理由・解説を必ず含めてください。
            
            【出力JSONフォーマット例】
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
            あなたは優秀な介護請求事務の講師です。添付された画像は「教科書」のページです。
            以下のJSON形式でデータを出力してください。
            
            1. テキスト化: 画像の文章を抽出し、テストに出やすい重要語句を <mark class='ai-mark'>タグで囲んでください。
            2. 問題生成: その内容から、確認のためのオリジナル問題を3問作成してください。正解とともに、「なぜその答えになるのか」の詳細な理由・解説を必ず含めてください。
            
            【出力JSONフォーマット例】
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

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_data,
                    mime_type=file.content_type,
                )
            ]
        )
        
        result_text = response.text.replace('```json', '').replace('```', '').strip()
        return JSONResponse(content=json.loads(result_text))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)