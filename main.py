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

# 【変更】1枚ずつ確実に処理するため、複数のListではなく単一の file を受け取る形に変更しました
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
            画像内の問題を読み取り、デジタルで解ける形式（選択式や穴埋め）に変換してください。
            写真の中に「表」や「グラフ」が含まれている場合は、HTMLの <table> タグを使用して見やすい表形式で解説に組み込んでください。
            【重要】JSONエラーを防ぐため、table, tr, th, tdタグには class や style などの装飾属性を一切付けず、最もシンプルなタグのみを使用してください。
            正解と、詳細な理由・解説を含めてください。
            以下のJSON形式で出力してください。
            {{
                "type": "workbook",
                "questions": [
                    {{
                        "question_text": "問題文",
                        "options": ["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
                        "answer": "正解の選択肢",
                        "explanation": "解説"
                    }}
                ]
            }}
            """
        else:
            prompt = f"""
            1. 画像の文章を抽出し、テストに出やすい重要語句を <mark class='ai-mark'>タグで囲んでください。
            2. 写真の中に「表」や「グラフ」が含まれている場合、絶対に無視せず、HTMLの <table> タグを使用して視覚的な表として抽出テキスト内に組み込んでください。
               【最重要】文字数オーバーによるJSONエラーを確実に防ぐため、HTMLタグには class や style などの装飾属性を一切付けないでください。最もシンプルな <table><thead><tr><th>...</th></tr></thead><tbody><tr><td>...</td></tr></tbody></table> のみを使用してください。
               ※グラフの場合は、目盛りから読み取れる数値を推測し、シンプルな表形式に変換して出力してください。
            3. その内容から、確認のためのオリジナル問題を{num_questions}問作成してください。もし指定された問題数が0の場合は、空の配列 [] にしてください。
            以下のJSON形式で出力してください。
            {{
                "type": "textbook",
                "extracted_text": "抽出テキストと生成されたシンプルなHTMLテーブル",
                "generated_questions": [
                    {{
                        "question_text": "作成した問題文",
                        "options": ["選択肢1", "選択肢2", "選択肢3"],
                        "answer": "正解",
                        "explanation": "解説"
                    }}
                ]
            }}
            """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
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

        raw_text = raw_text.replace('```json', '').replace('```', '').strip()
        
        try:
            parsed_data = json.loads(raw_text, strict=False)
            return JSONResponse(content=parsed_data)
        except json.JSONDecodeError:
            return JSONResponse(content={"error": "AIの出力データが途中で切れてしまいました。画像の文字密度が高すぎる可能性があります。"}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)