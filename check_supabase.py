import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. .envファイルから環境変数を読み込む
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 2. 設定のチェック
if not SUPABASE_URL or not SUPABASE_KEY:
    print("【エラー】.envファイルに SUPABASE_URL または SUPABASE_KEY が設定されていません。")
    exit(1)

try:
    # 3. Supabaseクライアントの初期化
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("====================================")
    print("✨ Supabaseへの接続クライアントの作成に成功しました！")
    print(f"接続先URL: {SUPABASE_URL}")
    print("====================================")
    
    # 4. 正しい通信テスト（先ほど作成した study_apps テーブルにアクセスしてみる）
    response = supabase.table("study_apps").select("*").limit(1).execute()
    
    print("🎉 サーバーとの通信テスト（テーブルの確認）も正常に完了しました！")
    print("Supabaseの準備はすべて完璧に整いました。")
    print("====================================")

except Exception as e:
    print("====================================")
    print("❌ Supabaseへの接続または通信に失敗しました。")
    print(f"エラー詳細: {str(e)}")
    print("====================================")