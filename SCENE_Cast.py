#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# SCENE分析（Systematic Character Extraction for Narrative Epilogue）
# SCENE_Cast分析
#=============================================================

#====================================================
# SCENE_Cast分析の準備
#====================================================

# ライブラリの準備
from bs4 import BeautifulSoup
import google.genai as genai 
from dotenv import load_dotenv 
import os
import pandas as pd
from sqlalchemy import create_engine
from pydantic import BaseModel, Field # JSONスキーマの定義に利用
from concurrent.futures import ThreadPoolExecutor # 並行処理に必要なモジュール

# モジュールの準備
import My_Global as g

# PostgreSQLの接続設定
dotenv_path = '/Users/trueocean/Desktop/Python_Code/Project_Key/.env'
load_dotenv(dotenv_path)

connection_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME')
    }

engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format(**connection_config))


# APIキーの設定とクライアント初期化
dotenv_path = "/Users/trueocean/Desktop/Python_Code/Project_Key/.env"
load_dotenv(dotenv_path) 
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key) 
MODEL = "gemini-2.5-flash" # 高速モデルを維持

print("APIクライアント初期化完了（モデル: gemini-2.5-flash）")


#====================================================
# ５代血統表（HTMLファイル）を堅牢に読み込む関数
#====================================================

def read_text_with_fallback(path, encodings=None):
    if encodings is None:
        encodings = ['utf-8', 'utf-8-sig', 'cp932', 'shift_jis', 'euc_jp', 'iso2022_jp', 'latin-1']
    last_exc = None
    with open(path, 'rb') as f:
        raw = f.read()
    for enc in encodings:
        try:
            return raw.decode(enc), enc
        except Exception as e:
            last_exc = e
    return raw.decode('utf-8', errors='replace'), 'utf-8 (forced replace)'


#====================================================
# HTMLクリーンアップ関数の定義
#====================================================

def clean_html_for_analysis(html_content: str) -> str:
    """
    HTMLから、馬の基本情報、血統表のデータ、およびクロス情報を抽出し、
    HTMLタグを削除して純粋なテキストとして結合します。
    """
    soup = BeautifulSoup(html_content, 'lxml') 

    extracted_text = []

    # 1. <body>タグ内の全てのテキストを抽出
    #   -> これにより、H3や性別/年齢などの情報が含まれます。
    #   -> ただし、<table>タグ内の情報も含まれるため、後で重複を避ける。
    
    if soup.body:
        # <body>タグ内のテキストを改行区切りで取得
        # HTMLの構造上、このテキストにはH3や性別/年齢の情報が含まれる
        body_text_lines = soup.body.get_text('\n', strip=True).split('\n')
        
        # 2. 基本情報が含まれる最初の数行を抽出
        # 今回の構造では、<table>タグ以前の情報は最初の数行に集約されているはず
        # 安全を見て、最初の5行程度を抽出します
        for line in body_text_lines[:5]:
            line_stripped = line.strip()
            # 既に抽出済みの血統表のクロス情報部分（[Hail to Reason]など）は除外する
            if not line_stripped.startswith('['):
                extracted_text.append(line_stripped)
        
    
    # 3. 血統表の<table>タグを特定する
    blood_table_tag = soup.find('table') 

    if blood_table_tag:
        
        # 4. テーブルの中の純粋なテキストを抽出
        # get_text() を使用して、<td>タグ内の馬名や情報だけを取り出す
        # 重複を避けるため、抽出されるのは<table>タグ内部の血統情報のみ
        table_text = blood_table_tag.get_text('\n', strip=True)
        extracted_text.append(table_text)

        # 5. テーブルの直後にあるクロス情報を抽出する (以前のロジックを維持)
        current_tag = blood_table_tag.next_sibling
        
        while current_tag:
            if isinstance(current_tag, str):
                text_content = current_tag.strip()
                if text_content:
                    extracted_text.append(text_content)
            elif current_tag.name is not None and current_tag.name not in ('br', 'font'):
                 break
            
            current_tag = current_tag.next_sibling
    
    # 抽出した全てのテキストを改行で結合し、過剰な空白を整理して返します
    return '\n'.join(extracted_text).strip()


#====================================================
# 出力フォーマットを定義するJSONスキーマ（Pydanticモデル）
#====================================================

class AnalysisResult(BaseModel):
    """血統分析、キャラクター設定、自己紹介の結果を格納するスキーマ"""
    血統分析: str = Field(
        ..., 
        description="5代血統表に含まれる名馬やクロスに関する簡潔な分析結果。合計200字程度。"
    )
    キャラ設定: str = Field(
        ..., 
        description="タイプ、外見、性格、一人称、口調を含む、簡潔なキャラクター設定。合計200字程度。"
    )
    自己紹介: str = Field(
        ..., 
        description="設定に基づいた200字程度の簡潔な自己紹介のセリフ。「」を含む。"
    )


#====================================================
# Gemini API によるキャラ分析
#====================================================

def analyze_single_horse(features_text, blood_text, race_info, max_retries=5):
    """
    特徴（数値データ）と血統テキストを統合してGeminiに渡し、
    構造化されたキャラクターデータを取得する。
    """
    
    analysis_prompt = f"""
    提供された「能力データ」と「5代血統表」を分析し、キャラクター設定を生成してください。

    ## コンテキスト
    - 対象レース: {race_info}
    - 馬の能力・特徴データ: {features_text}
    - 馬の５代血統データ: {blood_text}
    
    ## タスク内容 (日本語で出力)
    1. **血統・能力分析**: 
       血統背景（名馬やクロス）が、現在の能力数値（先行指数、基礎能力、適合率等）にどう影響しているか、物語的に分析してください（200字程度）。
    2. **キャラクター設定**: 
       分析結果、性別、年齢、毛色、そして「勝負服色」のイメージを考慮し、「タイプ」「外見」「性格」「一人称」「口調」を構成してください（200字程度）。
       「一人称」「口調」については、性別、年齢（2歳：幼い、3歳：若い、4歳：青年、5歳：成人、6歳：壮年、7歳以上：老齢）と、血統から推察される家の育ちを考慮して下さい。
       （参考）一人称の年齢順: ぼく、わたし、おれ、あたい、うち、おいら、僕、私、俺、自分、拙者、わし、わしゃ 等。口調の例: ですます調、丁寧、優しい、厳しい、関西弁、お嬢様、侍 等。
    3. **自己紹介**: 
       上記設定に基づき、レースへ臨む決意を含めた自己紹介のセリフを生成してください（200字程度）。

    回答は必ず指定されたJSONスキーマに従ってください。

    """

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[analysis_prompt],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisResult,
                )
            )
            
            # response.parsed を使うと、自動的に AnalysisResult オブジェクトになる
            if response.parsed:
                return response.parsed.model_dump()
            # もし response.text からパースする場合も同様
            data_model = AnalysisResult.model_validate_json(response.text)
            return data_model.model_dump()
        
        except Exception as e:
            print(f"Retry {attempt+1}: Error {e}")
            if attempt == max_retries - 1: return None


#====================================================
# 並列処理により時間短縮を図る
#====================================================

def process_all_horses_parallel(df, max_workers=5):
    """
    データフレームの全馬を並列で分析し、結果を結合して返す。
    max_workers: 同時に実行するスレッド数（Geminiの無料枠なら5〜8程度が安定）
    """
    # 重複防止 (既存のカラムがあれば消す) 
    target_cols = ['血統分析', 'キャラ設定', '自己紹介']
    df = df.drop(columns=[c for c in target_cols if c in df.columns], errors='ignore')
    
    # 1頭分の処理をラップする関数
    def task(row_data):
        idx, row = row_data
        print(f" 🐎{row['馬名']}の分析開始...")
        result = analyze_single_horse(
            features_text=row['特徴'],
            blood_text=row['血統情報'],
            race_info=Race_Info
        )
        if result:
            result['馬名'] = row['馬名']
            print(f" ⭕️{row['馬名']}の分析完了")
            return result
        else:
            print(f" ⚠️{row['馬名']}の分析失敗")
            return None

    # 並列実行の開始
    print(f"--- SCENE_Cast分析の並列処理を開始します（同時実行数: {max_workers}）... ---")
    results = []
    
    # rowをリスト化して渡す
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # mapを使って全行をタスクとして投入
        rows = list(df.iterrows())
        results = list(executor.map(task, rows))

    # None（失敗）を除去してデータフレーム化
    valid_results = [r for r in results if r is not None]
    res_df = pd.DataFrame(valid_results)
    
    # 元のデータフレームに結合
    final_df = pd.merge(df, res_df, on='馬名', how='left')

    print(f"--- SCENE_Cast分析の並列処理を完了しました。 ---")

    return final_df

#====================================================
# SCENE_Cast分析の実行
#====================================================

if __name__ == "__main__":

    # 必要情報の収集
    Race_Info = f'{g.stadium} {g.clas} {g.td} {g.distance}m {g.race_name}'
    SCENE_Script_df = pd.read_sql('SELECT * FROM "SCENE_Script"', con=engine)
    SCENE_Cast_df = SCENE_Script_df

    # HTMLファイルから抽出した血統情報の格納
    blood_info_list = []
    for i in range(g.hr_num):
        blood_file = f'/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/Blood{(i+1):02d}.html'
        file_text, used_encoding = read_text_with_fallback(blood_file)
        cleaned_text = clean_html_for_analysis(file_text) 
        blood_info_list.append(cleaned_text)
    SCENE_Cast_df['血統情報'] = blood_info_list

    # 各馬のキャラ設定を実行
    SCENE_Cast_df = process_all_horses_parallel(SCENE_Cast_df, max_workers=8)

    # 結果をPostgreSQLに保存
    SCENE_Cast_df.to_sql('SCENE_Cast', con=engine, if_exists='replace', index=False)