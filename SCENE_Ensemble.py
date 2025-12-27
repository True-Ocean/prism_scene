#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# SCENE分析（Systematic Character Extraction for Narrative Epilogue）
# SCENE_Ensemble分析
#=============================================================

#====================================================
# SCENE_Ensemble分析の準備
#====================================================

# ライブラリの準備
import os
import numpy as np
import re
import pandas as pd
from itertools import combinations
from sqlalchemy import create_engine
from google import genai  # Gemini APIクライアント
from google.genai.errors import APIError # Gemini APIのエラー処理
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv 
import json

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
# Matchup_df作成関数
#====================================================

def create_matchup_df(df):
        
    Matchup_df = pd.merge(
        df,
        df,
        on='Race_Key',
        suffixes=('_A', '_B')
    )
    
    # 自分自身との対戦と、重複ペア（B vs A）を除外
    Matchup_df = Matchup_df[Matchup_df['馬名_A'] != Matchup_df['馬名_B']]
    Matchup_df = Matchup_df[Matchup_df['馬名_A'] < Matchup_df['馬名_B']].reset_index(drop=True)
    
    Matchup_df = Matchup_df[[
        'Race_Key', '日付_A', 'レース名_A', 'TD_A', '距離_A',
        '馬名_A', '年齢_A', 'Ave-3F_A', '上り3F_A', '走破タイム_A', '人気_A', '着順_A', 'PCI_A',
        '馬名_B', '年齢_B', 'Ave-3F_B', '上り3F_B', '走破タイム_B', '人気_B', '着順_B', 'PCI_B'
    ]].rename(columns={
        '日付_A': '日付',
        'レース名_A': 'レース名',
        'TD_A': 'TD',
        '距離_A': '距離',
        'RPCI_A': 'RPCI'
    })
    
    return Matchup_df


#====================================================
# 平均年齢に基づいたパラメータ k を設定する関数
#====================================================

def get_decay_parameter(age_a, age_b):
    avg_age = (age_a + age_b) / 2
    
    # 提案された設定
    if avg_age <= 3.5:
        # 若齢期は成長が速いため、過去の重みを強く減衰させる
        return 3.0 
    elif avg_age <= 5.5:
        # 完成期は中程度の減衰
        return 2.0
    else:
        # 古馬は経験も重要なので、減衰を緩やかにする
        return 1.0


#====================================================
# 各指標の (指標 * 重み) の合計を計算するカスタム集計関数
#====================================================

def weighted_sum(df, column):
    # NaNを0で埋める、あるいはその行を除外して重み付き合計を出す
    valid_mask = df[column].notna()
    return (df.loc[valid_mask, column] * df.loc[valid_mask, 'W_time']).sum()


#====================================================
# 時間軸トレンドスコアの計算関数
#====================================================

# 時間軸トレンドは、優劣指標（y）を時間（x）で回帰したときの「傾き」を評価
# ここでは、Order_Diff（着順差）を主要な優劣指標とする。
# 傾き > 0: 時間経過と共に Order_Diff が増加（馬Aが優位になっている）
# 傾き < 0: 時間経過と共に Order_Diff が減少（馬Bが優位になっている）

# 重み付き線形回帰を適用するための準備
# W_time を重みとして使用し、最近の対戦を重視

def calculate_weighted_trend(group):
    # 対戦回数が少ない場合はトレンドを計算しない（最低3回を推奨）
    if len(group) < 3:
        return np.nan
    
    x = group['Days_From_Start'] # 時間軸 (t)
    y = group['Order_Diff']      # 優劣指標
    w = group['W_time']          # 重み
    
    # 重み付き線形回帰の係数（傾きと切片）を計算
    # 傾き（b）は、以下の式で計算
    # b = Sum(w * (x - mean_x) * (y - mean_y)) / Sum(w * (x - mean_x)^2)
    
    mean_x = np.average(x, weights=w)
    mean_y = np.average(y, weights=w)
    
    numerator = np.sum(w * (x - mean_x) * (y - mean_y))
    denominator = np.sum(w * (x - mean_x)**2)
    
    # 分母がゼロの場合は計算不可
    if denominator == 0:
        return 0.0 # トレンドなし
        
    trend_slope = numerator / denominator
    return trend_slope


#====================================================
# ナラティブ抽出のためのプロンプトテンプレート
#====================================================

def create_json_analysis_prompt(horse_a_name, horse_b_name, data_json):
    """ライバル分析のためのシステム指示と、JSON出力を求めるユーザープロンプトを生成する"""
    structured_data = json.dumps(data_json, indent=2, ensure_ascii=False)
    
    # JSON出力のスキーマを定義
    json_schema = f"""
    {{
      "analysis_results": {{
        "conclusion_type": "string (例: 真のライバル | 成長ドラマ | 優劣固定)",
        "narrative_summary": "string (このライバル関係をドラマチックに要約するナラティブ。日本語で250文字～300文字程度)",
        "turning_point_race": "string (例: 2023-10-29 天皇賞秋G1)",
        "current_dominance": "string (現在の優位性: 例: {horse_a_name}が優位 | {horse_b_name}が優位 | 互角)",
        "dominance_reason": "string (current_dominanceの根拠となる直近の対戦事実や、これまでの総合的な優勢を、**一般人が理解できる平易な言葉で**記述)"
      }}
    }}
    """

    system_instruction = (
        "あなたは、競走馬のパフォーマンス分析の専門家です。入力データに基づき、ライバル関係を深く分析してください。\n"
        f"**【最重要】分析結果の記述においては、「馬A」「馬B」といった抽象的な表現や、「加重スコア」「トレンド係数」などの**\n"
        "**プログラム内部で用いられる専門用語は**【厳禁】**です。**\n"
        "**結果の根拠を説明する際は、「総合的な対戦成績」「直近の連勝記録」「僅差での優位性」など、一般人が理解できる平易な表現を使用してください。**\n"
        f"**必ず具体的な馬名（{horse_a_name} または {horse_b_name}）を使用してください。**\n"
        "**回答は、必ず指定されたJSONスキーマに厳密に従い、JSON形式の文字列のみを出力してください。**\n"
        "Markdownや他の解説テキストは一切含めないでください。"
    )

    user_prompt = f"""
    ライバル候補：{horse_a_name} (馬A) vs {horse_b_name} (馬B)
    
    以下のJSONデータ（対戦履歴とスコア）を分析してください。
    
    **入力JSONデータ:**
    ```json
    {structured_data}
    ```
    
    **出力指示:**
    上記のデータに基づき、以下のJSONスキーマに従って分析結果をJSON形式で出力してください。
    {json_schema}
    """
    
    return system_instruction, user_prompt


#====================================================
# 並列処理のためのAPI呼び出し関数
#====================================================

def analyze_single_pair(row, analysis_input, client, model):
    """単一のペアに対してAPIを呼び出し、分析結果を返す"""
    horse_a = row.馬名_A
    horse_b = row.馬名_B
    pair_key = f"{horse_a}_vs_{horse_b}"
    
    if pair_key not in analysis_input:
        return {'馬名_A': horse_a, '馬名_B': horse_b, 'Analysis_Status': 'データなし', 'Narrative_JSON': ''}

    data_for_api = analysis_input[pair_key]
    system_instruction, user_prompt = create_json_analysis_prompt(horse_a, horse_b, data_for_api)
    
    try:
        # ** Gemini APIを呼び出す **
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json"
            }
        )
        
        # JSON出力を期待
        narrative_json = response.text
        status = "成功"
        
        # 応答がJSONでない場合の簡易的なチェック
        if not narrative_json.strip().startswith('{'):
             status = f"JSON形式エラー: {narrative_json[:50]}..."

    except Exception as e:
        narrative_json = f"API呼び出しエラー: {e}"
        status = "失敗"

    return {
        '馬名_A': horse_a,
        '馬名_B': horse_b,
        'Analysis_Status': status,
        'Narrative_JSON': narrative_json
    }


#====================================================
# メインの並列実行ループ
#====================================================

def run_parallel_analysis(df, analysis_input, client, model, max_workers=5):
    """DataFrameの各行に対して並列にAPI分析を実行する"""
    analysis_results = []
    
    print(f"--- SCENE_Ensemble分析の並列処理を開始します: {len(df)} ペアを分析 (並列度: {max_workers}) ---")
    
    # ThreadPoolExecutorで並列処理を実行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        # 各ペアの分析タスクを投入
        future_to_pair = {
            executor.submit(analyze_single_pair, row, analysis_input, client, model): row 
            for row in df.itertuples()
        }
        
        # タスク完了を待機し、結果を収集
        for i, future in enumerate(as_completed(future_to_pair)):
            row = future_to_pair[future]
            print(f" [{i+1}/{len(df)}] 完了: {row.馬名_A} vs {row.馬名_B}")
            try:
                result = future.result()
                analysis_results.append(result)
            except Exception as e:
                # スレッド内での例外処理
                analysis_results.append({
                    '馬名_A': row.馬名_A, 
                    '馬名_B': row.馬名_B, 
                    'Analysis_Status': f'Executorエラー: {e}', 
                    'Narrative_JSON': ''
                })

    print(f"--- SCENE_Ensemble分析の並列処理を完了しました。 ---")

    return pd.DataFrame(analysis_results)


#====================================================
# JSONデータのパース関数
#====================================================

# JSONから展開するキーのリストを定義
new_columns = [
    'conclusion_type', 
    'narrative_summary', 
    'turning_point_race', 
    'current_dominance', 
    'dominance_reason'
]

def parse_narrative_json(json_str):
    """
    JSONパースを試行し、失敗した場合はすべてのカラムにエラー情報を返す。
    """    
    global new_columns 
    
    if pd.isna(json_str) or not isinstance(json_str, str):
        return {k: 'データ欠損' for k in new_columns}
    
    # クリーニング
    cleaned_str = re.sub(r'```json|```', '', json_str).strip()
    
    try:
        data = json.loads(cleaned_str)
        results = data.get('analysis_results', {})
        
        parsed_data = {}
        for key in new_columns:
            # キーが存在しない場合も、エラーにならないように 'N/A' を挿入
            parsed_data[key] = results.get(key, 'キー欠損')
            
        return parsed_data
        
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        # パースに失敗した場合、全ての期待されるカラムにエラー情報を返すことでKeyErrorを回避
        error_info = f"パース失敗: {e.__class__.__name__} - 元データ:{json_str[:50]}..."
        # エラー情報を 'narrative_summary' に格納し、他は NaN/エラー情報で埋める
        return {
            'conclusion_type': 'パースエラー', 
            'narrative_summary': error_info, 
            'turning_point_race': 'N/A', 
            'current_dominance': 'N/A', 
            'dominance_reason': 'N/A'
        }


#====================================================
# SCENE_Ensemble分析実行
#====================================================

def SCENE_Ensemble_Analysis():

    # ライバル関係抽出
    HorseRecords_df = pd.read_sql('SELECT * FROM "HorseRecords"', con=engine)
    SCENE_Cast_df = pd.read_sql('SELECT * FROM "SCENE_Cast"', con=engine)
    SCENE_Ensemble_df = SCENE_Cast_df

    # 日付とレース名の結合
    HorseRecords_df['Race_Key'] = HorseRecords_df['日付'].dt.strftime('%Y-%m-%d') + "_" + HorseRecords_df['レース名']

    # Matchup_dfの作成
    Matchup_df = create_matchup_df(HorseRecords_df)
    Matchup_df.to_sql('Matchup', con=engine, if_exists = 'replace', index=False)


    #====================================================
    # マッチアップのスコアリング（時間軸考慮）
    #====================================================

    # 1. 時間差（Time Delta）の計算
    # 現在の分析対象となる「基準日」を設定
    # 通常はデータセット内の最新の日付を基準とする。
    current_date = Matchup_df['日付'].max()

    # 基準日から各レース日までの「経過日数」を計算
    Matchup_df['Days_Since_Race'] = (current_date - Matchup_df['日付']).dt.days

    # 2. 時間差の正規化、kパラメータ適用、時間軸の重み計算
    # 経過日数を正規化し、[0, 1] の範囲に収める
    # 最も古い対戦が 1.0、最も新しい（基準日と同じ）対戦が 0.0 になるようにする
    max_days = Matchup_df['Days_Since_Race'].max()
    Matchup_df['Normalized_Time_Diff'] = Matchup_df['Days_Since_Race'] / max_days

    # 3. Matchup_dfに kパラメータを適用
    Matchup_df['k_decay'] = Matchup_df.apply(
        lambda row: get_decay_parameter(row['年齢_A'], row['年齢_B']),
        axis=1
    )

    # 4. 指数関数による時間軸の重み付けを計算
    Matchup_df['W_time'] = np.exp(-Matchup_df['k_decay'] * Matchup_df['Normalized_Time_Diff'])


    #====================================================
    # 定常的スコアのための指標計算
    #====================================================

    # 1. 勝敗（Win_A）と着順差（Order_Diff）の計算
    Matchup_df['Win_A'] = np.where(Matchup_df['着順_A'] < Matchup_df['着順_B'], 1, 0)
    Matchup_df['Order_Diff'] = Matchup_df['着順_B'] - Matchup_df['着順_A']

    # 2. 走破タイム差（Time_Diff）の計算
    # Time_Diff > 0 なら Aの走破タイムがBより速い
    Matchup_df['Time_Diff'] = Matchup_df['走破タイム_B'] - Matchup_df['走破タイム_A']

    # 3. Ave-3Fタイム差（Ave3F_Diff）の計算
    # Ave3F_Diff > 0 なら Aの前半平均速度相当タイムがBより速い
    Matchup_df['Ave3F_Diff'] = Matchup_df['Ave-3F_B'] - Matchup_df['Ave-3F_A']

    # 4. 上り3Fタイム差（Last3F_Diff）の計算
    # Last3F_Diff > 0 なら Aの上り3FタイムがBより速い
    Matchup_df['Last3F_Diff'] = Matchup_df['上り3F_B'] - Matchup_df['上り3F_A']


    #====================================================
    # 定常的ライバルスコアの集計
    #====================================================

    # 1. タイムパフォーマンス指標の統合 (Wtd_Time_Perfの準備)
    # 複数のタイム差指標（Time_Diff, Ave3F_Diff, Last3F_Diff）を統合するスコアを作成
    # ここでは、単純に平均を取ることで、全体、後半、前半のパフォーマンスを均等に評価
    Matchup_df['Combined_Time_Perf'] = (
        Matchup_df['Time_Diff'] + 
        Matchup_df['Ave3F_Diff'] + 
        Matchup_df['Last3F_Diff']
    ) / 3

    # 2. グループ化と重み付き集計
    # グループ化するキー
    group_keys = ['馬名_A', '馬名_B']

    # 時間軸重み（W_time）の合計を計算
    Total_Weight_df = Matchup_df.groupby(group_keys)['W_time'].sum().reset_index()
    Total_Weight_df = Total_Weight_df.rename(columns={'W_time': 'Total_Weight'})

    # 各指標の (指標 * 重み) の合計を計算するカスタム集計関数
    # def weighted_sum(df, column):
    #     return (df[column] * df['W_time']).sum()
    # # --- 修正案 ---
    def weighted_sum(df, column):
        # NaNを0で埋める、あるいはその行を除外して重み付き合計を出す
        valid_mask = df[column].notna()
        return (df.loc[valid_mask, column] * df.loc[valid_mask, 'W_time']).sum()

    # 重み付き集計
    Wtd_Sum_df = Matchup_df.groupby(group_keys).apply(
        lambda x: pd.Series({
            # 勝利数（Win_A）の重み付き合計
            'Sum_Wtd_Win_A': weighted_sum(x, 'Win_A'),
            # 着順差（Order_Diff）の重み付き合計
            'Sum_Wtd_Order_Diff': weighted_sum(x, 'Order_Diff'),
            # 統合タイムパフォーマンスの重み付き合計
            'Sum_Wtd_Time_Perf': weighted_sum(x, 'Combined_Time_Perf'),
            
            # 新規追加: ペアの人気度の合計（人気度の重み付けはしない）
            # 人気度は時間とともに変動する指標ですが、ここではシンプルに「対戦時の人気」の合計を計算します。
            'Sum_Popularity_A': x['人気_A'].sum(),
            'Sum_Popularity_B': x['人気_B'].sum(),

            # 総対戦回数 (今後のためにカウント)
            'Total_Matches': len(x)
        }),
            include_groups=False
    ).reset_index()

    # 3. Final_Pair_DFの作成（重み付き平均の計算）
    # 重み付き合計と総重みを結合
    Pair_df = pd.merge(Wtd_Sum_df, Total_Weight_df, on=group_keys)

    # 重み付き平均を計算
    Pair_df['Wtd_Win_Rate'] = Pair_df['Sum_Wtd_Win_A'] / Pair_df['Total_Weight']
    Pair_df['Wtd_Order_Diff'] = Pair_df['Sum_Wtd_Order_Diff'] / Pair_df['Total_Weight']
    Pair_df['Wtd_Time_Perf'] = Pair_df['Sum_Wtd_Time_Perf'] / Pair_df['Total_Weight']

    # 人気度の平均値（数値が小さいほど人気が高い）
    # Total_Matches * 2 は、馬Aと馬Bの人気の総数
    Pair_df['Average_Popularity_Score'] = (
        Pair_df['Sum_Popularity_A'] + Pair_df['Sum_Popularity_B']
    ) / (Pair_df['Total_Matches'] * 2) 

    # 最終スコアとして必要なカラムを選択 (Average_Popularity_Scoreを追加)
    Final_Pair_df = Pair_df[[
        '馬名_A', '馬名_B', 'Total_Matches', 'Total_Weight', 
        'Wtd_Win_Rate', 'Wtd_Order_Diff', 'Wtd_Time_Perf', 
        'Average_Popularity_Score' # <- 追加
    ]]


    #====================================================
    # 時間軸トレンドスコアの計算
    #====================================================

    # 1. 時間軸変数 'Days_From_Start' を作成
    # 最も古い日付を取得
    min_date = Matchup_df['日付'].min()

    # 最も古い日付からの経過日数（これが時間軸変数 t になる）
    Matchup_df['Days_From_Start'] = (Matchup_df['日付'] - min_date).dt.days

    # 2. 時間軸トレンドスコアの計算
    # グループ化して、トレンドスコア（傾き）を計算
    Trend_df = Matchup_df.groupby(['馬名_A', '馬名_B']).apply(
        lambda x: pd.Series({
            'Trend_Slope': calculate_weighted_trend(x)
        }),
        include_groups=False
    ).reset_index()

    # 3. Final_Pair_DFにトレンドスコアを結合
    Final_Pair_df = pd.merge(Final_Pair_df, Trend_df, on=['馬名_A', '馬名_B'], how='left')

    # NaN値（対戦回数 < 3 のペア）を 0.0 で埋める（トレンドなしと見なす）
    Final_Pair_df['Trend_Slope'] = Final_Pair_df['Trend_Slope'].fillna(0.0)


    #====================================================
    # 総合ライバルスコアの算出とフィルタリング
    #====================================================

    # 1. 接戦度（Closeness）の計算
    # 着順差の最大値（正規化のため）を計算。例: 1着〜18着なら最大17
    # ただし、データに存在する最大の着順差を使用するのがより現実的
    max_abs_order_diff = Final_Pair_df['Wtd_Order_Diff'].abs().max()

    # 正規化された着順差（0〜1）。小さいほど接戦。
    Final_Pair_df['Normalized_Closeness'] = Final_Pair_df['Wtd_Order_Diff'].abs() / max_abs_order_diff

    # 接戦度（Closeness Score）：差が小さいほど1に近づく
    Final_Pair_df['Closeness_Score'] = 1 - Final_Pair_df['Normalized_Closeness']

    # 2. 継続度（Consistency）の計算
    # 対戦総重み（Total_Weight）にLog関数を適用し、極端な値を丸める
    # log(1+x)を使うことで、Total_Weight=0の場合のエラーを防ぎ、0以上の値になるように調整
    Final_Pair_df['Consistency_Score'] = np.log1p(Final_Pair_df['Total_Weight']) 

    # 3. 総合ライバルスコア（Rivalry Score）の算出
    # 総合スコア = 接戦度 * 継続度
    Final_Pair_df['Rivalry_Score'] = (
        Final_Pair_df['Closeness_Score'] * Final_Pair_df['Consistency_Score']
    )

    # 4. フィルタリングとランキング
    # 総対戦回数が少ないペアは除外する（例：最低3回以上の対戦）
    MIN_MATCHES = 3 
    Filtered_Pair_df = Final_Pair_df[Final_Pair_df['Total_Matches'] >= MIN_MATCHES].copy()
    if len(Filtered_Pair_df) <= 3:
        Filtered_Pair_df = Final_Pair_df[Final_Pair_df['Total_Matches'] >= MIN_MATCHES - 1].copy()

    if len(Filtered_Pair_df) <= 5:
        Filtered_Pair_df = Final_Pair_df[Final_Pair_df['Total_Matches'] >= MIN_MATCHES - 2].copy()

    if len(Filtered_Pair_df) == 0:
        print('本レースにおいて、ライバル関係は確認されませんでした。')
        g.rival = 0
        os._exit(0)

    Filtered_Pair_df = Filtered_Pair_df.sort_values(by='Rivalry_Score', ascending=False).reset_index(drop=True)

    #====================================================
    # SCENE_Ensemble分析（ナラティブ抽出）
    #====================================================

    # Filtered_Pair_DFの上位Nペアを取得
    TOP_N = min(g.hr_num, len(Filtered_Pair_df))
    Top_Rival_Pairs = Filtered_Pair_df.head(TOP_N)

    # ライバル候補のペアリストを作成
    rival_pair_names = set()
    for index, row in Top_Rival_Pairs.iterrows():
        # 馬名がアルファベット順になっているため、タプルでセットに追加
        rival_pair_names.add(tuple(sorted((row['馬名_A'], row['馬名_B']))))

    # Top_Matchup_dfの作成
    # Matchup_dfから、Top_Rival_Pairsに含まれる対戦履歴のみを抽出
    def is_top_rival_pair(row):
        # 馬名の順序を気にせずペアが一致するか確認
        pair = tuple(sorted((row['馬名_A'], row['馬名_B'])))
        return pair in rival_pair_names

    # applyを使って、対象のペアのみをフィルタリング（データ量によっては時間がかかる可能性あり）
    Top_Matchup_df = Matchup_df[Matchup_df.apply(is_top_rival_pair, axis=1)].copy()

    # ペアごとのJSONデータ構造の作成
    # Gemini APIに渡すための最終的なデータ構造
    analysis_input = {}

    for horse_a, horse_b in rival_pair_names:
        # 常に馬名がアルファベット順になるようにフィルタリング
        pair_matchup = Top_Matchup_df[
            (Top_Matchup_df['馬名_A'] == horse_a) & (Top_Matchup_df['馬名_B'] == horse_b)
        ].sort_values(by='日付')
        
        if pair_matchup.empty:
            # A, Bの定義が逆の場合を考慮し、再度フィルタリング
            pair_matchup = Top_Matchup_df[
                (Top_Matchup_df['馬名_A'] == horse_b) & (Top_Matchup_df['馬名_B'] == horse_a)
            ].sort_values(by='日付')
            
            # この場合、AとBの指標を入れ替える必要があるが、ここでは処理を簡略化するため、
            # Matchup_DFの作成時に順序を厳密に定義している前提で進めます。
            # 実際の実装では、ここでAとBの指標を入れ替える処理が必要です。
            
            # 今回のコードでは、Matchup_DF作成時に (A < B) の順序を保証しているため、最初のフィルタリングで全て取得されます。
            pass 
            
        # ペアごとの詳細な対戦履歴を作成
        race_details = []
        for index, row in pair_matchup.iterrows():
            race_details.append({
                "date": row['日付'].strftime('%Y-%m-%d'),
                "race_name": row['レース名'],
                "track": row['TD'],
                "distance": row['距離'],
                "A_order": row['着順_A'],
                "B_order": row['着順_B'],
                "order_diff": row['Order_Diff'],
                "time_diff": row['Time_Diff'],
                "A_age": row['年齢_A'],
                "B_age": row['年齢_B']
            })
        
        # Final_Pair_dfから総合スコアとトレンドを取得
        score_data = Filtered_Pair_df[
            (Filtered_Pair_df['馬名_A'] == horse_a) & (Filtered_Pair_df['馬名_B'] == horse_b)
        ].iloc[0]
        
        # 最終的な入力構造
        analysis_input[f"{horse_a}_vs_{horse_b}"] = {
            "summary_scores": {
                "total_matches": score_data['Total_Matches'],
                "rivalry_score": score_data['Rivalry_Score'],
                "wtd_win_rate_A": score_data['Wtd_Win_Rate'],
                "wtd_order_diff_A": score_data['Wtd_Order_Diff'],
                "trend_slope_A": score_data['Trend_Slope'] # 馬Aが優位になる傾向
            },
            "race_history": race_details
        }


    #====================================================
    # メインのAPI実行
    #====================================================

    # 並列実行 (例: 5並列)
    MAX_WORKERS = 5
    Analysis_Result_DF_Parallel = run_parallel_analysis(
        Filtered_Pair_df,
        analysis_input,
        client,
        MODEL,
        max_workers=MAX_WORKERS
    )

    # 結果の整理と結合
    # Final_Rival_Analysis_DFを作成 (定量スコア + 定性分析結果)
    Final_Rival_Analysis_df = pd.merge(
        Filtered_Pair_df,
        Analysis_Result_DF_Parallel.drop(columns=['Analysis_Status']), 
        on=['馬名_A', '馬名_B'],
        how='left'
    )


    #====================================================
    # 最終ライバル候補トップリストの提示
    #====================================================

    # 最終リスト作成コードの再実行（構造の強化）
    # JSONデータをDFに展開
    # 戻り値は辞書のシリーズ
    Narrative_Series = Final_Rival_Analysis_df['Narrative_JSON'].apply(parse_narrative_json)

    # 辞書のシリーズを明示的にDataFrameに変換
    Narrative_Expanded_df = pd.DataFrame(Narrative_Series.tolist(), index=Final_Rival_Analysis_df.index)

    # 最終的な統合DFを作成
    Final_List_df = pd.concat([
        # ★★★ 修正箇所: Average_Popularity_Score を追加 ★★★
        Final_Rival_Analysis_df[['馬名_A', '馬名_B', 'Rivalry_Score', 'Total_Matches', 'Wtd_Order_Diff', 'Average_Popularity_Score']],
        Narrative_Expanded_df
    ], axis=1).sort_values(by='Rivalry_Score', ascending=False).reset_index(drop=True)

    # 「人気度優先」でDFを並べ替えてトップ10を抽出
    # abs(Wtd_Order_Diff) をソート用に計算
    Final_List_df['Abs_Order_Diff'] = Final_List_df['Wtd_Order_Diff'].abs()

    # 人気度優先ソート基準を適用:
    # 1. Average_Popularity_Score (昇順, 数値が小さい＝人気が高い)
    # 2. Abs_Order_Diff (昇順, 互角なほど上位)
    # 3. Rivalry_Score (降順, ドラマ性が高いほど上位)
    Final_List_df_Sorted_Popular = Final_List_df.sort_values(
        by=['Average_Popularity_Score', 'Abs_Order_Diff', 'Rivalry_Score'],
        ascending=[True, True, False] # True=昇順 (人気/互角), False=降順 (ドラマ性)
    ).reset_index(drop=True)

    TOP_N_RIVALS = 10 # 件数は10を維持
    Top_Rival_List_df = Final_List_df_Sorted_Popular.head(TOP_N_RIVALS).copy()

    narrative_list = []

    for i, row in Top_Rival_List_df.iterrows():
        # テンプレートを定義（インデントは無視されるように .replace や .strip を活用）
        template = (
            f"【ライバル注目度: 第{i+1}位：{row['馬名_A']} vs {row['馬名_B']}】\n"
            f"関係性: {row['conclusion_type']}\n"
            f"優位性: {row['current_dominance']}（{row['dominance_reason']}）\n"
            f"転機: {row['turning_point_race']}\n"
            f"総評: {row['narrative_summary']}"
        )
        narrative_list.append(template)

    Top_Rival_List_df['ライバル関係'] = narrative_list

    Top_Rival_List_df.to_sql('SCENE_Ensemble', con=engine, if_exists = 'replace', index=False)


if __name__ == "__main__":

    SCENE_Ensemble_Analysis()