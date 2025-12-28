#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# SCENE分析（Systematic Character Extraction for Narrative Epilogue）
#=============================================================

#====================================================
# SCENE分析の準備
#====================================================

# ライブラリの準備
import os
import time
import numpy as np
import re
from colorama import Fore, Back, Style
import pandas as pd
from itertools import combinations
from sqlalchemy import create_engine
from google import genai  # Gemini APIクライアント
from google.genai.errors import APIError # Gemini APIのエラー処理
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv 
import json
import asyncio
import edge_tts

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


#====================================================
# バックグラウンドシミュレーション関数
#====================================================

# 出走馬全頭が1着になる世界線を出走頭数回バックグラウンドでシミュレーションする（Gemini API使用）
def run_background_simulation_parallel(cast_df, ensemble_df, race_table_df, client, model, max_workers=4):
    race_info = f'{g.stadium} {g.clas} {g.td} {g.distance}m {g.race_name} ({g.cond})'
    merged_df = pd.merge(cast_df, race_table_df[['馬名', '毛色']], on='馬名')
    
    def build_compact_context(df):
        context = ""
        for _, row in df.iterrows():
            blood = row['血統分析'].split("血統分析：")[-1] if "血統分析：" in row['血統分析'] else row['血統分析']
            context += (f"【{row['番']}番 {row['馬名']}】脚質:{row['脚質']}, 指数:{row['先行指数']}, "
                        f"適合率:{row['レース条件適合率']}, 最終期待値:{row['最終期待値']}, 血統:{blood[:40]}...\n")
        return context

    base_cast_context = build_compact_context(merged_df)
    system_instruction = (
        f"あなたは「{race_info}」の勝負の行方を予測するシミュレーターです。"
        "提供されたデータに基づき、展開の紛れを含めた着順を算出してください。出力はJSONのみ。"
    )

    print(f"バックグラウンド・シミュレーション開始 (並列数:{max_workers})")

    # 1回分のシミュレーションを実行する関数（スレッド用）
    def process_target_horse(row):
        target_horse = row['馬名']
        relevant_rivalries = ensemble_df[
            (ensemble_df['馬名_A'] == target_horse) | (ensemble_df['馬名_B'] == target_horse)
        ]
        
        rivalry_context = ""
        if not relevant_rivalries.empty:
            for _, r_row in relevant_rivalries.iterrows():
                rivalry_context += f"■{r_row['ライバル関係']}\n"
        else:
            rivalry_context = "特筆すべきライバル関係なし"

        user_prompt = f"""
            レース：{race_info}
            今回のフォーカス馬（主役）：{target_horse}
            【出走馬能力データ】
            {base_cast_context}
            【主役馬に関わる重要な因縁・ライバル関係】
            {rivalry_context}
            指示：16頭の着順を以下のJSON形式で出力。
            ```json
            {{
            "target": "{target_horse}",
            "logic": "30文字以内の主要因",
            "rank": ["1着馬名", "2着馬名", ..., "16着馬名"]
            }}
            ```
        """
        
        try:
            # APIリクエスト
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config={"system_instruction": system_instruction}
            )
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"  エラー ({target_horse}): {e}")
        return None

    all_ranks = []
    # ThreadPoolExecutorによる並列実行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 各行に対して非同期タスクを生成
        futures = {executor.submit(process_target_horse, row): row['馬名'] for _, row in merged_df.iterrows()}
        
        for future in as_completed(futures):
            horse_name = futures[future]
            result = future.result()
            if result:
                all_ranks.append(result)
                print(f"  [完了] {horse_name} の世界線")

    print(Fore.YELLOW + f"バックグラウンド・シミュレーション完了 (計 {len(all_ranks)}件) " + Style.RESET_ALL)
    print('')
    return all_ranks


#====================================================
# バックグラウンドシミュレーションによる最終結果統合関数
#====================================================

def run_final_aggregation(all_ranks, cast_df):
    """
    出走頭数回のシミュレーション結果をクリーニング・統合・統計処理する
    """
    # 1. 正解の馬名リストを取得（表記揺れ判定用）
    # カラム名が '馬name' の場合は適宜修正してください
    valid_names = cast_df['馬名'].tolist()
    
    horse_ranks = {}
    total_sims = len(all_ranks)

    # 2. 表記揺れを正規化する内部関数
    def clean_horse_name(raw_name, valid_names):
        name = raw_name.strip()
        # 「1番 」「1番」などの番号付き表記を削除
        name = re.sub(r'^\d+番\s*', '', name)
        
        # 完全一致確認
        if name in valid_names:
            return name
        
        # 部分一致によるレスキュー（例：「シンエンペラー」が含まれていればOK）
        for valid_n in valid_names:
            if valid_n in name:
                return valid_n
        return name

    # 3. 各世界線の着順を集計
    for result in all_ranks:
        rank_list = result['rank']
        for i, raw_horse in enumerate(rank_list):
            horse = clean_horse_name(raw_horse, valid_names)
            rank = i + 1
            
            if horse not in horse_ranks:
                horse_ranks[horse] = []
            horse_ranks[horse].append(rank)

    # 4. 統計データの算出
    summary_list = []
    for horse, ranks in horse_ranks.items():
        # クリーニングできなかった不明な名前は除外
        if horse not in valid_names:
            continue
            
        avg_rank = sum(ranks) / len(ranks)
        win_count = len([r for r in ranks if r == 1])
        top2_count = len([r for r in ranks if r <= 2])
        top3_count = len([r for r in ranks if r <= 3])
        
        summary_list.append({
            "馬名": horse,
            "平均着順": round(avg_rank, 2),
            "勝率": win_count / total_sims,
            "連対率": top2_count / total_sims,
            "複勝率": top3_count / total_sims,
            "最高位": min(ranks),
            "最下位": max(ranks)
        })

    # 5. 基本統計DataFrameの作成
    df_stats = pd.DataFrame(summary_list)

    # 6. 枠番・馬番情報の統合
    # cast_dfから必要な列だけを抽出してマージ
    info_df = cast_df[['枠番', '番', '馬名']]
    final_df = pd.merge(info_df, df_stats, on='馬名', how='left')

    # 7. 表示用にフォーマット整形
    final_df['勝率'] = final_df['勝率'].apply(lambda x: f"{x*100:.1f}%")
    final_df['連対率'] = final_df['連対率'].apply(lambda x: f"{x*100:.1f}%")
    final_df['複勝率'] = final_df['複勝率'].apply(lambda x: f"{x*100:.1f}%")

    # 8. 平均着順でソート（期待値が高い順）
    final_df = final_df.sort_values("平均着順").reset_index(drop=True)
    
    return final_df


#====================================================
# 最終結果から馬印を付与する関数
#====================================================

def assign_race_marks_advanced(final_df, ensemble_df):
    """
    シミュレーション結果とライバル関係に基づき、戦略的に印を付与する
    """
    df = final_df.copy()
    # ％表記を計算用に数値に戻す
    df['win_rate'] = df['勝率'].str.rstrip('%').astype('float')
    
    # 必須条件：最高位が3着以内
    mask_eligible = df['最高位'] <= 3
    df['印'] = ""

    # 1. ◎ 本命 & 2. ○ 対抗 (平均着順上位)
    eligible_indices = df[mask_eligible].index
    if len(eligible_indices) >= 1:
        df.at[eligible_indices[0], '印'] = "◎ 本命"
    if len(eligible_indices) >= 2:
        df.at[eligible_indices[1], '印'] = "○ 対抗"

    # 3. ▲ 単穴: 1着率が最も高い（印なしの馬から）
    remaining = df[mask_eligible & (df['印'] == "")]
    if not remaining.empty:
        tan_ana_idx = remaining['win_rate'].idxmax()
        if remaining.loc[tan_ana_idx, 'win_rate'] > 0:
            df.at[tan_ana_idx, '印'] = "▲ 単穴"

    # 5. △　ドラマ: ライバル関係が存在する馬を優先
    # ensemble_dfの「ライバル関係」に含まれる馬名を抽出
    remaining = df[mask_eligible & (df['印'] == "")]
    rival_names = " ".join(ensemble_df['ライバル関係'].tolist())
    
    roman_idx = None
    for idx in remaining.index:
        if df.loc[idx, '馬名'] in rival_names:
            roman_idx = idx
            break
    
    if roman_idx is not None:
        df.at[roman_idx, '印'] = "△ ドラマ"
    elif not remaining.empty:
        # ライバルが見つからない場合は残りの最上位
        df.at[remaining.index[0], '印'] = "△ ドラマ"

    # 4. ★ロマン: 最高位が1着（一撃の可能性）
    remaining = df[mask_eligible & (df['印'] == "")]
    drama_candidates = remaining[remaining['最高位'] == 1]
    if not drama_candidates.empty:
        df.at[drama_candidates.index[0], '印'] = "★ ロマン"

    # 6. ☆ ドリーム: 条件を満たす中で、最も平均着順が低い（＝一番下にいる）馬
    remaining = df[mask_eligible & (df['印'] == "")]
    if not remaining.empty:
        # remainingは既に平均着順でソートされているので、最後の一頭
        dream_idx = remaining.index[-1]
        df.at[dream_idx, '印'] = "☆ ドリーム"

    return df.drop(columns=['win_rate'])


#====================================================
# ファイナル・ドラマ生成関数
#====================================================

# 最終結果の世界線をファイナル・ドラマとして再現する（Gemini API利用）
def generate_final_drama(cast_df, ensemble_df, final_report, g, client, model):
    race_info = f'{g.stadium} {g.clas} {g.td} {g.distance}m {g.race_name} ({g.cond})'
    
    # 1. 統計データ
    stats_summary = final_report[['枠番', '番', '馬名', '平均着順', '勝率', '最高位']].to_string(index=False)
    
    # 2. 戦術・能力データ (展開に直結する情報を追加)
    tactical_context = ""
    # 必要なカラムを抽出
    for _, row in cast_df.iterrows():
        tactical_context += (f"【{row['番']}番 {row['馬名']}】"
                             f"枠:{row['枠番']}, 脚質:{row['脚質']}, 先行指数:{row['先行指数']}, "
                             f"背景:{row['自己紹介'][:60]}...\n") # 背景は要約してトークン節約

    # 3. 因縁・ライバル
    rivalry_context = "\n".join(ensemble_df['ライバル関係'].tolist())

    system_instruction = f"""
        あなたは競馬の歴史を刻む偉大な作家、あるいは魂を揺さぶる実況アナウンサーです。
        提供された『統計データ』を運命の筋書きとし、『戦術データ』を展開の根拠として、{race_info}の物語を執筆してください。
        """

    user_prompt = f"""
        【重要：執筆および展開ルール】
        ・文章の中に数値（平均着順、勝率、指数など）は出さず、描写に変換してください。
        ・**先行指数が高い馬は序盤の主導権争いを描き、脚質（逃げ・先行・差し・追込）と枠番に応じた位置取りを正確に描写してください。**
        ・実況のテンポを意識し、冗長な表現を避けて、刻一刻と変わる情勢を熱く、しかし簡潔に描いてください。

        【構成】
        1. 以下の9シーンに分けて物語を描いてください。
        (1.ゲート前 2.スタート 3.1-2角 4.向正面 5.3角 6.4角 7.直線 8.ゴール 9.エピローグ)
        2. 前半の位置どりで、必ず全ての馬の名前が1回は登場するようにしてください。

        
        【資料：運命の設計図（統計）】
        {stats_summary}

        【資料：戦術・能力データ】
        {tactical_context}

        【資料：因縁とライバル】
        {rivalry_context}
        """

    print(f"{g.race_name}：ファイナル・ストーリー生成中...")
    
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.8,
        }
    )
    
    print(Fore.YELLOW + f"{g.race_name}：ファイナル・ストーリー生成完了" + Style.RESET_ALL)
    print('')
    
    return response.text



#====================================================
# 競馬実況風テキスト生成関数
#====================================================

# ファイナル・ドラマから、競馬実況風のテキストを生成（Gemeni API利用）
def generate_race_broadcast(final_story, final_report, g, client, model):
    horse_info = final_report[['番', '馬名']].to_string(index=False)
    race_header = f"{g.race_date} {g.stadium} {g.race_name} ({g.td}{g.distance}m / 馬場:{g.cond})"

    system_instruction = f"""
        あなたはラジオNIKKEIのベテラン実況アナウンサーです。
        情緒的な修飾語、比喩、馬の心理描写を**すべて排除**し、
        「どの馬がどこにいるか」という視覚情報のみを短文で繋いでください。
        """

    user_prompt = f"""
        【実況執筆ルール：スピードと正確性】
        1. **秒数制限**: 全体の読み上げ文字数を「1000文字以内」に抑え、現実のレース時間（{g.distance/20}秒程度）に収まる分量にしてください。
        2. **導入**: 「{race_header}。各馬ゲートイン完了、スタートしました！」と、2秒で始めてください。
        3. **隊列描写の徹底**:         
        - 序盤は「先頭は○番○○、2番手○番○○、内○番○○、外○番○○」という馬番と馬名を併用する形式を多用。
        - 逃げ、先行が多い場合、スタート直後の先行争いを詳しめに描写。
        - 中盤以降は「先頭は○○、2番手○○、内○○、外○○」と馬名のみの形式を多用。
        - 「〜が素晴らしい走り」「〜のドラマが」等の感想は一切不要。
        4. **後半（3角〜直線）の加速**: 
        - 1文を5文字〜15文字程度に短縮。
        - 最終直線は「○○上がってきた！」「内から○○！」「外は○○！」「○○か、○○か、○○、○○、並んでゴール！」といった感じで馬名を連呼しつつ叩きつける表現に。
        5. **入線後**: 1着を確定させ、2着、3着と思われる馬を伝え、レース展開によって「接戦」「際どい」「圧勝」「快勝」等の表現で簡潔に。

        【データ参照】
        出走馬: {horse_info}
        展開・結果: {final_story} (ここにある位置関係と結果を「位置情報」としてのみ抽出してください)
        """

    print(f"{g.race_name}：レース実況生成中...")
    
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.3, # 創作を抑え、事実描写に固定
        }
    )
    
    print(Fore.YELLOW + f"{g.race_name}：レース実況生成完了" + Style.RESET_ALL)
    print('')

    return response.text


#====================================================
# テキスト保存関数
#====================================================

def save_text_to_file(text, filename):
    """
    生成されたテキストを指定のパスに保存する
    """
    # ファイル書き込み (utf-8を指定して文字化けを防止)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return filename


#====================================================
# 実況音声ファイル生成関数
#====================================================

# 競馬実況テキストから実況音声を生成（Microsoft edge_tts利用）
async def save_race_audio(text, filename):

    print(f"{g.race_name}：レース実況音声ファイル生成中...")

    # --- 1. テキストの徹底クリーニング ---
    # 文末にある「〇〇文字」というメタ情報を削除
    text = re.sub(r'\d+文字.*$', '', text)
    
    # 読み間違い修正
    text = text.replace("m", "メートル").replace("M", "メートル")
    text = text.replace("先行馬", "せんこうば")
    text = text.replace("内", "うち").replace("外", "そと")
    text = text.replace("向こう正面", "むこうじょうめん")
    text = text.replace("前目", "まえめ")
    text = text.replace("大外", "おおそと")
    text = text.replace("最内", "さいうち")

    # 全体のクレンジング
    clean_text = re.sub(r'#+\s*\[?.*?\]?', '', text).replace('**', '').replace('---', '')

    # --- 2. ゴール後の「感動的な余韻」の演出 ---
    # 文末の「ゴール！」や「優勝です！」の後に空白を追加して、急に音声が切れないようにする
    clean_text = clean_text.rstrip() + " 。 。 。 。 " 

    # --- 3. 分割と生成 ---
    split_keywords = ["最終コーナー", "最後の直線", "最終直線", "残り200", "向いた", "直線コース"]
    parts = []
    for kw in split_keywords:
        if kw in clean_text:
            parts = clean_text.split(kw, 1)
            parts[1] = kw + parts[1]
            break
            
    # 音声の選択（現在は女性ボイス）
    voice_name = "ja-JP-NanamiNeural" 

    if len(parts) == 2:
        comm1 = edge_tts.Communicate(parts[0], voice_name, rate="+30%", pitch="+2Hz")
        # クライマックスはより高く、情熱的に
        comm2 = edge_tts.Communicate(parts[1], voice_name, rate="+40%", pitch="+12Hz", volume="+50%")
        
        await comm1.save("temp_1.mp3")
        await comm2.save("temp_2.mp3")
        
        ffmpeg_cmd = f'ffmpeg -i "concat:temp_1.mp3|temp_2.mp3" -acodec copy {filename} -y -loglevel quiet'
        os.system(ffmpeg_cmd)        

        if os.path.exists("temp_1.mp3"): os.remove("temp_1.mp3")
        if os.path.exists("temp_2.mp3"): os.remove("temp_2.mp3")
    else:
        comm = edge_tts.Communicate(clean_text, voice_name, rate="+30%", pitch="+5Hz")
        await comm.save(filename)

    print(Fore.YELLOW + f"{g.race_name}：レース実況音声ファイル生成完了" + Style.RESET_ALL)

    return filename


#====================================================
# SCENE分析実行
#====================================================

if __name__ == "__main__":

    # 生成データの保存先フォルダ
    save_dir_path = f'/Users/trueocean/Desktop/Python_Code/PRISM_SCENE/Media_files/'

    # データフレームの読み込み
    SCENE_Cast_df = pd.read_sql('SELECT * FROM "SCENE_Cast"', con=engine)
    SCENE_Ensemble_df = pd.read_sql('SELECT * FROM "SCENE_Ensemble"', con=engine)
    RaceTable_df = pd.read_sql('SELECT * FROM "RaceTable"', con=engine)

    # バックグラウンドシミュレーション実行
    all_ranks = run_background_simulation_parallel(SCENE_Cast_df, SCENE_Ensemble_df, RaceTable_df, client, MODEL)

    # 最終結果統合
    final_report = run_final_aggregation(all_ranks, SCENE_Cast_df)
    final_report.to_sql('FinalReport', con=engine, if_exists = 'replace', index=False)

    # 馬印の付与
    final_df_with_marks = assign_race_marks_advanced(final_report, SCENE_Ensemble_df)
    final_df_with_marks = final_df_with_marks[final_df_with_marks['印'] != ""][['印', '枠番', '番', '馬名']]
    final_df_with_marks.to_sql('FinalMark', con=engine, if_exists = 'replace', index=False)

    # ファイナル・ドラマ生成
    final_story = generate_final_drama(SCENE_Cast_df, SCENE_Ensemble_df, final_report, g, client, MODEL)
    save_drama_name = f'{save_dir_path}Final_Drama.txt'
    save_text_to_file(final_story, save_drama_name)

    # レース実況テキスト生成
    broadcast_script = generate_race_broadcast(final_story, final_report, g, client, MODEL)
    save_broadcast_name = f'{save_dir_path}Broadcast.txt'
    save_text_to_file(final_story, save_broadcast_name)

    # レース実況音声ファイル保存
    mp3_name = f"{save_dir_path}Broadcast.mp3"
    asyncio.run(save_race_audio(broadcast_script, mp3_name))

