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
    g.hr_num = len(race_table_df)
    merged_df = pd.merge(cast_df, race_table_df[['馬名', '毛色']], on='馬名')
    
    def build_compact_context(df):
        context = ""
        for _, row in df.iterrows():
            blood = row['血統分析'].split("血統分析：")[-1] if "血統分析：" in row['血統分析'] else row['血統分析']
            context += (f"【{row['番']}番 {row['馬名']}】先行指数:{row['先行指数']}, 脚質:{row['脚質']}, 実力のムラ:{row['実力のムラ']}, "
                        f"適合率:{row['レース条件適合率']}, 最終期待値:{row['最終期待値']}, 血統:{blood[:40]}...\n")
        return context

    base_cast_context = build_compact_context(merged_df)
    system_instruction = (
        f"あなたは「{race_info}」の勝負の行方を予測するシミュレーターです。"
        "提供されたデータに基づき、展開の紛れを含めた着順を算出してください。出力はJSONのみ。"
    )

    print(f"バックグラウンド・シミュレーション開始 (同時実行数:{max_workers})")

    # 1回分のシミュレーションを実行する関数（スレッド用）
    def process_target_horse(row):
        target_horse = row['馬名']
        temp_cast_context = ""
    
        for _, r in merged_df.iterrows():
            raw_mura = r['実力のムラ']
            
            # --- 解決策：スケーリング補正 ---
            # 1. 最小保証値（Base）を設定。ここでは例として 2.0 とします。
            # 2. 元のムラが小さいほど Base に近づけ、大きい場合は元の値を優先する
            # 下記の計算により、ムラ0の馬は2.0になり、ムラ1の馬は2.4程度、
            # ムラ3以上の馬は元の値をそのまま活かす、といった調整が可能です。
            
            if raw_mura < 3.0:
                # 3.0未満の馬を 2.0〜3.0 の範囲に押し上げる計算式
                # (3.0 - raw_mura) / 3.0 は 0〜1 の係数になる
                adjusted_mura = 2.0 + (raw_mura * (1.0 / 3.0))
            else:
                adjusted_mura = raw_mura

            # 主役の馬かどうかで条件分岐
            if r['馬名'] == target_horse:
                condition_label = "絶好調"
                # 主役はムラを最大限に活かす（期待値に下駄を履かせる）
                multiplier = 1.2
                # さらに、格上が相手でも「紛れ」を起こすための固定値（下駄）
                hero_bonus = 1.0
            else:
                # 他の馬は通常通りランダムにデキを決定
                hero_bonus = 0
                condition_z = np.random.normal(0, 1)
                if condition_z > 1.5:
                    condition_label = "絶好調"
                    multiplier = 1.0
                elif condition_z > 0.5:
                    condition_label = "好調"
                    multiplier = 0.5
                elif condition_z > -0.5:
                    condition_label = "普通"
                    multiplier = 0.0
                elif condition_z > -1.5:
                    condition_label = "不調"
                    multiplier = -0.5
                else:
                    condition_label = "絶不調"
                    multiplier = -1.0

            # 3. 最終期待値を計算 (実力のムラ × 倍率)
            # ※ multiplierが0(普通)でも、微小なランダム値(0.12等)を加えるとよりリアルです
            micro_fluctuation = np.random.uniform(-0.1, 0.1)
            simulated_value = round(r['最終期待値'] + (adjusted_mura * multiplier) + hero_bonus + micro_fluctuation, 2)
            
            blood = r['血統分析'].split("血統分析：")[-1] if "血統分析：" in r['血統分析'] else r['血統分析']
            temp_cast_context += (f"【{r['番']}番 {r['馬名']}】先行指数:{row['先行指数']}, 脚質:{r['脚質']}, "
                                f"今日の状態:{condition_label}, シミュレーション期待値:{simulated_value}, 血統:{blood[:40]}...\n")

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
            【出走馬能力データ（今回の世界線のコンディション）】
            {temp_cast_context}
            【主役馬に関わる重要な因縁・ライバル関係】
            {rivalry_context}

            指示：全{g.hr_num}頭の着順を以下のJSON形式で出力。

            ```json
            {{
            "target": "{target_horse}",
            "logic": "30文字以内の主要因",
            "rank": ["1着馬名", "2着馬名", ..., "{g.hr_num}着馬名"]
            }}
            ```
        """
        
        for attempt in range(3):
            try:
                # APIリクエスト
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 1.0,  # ここで調整。デフォルトは通常1.0前後
                        "top_p": 0.95,       # 累積確率に基づく制限（あわせて調整が推奨されます）
                    }
                )
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                break

            except Exception as e:
                if "503" in str(e) or "overloaded" in str(e):
                    print(f"  再試行中... ({target_horse}) {attempt+1}回目")
                    time.sleep(2 + attempt * 2) # 徐々に待ち時間を増やす
                else:
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
                print(f" [完了] {horse_name} の世界線")

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


import pandas as pd
#====================================================
# 最終結果から馬印を付与する関数（修正完了版）
#====================================================

def assign_race_marks_advanced(final_df, ensemble_df):
    # 1. 【重要】インデックスを0からの連番にリセット
    # これにより、.at や .loc での指定ズレを物理的に防ぎます
    df = final_df.copy().reset_index(drop=True)
    
    # 2. データ型の変換（安全策）
    # 文字列のまま比較してしまうリスクを排除するため、確実に数値化します
    df['win_rate'] = df['勝率'].str.replace('%', '', regex=False).replace('-', '0').replace('', '0').astype(float)
    df['rentai_rate'] = df['連対率'].str.replace('%', '', regex=False).replace('-', '0').replace('', '0').astype(float)
    df['fukusho_rate'] = df['複勝率'].str.replace('%', '', regex=False).replace('-', '0').replace('', '0').astype(float)
    
    # 最高位も念のため数値化（エラー値は99へ）
    import pandas as pd
    df['highest_rank'] = pd.to_numeric(df['最高位'], errors='coerce').fillna(99).astype(int)

    # 必須条件：最高位が3着以内
    mask_eligible = df['highest_rank'] <= 3
    df['印'] = ""

    # ------------------------------------------------
    # 1. ◎ 本命 & 2. ○ 対抗
    # ------------------------------------------------
    eligible_indices = df[mask_eligible].index
    if len(eligible_indices) >= 1:
        df.at[eligible_indices[0], '印'] = "◎ 本命"
    if len(eligible_indices) >= 2:
        df.at[eligible_indices[1], '印'] = "○ 対抗"

    # ------------------------------------------------
    # 3. ▲ 単穴: 連対率重視
    # ------------------------------------------------
    remaining = df[mask_eligible & (df['印'] == "")]
    if not remaining.empty:
        max_rentai_rate = remaining['rentai_rate'].max()
        if max_rentai_rate > 0:
            # 連対率最大の馬
            tan_ana_idx = remaining[remaining['rentai_rate'] == max_rentai_rate].index[0]
            df.at[tan_ana_idx, '印'] = "▲ 単穴"
        else:
            # 連対率0なら複勝率
            # idxmax()でインデックスを直接取得
            tan_ana_idx = remaining['fukusho_rate'].idxmax()
            df.at[tan_ana_idx, '印'] = "▲ 単穴"

    # ------------------------------------------------
    # 4. △ ドラマ: ライバル関係
    # ------------------------------------------------
    remaining = df[mask_eligible & (df['印'] == "")]
    if not remaining.empty: # 空チェック追加
        rival_set = set()
        if 'ライバル関係' in ensemble_df.columns:
            for val in ensemble_df['ライバル関係'].dropna():
                rival_set.update(str(val).replace('、', ' ').split())
        
        drama_idx = None
        for idx in remaining.index:
            if df.at[idx, '馬名'] in rival_set:
                drama_idx = idx
                break
        
        if drama_idx is not None:
            df.at[drama_idx, '印'] = "△ ドラマ"
        else:
            # ライバル不在なら残りの最上位
            df.at[remaining.index[0], '印'] = "△ ドラマ"

    # ------------------------------------------------
    # 5. ★ ロマン: 馬券に絡む1頭
    # ------------------------------------------------
    remaining = df[mask_eligible & (df['印'] == "")]
    
    if not remaining.empty:
        # 最高位が3着の馬を探す
        top_candidates = remaining[remaining['highest_rank'] == 3]
        
        target_idx = None
        
        if not top_candidates.empty:
            # 最高位3着がいればその先頭
            target_idx = top_candidates.index[0]
        else:
            # いなければ複勝率トップ
            target_idx = remaining['fukusho_rate'].idxmax()
            
        # 【重要】他と同じく .at を使用して書き込む
        if target_idx is not None:
            df.at[target_idx, '印'] = "★ ロマン"

    # ------------------------------------------------
    # 6. ☆ ドリーム: 大穴候補
    # ------------------------------------------------
    remaining = df[mask_eligible & (df['印'] == "")]
    if not remaining.empty:
        # 最後尾（平均着順が最も低い）
        df.at[remaining.index[-1], '印'] = "☆ ドリーム"


    # ----------------------------------------------------
    # 並べ替えと出力
    # ----------------------------------------------------
    sort_order = ["◎ 本命", "○ 対抗", "▲ 単穴", "△ ドラマ", "★ ロマン", "☆ ドリーム"]
    sort_map = {label: i for i, label in enumerate(sort_order)}
    df['sort_key'] = df['印'].map(sort_map)

    df = df.sort_values(by=['sort_key'], na_position='last')

    # 計算用の一時カラムを削除して返す
    return df.drop(columns=['win_rate', 'fukusho_rate', 'highest_rank', 'sort_key'])


#====================================================
# ファイナル・ドラマ生成関数
#====================================================

# 最終結果の世界線をファイナル・ドラマとして再現する（Gemini API利用）
def generate_final_drama(cast_df, ensemble_df, final_report, final_mark, client, model):
    race_info = f'{g.stadium}競馬場 {g.clas} {g.td} {g.distance}m {g.race_name} ({g.cond})'
    scene_instruction = "、".join(g.selected_scenes)
    
    # 1. 前提データ
    # 最終シミュレーション結果のまとめ
    stats_summary = final_report[['枠番', '番', '馬名', '平均着順', '勝率', '最高位']].to_string(index=False)
    # 最終馬印
    mark_summary = final_mark[['印', '枠番', '番', '馬名']].to_string(index=False)

    
    # 2. 戦術・能力データ (展開に直結する情報を追加)
    tactical_context = ""
    # 必要なカラムを抽出
    for _, row in cast_df.iterrows():
        tactical_context += (f"【{row['番']}番 {row['馬名']}】"
                             f"枠:{row['枠番']}, 脚質:{row['脚質']}, 先行指数:{row['先行指数']}, "
                             f"実力のムラ:{row['実力のムラ']}, 中何週：{row['中何週']}週, 成長度：{row['調教成長ポイント']} ")

    # 3. キャラデータ (心情やセリフに関係する情報を追加)
    character_context = ""
    # 必要なカラムを抽出
    for _, row in cast_df.iterrows():
        character_context += (f"【{row['番']}番 {row['馬名']}】"
                             f"家柄:{row['血統分析']}, キャラ:{row['キャラ設定']},  勝負服色:{row['勝負服色']}"
                             f"背景:{row['自己紹介'][:60]}...\n") # 背景は要約してトークン節約

    # 4. ライバル関係
    rivalry_context = "\n".join(ensemble_df['ライバル関係'].tolist())

    # システム指示：作家性を定義し、データの「読み替え」を命じる
    system_instruction = f"""
        あなたは競馬の歴史を血と汗で綴る孤高の劇作家です。
        提供された資料を、数値の羅列ではなく「魂の設計図」として捉え、{race_info}を舞台にした重厚な短編小説を執筆してください。
        実況形式やスペック紹介は一切不要です。群像劇でありながら、特定の因縁に焦点を当てた、緩急のある物語を創出してください。
        """

    # ユーザープロンプト：ルールと構造の徹底
    user_prompt = f"""
        【重要：執筆および展開ルール】

        ■ 表記・呼称の厳格ルール（最優先・厳守）
        ・数字の表記： 数字はすべて「1、2、3」などの【アラビア数字】を使用してください。漢数字（一、二、三）は使用禁止です。
        ・名乗りの完全禁止： セリフを「僕は〇〇」「私こそが～」といった挨拶や自己紹介から始めることを【厳禁】とします。名前は地の文（情景描写）で示し、独り言や呟きはいきなり感情や感覚から書き始めてください。
        ・馬の呼び方： 序盤（シーン1）のみ「1番ジャスティンパレス」の形式。以降は数字を完全に排し「馬名のみ」で記述してください。
        ・情報の昇華： 指数や勝率、人気などの数値は、コース取り、手応え、立ち込める殺気などの情景描写に完全に変換してください。

        ■ 構造：血統と因縁の物語
        1. 内なる独り言、呟きの書式：【キャラデータ】を反映
        その馬の血統を背景とする心の声や、ライバルに対する熱き胸の内を、精神に潜り込むようなセリフにしてください。
        （例）「今回こそリベンジを果たす！ハーツクライから受け継いだこの熱き叫びを聞くがいい！」
        2. 外なる叫び、会話の書式：【ライバル関係】を反映
        ライバル同士の熱きセリフの応酬を描写してください。
        （例）マスカレードボールがクロワデュノールに迫る。「そのポジションは僕のものだ！」クロワデュノールも応戦する。「上等だ！来るなら来い！」
        3. 描写の濃淡（重要）:
        18頭全員のセリフを順番に書く「点呼描写」を【厳禁】とします。
        【ライバル関係】にある馬同士や【注目キャラ】の心理を深く、長く掘り下げる一方で、他の馬は地の文での位置取り描写のみに留めてください。これにより物語に「主人公」を作ってください。

        ■ 文体とクオリティ
        ・五感： 蹄音、泥跳ね、風の音、そして数万人の観客が発する地鳴りのような熱狂を織り交ぜてください。
        ・文体： 解説者ではなく「劇作家」の視点を徹底してください。情景描写は重厚に。加速局面では「体言止め」を多用し、読者の鼓動を早めるテンポを作り出してください。

        
        【構成とシーン別指示】
        レースのシーンは{scene_instruction}のみとし、概ね以下の5部構成で執筆してください。
        【注目キャラ】を念頭に置いた着順となるよう、ストーリを構築してください。
        
        ### 1. プロローグ：（レース前の競馬場の描写、緊張感、ゲートインの状況）
        静寂の中に響く蹄の音。観客席のざわめき。馬たちの息遣い。レース前の緊張感を、五感を駆使して描写してください。

        ### 2. 序盤：（スタート、先行争い）
        ゲート開放の衝撃から描写。先団・中団・後方という「馬群の動き」の中で自然に全馬を一度登場させてください。一頭ずつ順番に紹介するリスト形式は【厳禁】です。その流れの中で、主要な数頭の「血の昂ぶり」を独白として挿入します。

        ### 3. 中盤：（第3コーナーまで）
        心理戦の極地。ライバルの息遣い、視線の交錯を描写。【ライバル関係】にある馬同士の、標的を射抜くような鋭い思考や感情をぶつけ合わせてください。

        ### 4. クライマックス：（第4コーナー〜最終直線） 
        限界を超えた叩き合い。観客の絶叫。体言止めと馬名の連呼、そして【キャラデータ】を反映した「剥き出しの叫び」のみで、ゴールへ向かう狂熱を畳み掛けてください。

        ### 5. エピローグ：（入線後） 
        【運命の設計図】が導き出した結末。勝者の光と敗者の誇り。受け継いだ血が次にどこへ向かうのか、短編小説に相応しい叙情的な余韻で締めくくってください。

        
        【資料：運命の設計図（統計）】
        {stats_summary}
        【資料：戦術・能力データ】
        {tactical_context}
        【資料：キャラデータ】
        {character_context}
        【資料：ライバル関係】
        {rivalry_context}
        【資料：注目キャラ】
        {mark_summary}
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
def generate_race_broadcast(final_story, final_report, client, model):
    horse_info = final_report[['番', '馬名']].to_string(index=False)
    if g.cond == '稍':
        g.cond = '稍重'
    elif g.cond == '不':
        g.cond = '不良'

    race_header = f"{g.race_date} {g.stadium}競馬場 {g.race_name} ({g.td}{g.distance}m / 馬場:{g.cond})"

    # generate_race_broadcast の system_instruction を強化
    system_instruction = f"""
    あなたはラジオNIKKEIのベテラン実況アナウンサーです。
    【厳禁事項】
    ・馬のセリフ、内面の声（「」内の文章）は1文字たりとも出力しないでください。
    ・「〜を燃やす」「〜の決意」といった情緒的な描写はすべてカットしてください。
    ・「スタート！」「向正面」といった見出し、および ** による強調は不要です。
    【推奨事項】
    ・「先頭は〇〇、2番手〇〇」という事実のみを、句点で区切って短文で並べてください。
    """

    user_prompt = f"""
        【実況執筆ルール：スピードと正確性】
        1. **秒数制限**: 全体の読み上げ文字数を「1000文字以内」に抑え、現実のレース時間（{g.distance/20}秒程度）に収まる分量にしてください。
        2. **導入**: 「{race_header}。各馬ゲートイン完了、スタートしました！」と、2秒で始めてください。
        3. **隊列描写の徹底**:         
        - 序盤は「先頭は〇番〇〇、2番手〇番〇〇、内〇番〇〇、外〇番〇〇」という番と馬名を併用する形式（例：7番ディープインパクト）を多用。ただし、枠番は使わないこと。
        - 逃げ、先行が多い場合、スタート直後の先行争いを詳しめに描写。
        - 隊列が落ち着いたら、改めて全馬の名前と位置どりを、前から順に簡潔に説明していくこと。場合に応じて何馬身離れているかも補足すること。
        （例）「先頭は〇〇、1馬身離れて〇〇、その内〇〇、半馬身差で〇〇、・・・2馬身離れて〇〇、最後方に〇〇という隊列」
        - 「〜が素晴らしい走り」「〜のドラマが」等の感想は一切不要。
        4. **後半（4角〜最終直線）の加速**: 
        - 1文を5文字〜15文字程度に短縮。
        - 最終直線では、先頭争いに加わる馬を最大でも5頭までに絞り、馬名と位置取りのみとすること。
        【禁止】最終直線の間は、絶対に語尾を「ですます」調にしないこと。
        （例）「〇〇上がってきた！」「内から〇〇！」「外は〇〇！」「馬群の中から一気に〇〇！」「〇〇先頭！」「大外から〇〇きた！」「〇〇か、〇〇か、〇〇！〇〇！」
        5. **入線後**:
        - 1着、2着、3着と思われる馬を伝え、レース展開によって「接戦」「際どい」「圧勝」「快勝」「余裕」等の表現で簡潔に。
        - 4着以下は不要です。
        - 最後に、「確定までしばらくお待ちください。」で締めてください。

        【データ参照】
        出走馬: {horse_info}
        展開・結果: {final_story} (ここにある位置関係と結果のみを忠実に抽出してください。)
        """

    print(f"{g.race_name}：レース実況テキスト生成中...")
    
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.3, # 創作を抑え、事実描写に固定
        }
    )
    
    print(Fore.YELLOW + f"{g.race_name}：レース実況テキスト生成完了" + Style.RESET_ALL)
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

    # --- 1. テキストの徹底クリーニング ---
    # 文末にある「〇〇文字」というメタ情報を削除
    text = re.sub(r'\d+文字.*$', '', text)
    
    # 読み間違い修正用の辞書（メンテナンス性を高める）
    replace_dict = {
        "コーナー": "コオナー",
        "m": "メートル",
        "M": "メートル",
        "弾ける": "はじける",
        "重賞": "じゅうしょう",
        "先行馬": "せんこうば",
        "最後方": "さいこうほう",
        "向正面": "むこうじょうめん",
        "向こう正面": "むこうじょうめん",
        "数馬身": "すうばしん",
        "前目": "まえめ",
        "外々": "そとそと",
        "大外": "おおそと",
        "最内": "さいうち",
        "末脚": "すえあし",
        "好スタート": "こうスタート", 
        "脚色": "あしいろ", 
        "先行勢": "せんこうぜい",
        "先行": "せんこう",
        "追込": "追い込み",
        "内": "うち",
        "外": "そと",
        "間": "あいだ",
        "急坂": "急ざか",
        "18番": "18ばん",
        "馬群": "ばぐん",
        "重": "おも",
        "稍重": "ややおも",
        "ゴール板": "ゴールばん",
        "S": "ステークス",
        "C": "カップ",
        "T": "トロフィー"
    }

    # 一括置換の実行
    for old, new in replace_dict.items():
        text = text.replace(old, new)

    # 全体のクレンジング
    clean_text = re.sub(r'#+\s*\[?.*?\]?', '', text).replace('**', '').replace('---', '')

    # --- 2. ゴール後の「感動的な余韻」の演出 ---
    # 文末の「ゴール！」や「優勝です！」の後に空白を追加して、急に音声が切れないようにする
    clean_text = clean_text.rstrip() + " 。 。 。 。 " 

    # --- 3. 分割と生成 ---
    split_keywords = ["最終コーナー", "第4コーナー", "最後の直線", "最終直線", "残り200", "残り100"]
    parts = []
    for kw in split_keywords:
        if kw in clean_text:
            parts = clean_text.split(kw, 1)
            parts[1] = kw + parts[1]
            break
            
    # 音声の選択（現在は女性ボイス）
    voice_name = "ja-JP-NanamiNeural" 
    # # 男性ボイス（Keita）に変更
    # voice_name = "ja-JP-KeitaNeural"

    if len(parts) == 2:
        print(f"{g.race_name}：レース実況オーディオ生成中...")
        print('')
        comm1 = edge_tts.Communicate(parts[0], voice_name, rate="+35%", pitch="+2Hz")
        # クライマックスはより高く、情熱的に
        comm2 = edge_tts.Communicate(parts[1], voice_name, rate="+40%", pitch="+8Hz", volume="+20%")
        
        await comm1.save("temp_1.mp3")
        await comm2.save("temp_2.mp3")
        
        ffmpeg_cmd = f'ffmpeg -i "concat:temp_1.mp3|temp_2.mp3" -acodec copy {filename} -y -loglevel quiet'
        os.system(ffmpeg_cmd)        

        if os.path.exists("temp_1.mp3"): os.remove("temp_1.mp3")
        if os.path.exists("temp_2.mp3"): os.remove("temp_2.mp3")
    else:
        print(f"{g.race_name}：レース実況オーディオ生成中...")
        comm = edge_tts.Communicate(clean_text, voice_name, rate="+30%", pitch="+5Hz")
        await comm.save(filename)

    print(Fore.YELLOW + f"{g.race_name}：レース実況オーディオ生成完了" + Style.RESET_ALL)

    return clean_text


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
    final_story = generate_final_drama(SCENE_Cast_df, SCENE_Ensemble_df, final_report, final_df_with_marks, client, MODEL)
    save_drama_name = f'{save_dir_path}Final_drama.txt'
    save_text_to_file(final_story, save_drama_name)

    # レース実況テキスト生成（クレンジング前）
    broadcast_script_draft = generate_race_broadcast(final_story, final_report, g, client, MODEL)

    # 最終レース実況テキスト・音声の生成・保存
    mp3_name = f"{save_dir_path}Broadcast.mp3"
    broadcast_name = f'{save_dir_path}Broadcast.txt'
    broadcast_script = asyncio.run(save_race_audio(broadcast_script_draft, mp3_name))
    save_text_to_file(broadcast_script, broadcast_name)

