#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# SCENE分析（Systematic Character Extraction for Narrative Epilogue）
#=============================================================

#====================================================
# アクチュアル・ドラマ、アフター・ストーリー生成の準備
#====================================================

# ライブラリの準備
import os
import time
import numpy as np
import re
import random
from colorama import Fore, Back, Style
import pandas as pd
from itertools import combinations
from sqlalchemy import create_engine
import shutil
from google import genai  # Gemini APIクライアント
from google.genai.errors import APIError # Gemini APIのエラー処理
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv 

# モジュールの準備
import My_Global as g
import SCENE

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
# アクチュアル・ドラマ生成関数
#====================================================

# 実際のレース結果をアクチュアル・ドラマとして再現する（Gemini API利用）
def Actual_Race_Projector(cast_df, ensemble_df, final_result, client, model):
    race_info = f'{g.stadium}競馬場 {g.clas} {g.td} {g.distance}m {g.race_name} ({g.cond})'
    scene_instruction = "、".join(g.selected_scenes)
    
    # 1. 前提データ（レース展開）
    race_context = ""
    # 必要なカラムを抽出
    for _, row in final_result.iterrows():
        race_context += (f"【{row['馬番']}番 {row['馬名']}】"
                             f"1角：{row['通過1']}番手 - 2角：{row['通過2']}番手 - 3角：{row['通過3']}番手 - 4角：{row['通過4']}番手 - 着順：{row['確定着順']}着 (着差：{row['着差']})\n")
    
    # 2. 戦術・能力データ (展開に直結する情報を追加)
    tactical_context = ""
    # 必要なカラムを抽出
    for _, row in cast_df.iterrows():
        tactical_context += (f"【{row['番']}番 {row['馬名']}】"
                             f"枠:{row['枠番']}, 脚質:{row['脚質']}, 先行指数:{row['先行指数']}, "
                             f"安定度:{row['安定度']}, 中何週：{row['中何週']}週, 成長度：{row['調教成長ポイント']} ")

    # 3. キャラデータ (心情やセリフに関係する情報を追加)
    character_context = ""
    # 必要なカラムを抽出
    for _, row in cast_df.iterrows():
        character_context += (f"【{row['番']}番 {row['馬名']}】"
                             f"家柄:{row['血統分析']}, キャラ:{row['キャラ設定']},  勝負服色:{row['勝負服色']}"
                             f"背景:{row['自己紹介'][:60]}...\n") # 背景は要約してトークン節約

    # 5. ライバル関係
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
        必ずレース展開の通りとなるよう、特に、各コーナーでの通過順に忠実に、ストーリを構築してください。
        
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

        
        【資料：レース展開（各コーナー通過順・着順）】
        {race_context}
        【資料：戦術・能力データ】
        {tactical_context}
        【資料：キャラデータ】
        {character_context}
        【資料：ライバル関係】
        {rivalry_context}
        """

    print(f"{g.race_name}：アクチュアル・ドラマ生成中...")
    
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.8,
        }
    )
    
    print(Fore.YELLOW + f"{g.race_name}：アクチュアル・ドラマ生成完了" + Style.RESET_ALL)
    print('')
    
    return response.text


#====================================================
# アフター・ストーリー生成関数
#====================================================

# アクチュアル・ドラマから、その後の馬同士のとある場面での会話を生成（Gemeni API利用）
def After_Story_Extractor(actual_drama, final_result, cast_df, ensemble_df, client, model):
    horse_info = final_result[['馬名', '確定着順']].to_string(index=False)

    if g.cond == '稍':
        g.cond = '稍重'
    elif g.cond == '不':
        g.cond = '不良'

    race_header = f"{g.race_date} {g.stadium}競馬場 {g.race_name} ({g.td}{g.distance}m / 馬場:{g.cond})"

    # 前提データ（レース展開）
    final_result = final_result.head(3)
    race_context = ""
    # 必要なカラムを抽出
    for _, row in final_result.iterrows():
        race_context += (f"【{row['馬番']}番 {row['馬名']}】"
                             f"1角：{row['通過1']}番手 - 2角：{row['通過2']}番手 - 3角：{row['通過3']}番手 - 4角：{row['通過4']}番手 - 着順：{row['確定着順']}着 (着差：{row['着差']}) ({row['人気']}番人気)\n")

    # キャラデータ (心情やセリフに関係する情報を追加)
    character_context = ""
    # 必要なカラムを抽出
    for _, row in cast_df.iterrows():
        character_context += (f"【{row['番']}番 {row['馬名']}】"
                             f"家柄:{row['血統分析']}, キャラ:{row['キャラ設定']}\n")

    # ライバル関係
    rivalry_context = "\n".join(ensemble_df['ライバル関係'].tolist())

    # シーン抽選（時間帯とシーン）
    selected_time = ""
    times = ['早朝', '午前中', 'お昼', '午後', '夕刻', '夜']
    weights = [10, 20, 20, 20, 20, 10]
    selected_time = random.choices(times, weights=weights)[0]

    selected_scene = ""
    scenes = ['トレセンの中庭', 'トレセンの練習場', '寮の食堂', 'シャワールーム', 'トレーニングルーム', 'とある街角', 'カフェ', 'ショッピングモール', '遊歩道', '公園']
    weights = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    selected_scene = random.choices(scenes, weights=weights)[0]

    horses_order = ""
    orders = ['1着馬,2着馬,3着馬', '1着馬,3着馬,2着馬', '2着馬,1着馬,3着馬', '2着馬,3着馬,1着馬', '3着馬,1着馬,2着馬', '3着馬,2着馬,1着馬']
    weights = [15, 15, 20, 15, 20, 15]
    horses_order = random.choices(orders, weights=weights)[0]

    system_instruction = f"""
        あなたは短編ラノベ作家です。
        {race_header}に出走し、3着までに入った馬たちが、後日{selected_time}の{selected_scene}に偶然居合わせて会話するシーンを、アフターストーリーとして創出してください。
    """

    user_prompt = f"""
        【執筆ルール】
        - アフターストーリーは、実際のレースの物語とレース結果を前提としてください。
        - それぞれの馬のキャラ設定とライバル関係を踏まえて、アフターストーリーを展開してください。
        - 誰もが気軽に楽しめるラノベ風の描写とし、馬同士の会話を中心としたストーリーを描いてください。

        【ストーリー展開】
        - アフターストーリーは以下の2部構成とします。
        [第1部]（400文字程度）
        - **レース回顧** として、いきなり第4コーナーを回って最終直線に入るシーンから始め、ゴールまでの熱戦を、{actual_drama}から切り取って、簡潔に再現してください。
        - セリフは全て排除し、各馬の必死の表情や飛び散る汗、競馬場の熱気や歓声をメインに、短くもドラマチックに仕上げてください。    
        [第2部]（1600文字程度）
        - **アフターストーリー** として、冒頭で{selected_time}の{selected_scene}の雰囲気について簡潔に描写してください
        - それぞれの馬は、{horses_order}の順に、そのシーンに現れるようしてください。その際、同時に登場する場合や、間隔をあけて登場する場合等、タイミングはランダムに設定してください。
        （例）2着馬と3着馬が会話している途中で、1着馬が現れる、等
        - レースの反省や次回レースへの思い、お互いの複雑な心情、相手に対する尊敬や熱いライバル心、などを描いててください。
        - レース当日とは違い、普段のプライベートな会話となるよう意識し、レースとは違った、よりリラックスした会話にしてください。
        - キャラ設定を踏まえ、真面目な会話だけでなく、冗談、皮肉、からかい、といった内容も散りばめてください。
        - 3頭のいずれかが実際にライバル関係にある場合には、ライバル関係の背景、勝った喜び、滲み出る悔しさ、リベンジの誓い、因縁のライバル関係、といった熱い内容を入れてください。

        【登場キャラ紹介】
        - アフターストーリーに続き、登場した3人のキャラの紹介（馬名：キャラ設定）を、1着、2着、3着の順に、簡潔に記載してください。
        （例）「1着 アーモンドアイ：キャラ設定を簡潔に記述」
        
        【データ参照】
        出走馬: {horse_info}
        実際のレースの物語：{actual_drama} (この物語を踏まえて、各馬の会話を抽出してください。)
        レース結果: {race_context} (この結果を踏まえて、各馬の会話を抽出してください。)
        【資料：キャラ設定】
        {character_context}
        【資料：ライバル関係】
        {rivalry_context}
        """

    print(f"{g.race_name}：アフター・ストーリー生成中...")
    
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.3, # 創作を抑え、事実描写に固定
        }
    )
    
    print(Fore.YELLOW + f"{g.race_name}：アフター・ストーリー生成完了" + Style.RESET_ALL)
    print('')

    return response.text


#====================================================
# 実行
#====================================================

if __name__ == "__main__":

    # 作業用フォルダの設定
    work_dir = '/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/'
    # 生成データの保存先フォルダ
    media_dir = f'/Users/trueocean/Desktop/Python_Code/PRISM_SCENE/Media_files/'
    # アーカイブフォルダの設定
    race_dir = '/Users/trueocean/Desktop/PRISM_SCENE/Archive/' + g.race_date + '/' + g.stadium + '/' + g.r_num + '/'

    # データフレームの読み込み
    SCENE_Cast_df = pd.read_csv(f'{work_dir}SCENE_Cast.csv', encoding = 'utf-8')
    SCENE_Ensemble_df = pd.read_csv(f'{work_dir}SCENE_Ebsemble.csv', encoding = 'utf-8')
    RaceResult_df = pd.read_csv(f'{work_dir}RaceResult.csv', encoding = 'cp932')

    # ファイナル・ドラマ生成
    actual_drama = Actual_Race_Projector(SCENE_Cast_df, SCENE_Ensemble_df, RaceResult_df, client, MODEL)
    save_drama_name = f'{media_dir}Actual_Drama.txt'
    SCENE.save_text_to_file(actual_drama, save_drama_name)

    # アフター・ストーリー生成
    after_story = After_Story_Extractor(actual_drama, RaceResult_df, SCENE_Cast_df, SCENE_Ensemble_df, client, MODEL)
    save_story_name = f'{media_dir}After_Story.txt'
    SCENE.save_text_to_file(after_story, save_story_name)

    # 生成したデータをアーカイブフォルダにコピー
    shutil.copy(f'{media_dir}Actual_Drama.txt', race_dir)
    shutil.copy(f'{media_dir}After_Story.txt', race_dir)




