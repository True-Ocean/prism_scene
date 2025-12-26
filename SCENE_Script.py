#===================================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# SCENE分析（Systematic Character Extraction for Narrative Epilogue）
#===================================================================

#====================================================
# SCENE_Scriptの作成
#====================================================

def SCENE_Script():
    # ライブラリの準備
    import os
    import pandas as pd
    import numpy as np
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text
    from colorama import Fore, Back, Style

    import warnings
    warnings.filterwarnings('ignore')

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

    PRISM_R_df = pd.read_sql(sql = f'SELECT * FROM "PRISM_R";', con=engine)
    PRISM_RG_df = pd.read_sql(sql = f'SELECT * FROM "PRISM_RG";', con=engine)
    PRISM_B_df = pd.read_sql(sql = f'SELECT * FROM "PRISM_B";', con=engine)
    PRISM_RGB_df = pd.read_sql(sql = f'SELECT * FROM "PRISM_RGB";', con=engine)


    PRISM_R_df = PRISM_R_df.sort_values(by = '番', ascending=True)
    PRISM_RG_df = PRISM_RG_df.sort_values(by = '番', ascending=True)
    PRISM_B_df = PRISM_B_df.sort_index(ascending=True)
    PRISM_RGB_df = PRISM_RGB_df.sort_values(by = '番', ascending=True)

    SCENE_Script_df = PRISM_RG_df[['枠番', '番', '馬名']]
    SCENE_Script_df[['先行指数', '脚質', '安定度']] = PRISM_R_df[['EPI', '脚質', '安定度']]
    SCENE_Script_df[['基礎能力', 'レース条件適合率']] = PRISM_RG_df[['PRISM_R_Score', 'G_Avg']]
    SCENE_Script_df['調教成長度'] = PRISM_RGB_df['PRISM_B_Score']
    SCENE_Script_df['中何週'] = PRISM_B_df['中何週']

    # 馬名をキーにして左結合（Left Join）
    SCENE_Script_df = SCENE_Script_df.merge(
        PRISM_RGB_df[['馬名', 'PRISM_RGB_Score']], 
        on='馬名', 
        how='left'
    )

    # 'PRISM_RGB_Score' を '最終期待値' に変更
    SCENE_Script_df = SCENE_Script_df.rename(columns={'PRISM_RGB_Score': '最終期待値'})

    # 枠番と色の対応辞書を作成
    waku_color_map = {
        1: '白',
        2: '黒',
        3: '赤',
        4: '青',
        5: '黄',
        6: '緑',
        7: 'オレンジ',
        8: 'ピンク'
    }
    # mapメソッドを使って新しいカラム「枠色」を作成
    SCENE_Script_df['枠色'] = SCENE_Script_df['枠番'].map(waku_color_map)

    def generate_feature_text(row):
        # ここで計算（row['レース条件適合率'] * 100）をしても、
        # 元のデータフレームのカラム値は書き換わりません。
        return (
            f"{int(row['枠番'])}枠"
            f"({row['枠色']}) "
            f"{int(row['番'])}番 "
            f"{row['馬名']}, "
            f"先行指数:{row['先行指数']:.2f}, "
            f"脚質：{row['脚質']}, "
            f"安定度:{row['安定度']:.2f}, "
            f"基礎能力:{row['基礎能力']:.2f}, "
            f"レース条件適合率:{row['レース条件適合率']:.2%} "
            f"中:{int(row['中何週'])}週 "
            f"調教成長度:{row['調教成長度']:.2f}, "
            f"最終期待値:{row['最終期待値']:.2f}"
        )

    # 元の数値カラムは維持したまま、新しくテキスト用のカラムを作る
    SCENE_Script_df['特徴'] = SCENE_Script_df.apply(generate_feature_text, axis=1)

    SCENE_Script_df.to_sql('SCENE_Script', con=engine, if_exists = 'replace')

#====================================================
# SCENE_Scriptの実行
#====================================================

if __name__ == "__main__":
    SCENE_Script()