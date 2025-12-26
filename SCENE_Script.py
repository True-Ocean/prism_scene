#===================================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# SCENE分析（Systematic Character Extraction for Narrative Epilogue）
#===================================================================

#====================================================
# SCENE_Scriptの作成
#====================================================

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import warnings

# warningsの無視設定
warnings.filterwarnings('ignore')

def get_engine():
    dotenv_path = '/Users/trueocean/Desktop/Python_Code/Project_Key/.env'
    load_dotenv(dotenv_path)
    connection_config = {
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASS'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'database': os.getenv('DB_NAME')
    }
    return create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format(**connection_config))

def SCENE_Script():
    engine = get_engine()

    # データの読み込み
    PRISM_R = pd.read_sql('SELECT * FROM "PRISM_R"', con=engine)
    PRISM_RG = pd.read_sql('SELECT * FROM "PRISM_RG"', con=engine)
    PRISM_B = pd.read_sql('SELECT * FROM "PRISM_B"', con=engine)
    PRISM_RGB = pd.read_sql('SELECT * FROM "PRISM_RGB"', con=engine)

    # 1. ベースとなるデータフレームを作成
    # mergeを繰り返して、馬名をキーに全データを統合する（これが最も安全）
    df = PRISM_RG[['枠番', '番', '馬名', 'PRISM_R_Score', 'G_Avg']].copy()
    
    # 2. 各データを馬名キーで安全に結合
    df = df.merge(PRISM_R[['馬名', 'EPI', '脚質', '安定度']], on='馬名', how='left')
    df = df.merge(PRISM_B[['馬名', '中何週']], on='馬名', how='left')
    df = df.merge(PRISM_RGB[['馬名', 'PRISM_B_Score', 'PRISM_RGB_Score']], on='馬名', how='left')

    # 3. カラム名の整理
    column_map = {
        'EPI': '先行指数',
        'PRISM_R_Score': '基礎能力',
        'G_Avg': 'レース条件適合率',
        'PRISM_B_Score': '調教成長度',
        'PRISM_RGB_Score': '最終期待値'
    }
    df = df.rename(columns=column_map)

    # 4. 枠色の追加
    waku_color_map = {1:'白', 2:'黒', 3:'赤', 4:'青', 5:'黄', 6:'緑', 7:'オレンジ', 8:'ピンク'}
    df['枠色'] = df['枠番'].map(waku_color_map)

    # 5. 特徴テキストの生成（欠損値対策としてint変換前にfillnaを入れるとより安全）
    def generate_feature_text(row):
        try:
            return (
                f"{int(row['枠番'])}枠({row['枠色']}) {int(row['番'])}番 {row['馬名']}, "
                f"先行指数:{row['先行指数']:.2f}, 脚質：{row['脚質']}, 安定度:{row['安定度']:.2f}, "
                f"基礎能力:{row['基礎能力']:.2f}, レース条件適合率:{row['レース条件適合率']:.2%}, "
                f"中:{int(row['中何週'])}週, 調教成長度:{row['調教成長度']:.2f}, 最終期待値:{row['最終期待値']:.2f}"
            )
        except:
            return f"データ不足: {row['馬名']}"

    df['特徴'] = df.apply(generate_feature_text, axis=1)

    # 6. 並べ替えと保存
    df = df.sort_values(by='番').reset_index(drop=True)
    df.to_sql('SCENE_Script', con=engine, if_exists='replace', index=False)


#====================================================
# SCENE_Scriptの作成
#====================================================

if __name__ == "__main__":
    SCENE_Script()