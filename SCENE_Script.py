#===================================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# SCENE分析（Systematic Character Extraction for Narrative Epilogue）
# SCENE_Script分析
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
    PRISM_R = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/PRISM_R.csv', encoding = 'utf-8')
    PRISM_RG = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/PRISM_RG.csv', encoding = 'utf-8')
    PRISM_B = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/PRISM_B.csv', encoding = 'utf-8')
    PRISM_RGB = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/PRISM_RGB.csv', encoding = 'utf-8')

    HorseRecords_df = pd.read_sql('SELECT * FROM "HorseRecords"', con=engine)
    RaceTable_df = pd.read_sql('SELECT * FROM "RaceTable"', con=engine)

    # 1. 出馬表 (RaceTable_df) から最新の「性別」と「年齢」を取得する
    current_attributes = RaceTable_df[['馬名', '性別', '年齢']].drop_duplicates(subset=['馬名'])

    # 2. 実績データ (HorseRecords_df) からは「勝負服色」のみを取得する
    cloth_colors = HorseRecords_df.sort_values('日付', ascending=False).groupby('馬名').agg({
        '勝負服色': 'first'
    }).reset_index()

    # 3. 属性情報を統合する
    SCENE_Attributes_df = pd.merge(current_attributes, cloth_colors, on='馬名', how='left')

    # 馬名の重複を完全に排除（もしaggで漏れがあった場合用）
    SCENE_Attributes_df = SCENE_Attributes_df.drop_duplicates(subset=['馬名'])
    # インデックスをリセットして綺麗にする
    SCENE_Attributes_df = SCENE_Attributes_df.reset_index(drop=True)

    # 1. ベースとなるデータフレームを作成
    # mergeを繰り返して、馬名をキーに全データを統合する（これが最も安全）
    df = PRISM_RG[['枠番', '番', '馬名', 'PRISM_R_Score', 'G_Avg']].copy()
    
    # 2. 各データを馬名キーで安全に結合
    df = df.merge(PRISM_R[['馬名', 'EPI', '脚質', '実力のムラ']], on='馬名', how='left')
    df = df.merge(PRISM_B[['馬名', '中何週']], on='馬名', how='left')
    df = df.merge(PRISM_RGB[['馬名', 'PRISM_B_Score', 'PRISM_RGB_Score']], on='馬名', how='left')
    df = df.merge(SCENE_Attributes_df[['馬名', '性別', '年齢', '勝負服色']], on='馬名', how='left')

    # 3. カラム名の整理
    column_map = {
        'EPI': '先行指数',
        'PRISM_R_Score': '基礎能力',
        'G_Avg': 'レース条件適合率',
        'PRISM_B_Score': '調教成長ポイント',
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
                f"{int(row['枠番'])}枠({row['枠色']}) {int(row['番'])}番 {row['馬名']} {row['性別']}{int(row['年齢'])}歳, "
                f"先行指数:{row['先行指数']:.2f}, 脚質：{row['脚質']}, 実力のムラ:{row['実力のムラ']:.2f}, "
                f"基礎能力:{row['基礎能力']:.2f}, レース条件適合率:{row['レース条件適合率']:.2%}, "
                f"中{int(row['中何週'])}週, 調教成長ポイント:{row['調教成長ポイント']:.2f}, 最終期待値:{row['最終期待値']:.2f}, 勝負服色:({row['勝負服色']})"
            )
        except:
            return f"データ不足: {row['馬名']}"

    df['特徴'] = df.apply(generate_feature_text, axis=1)

    # 6. 並べ替えと保存
    df = df.sort_values(by='番').reset_index(drop=True)
    df.to_sql('SCENE_Script', con=engine, if_exists='replace', index=False)

    return df

#====================================================
# SCENE_Scriptの作成
#====================================================

if __name__ == "__main__":
    
    SCENE_Script_df = SCENE_Script()