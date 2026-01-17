#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
# レース情報の準備
#=============================================================

# ライブラリの準備
import os
import pandas as pd
# Pandasの未来の仕様変更への警告を抑制
pd.set_option('future.no_silent_downcasting', True)
from dotenv import load_dotenv
from sqlalchemy import create_engine
import time

# グローバル変数用モジュール
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


#====================================================
# 出馬表の準備
#====================================================

def Race_Table_Preparation ():

    # 出馬表データ（ファイル名：Race_Table.csv）の読み込み。
    race_table_df = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/RaceTable.csv', encoding = 'cp932')

    # 出走頭数の定数化
    g.hr_num = len(race_table_df)

    # 必要なカラム抽出とカラム名修正
    race_table_df = race_table_df[['枠番', '番', '  馬名', '性別', '年齢', '騎手', '斤量', '所属', '調教師', ' 馬主', ' 生産者', '毛色']]
    race_table_df = race_table_df.rename(columns = {'  馬名':'馬名'})

    # 欠損値nanを０に置換（主に「増減」）
    race_table_df = race_table_df.fillna(0)

    # 厩舎の表記を変更（ベクトル化・高速）
    mapping = {'(栗)': '栗東', '(美)': '美浦', '[地]': '地方', '[外]': '海外'}
    race_table_df['所属'] = race_table_df['所属'].replace(mapping)

    # 最終データフレーム化
    RaceTable_df = race_table_df

    # PostgreSQLに上書き保存
    RaceTable_df.to_sql('RaceTable', con=engine, if_exists = 'replace')
    
    return RaceTable_df


#====================================================
# 今回のレース出走馬の過去レース実績データの取得
#====================================================

def Horse_Records_Preparation(race_table_df):

    # 全馬のdf_umaをリスト化するための空のリストを用意
    list_uma_df = []

    for i in range(len(race_table_df)):

        # 事前にTFJVで各馬のレース実績データを取得し、csv形式（ファイル名：Uma**.csv）で保存しておき、これを読み込む。
        df_uma = pd.read_csv(f'/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/Uma{i+1}.csv', encoding = 'cp932')

        df_uma = df_uma[['日付', '場所', 'R','クラス名', 'レース名', '芝・ダート', '距離', '年齢限定', '馬場状態', '出走頭数', 'レースPCI', '年齢', '性別', '馬番', '斤量', '体重', '斤量体重比', \
            '通過1', '通過2', '通過3', '通過4', ' Ave-3F', '上り3F', ' PCI', 'タイムS', '人気', '着順', '着差', '決手', '勝負服色']]

        # カラム名を前処理（特に、' Ave-3F'と' PCI'は前に半角スペースがあるため、半角スペースを削除していることに注意。今後TFJVの仕様変更等でエラーが出る可能性あり）
        df_uma = df_uma.rename(columns = {'芝・ダート':'TD', '体重':'馬体重', '通過1':'1角', '通過2':'2角', '通過3':'3角', '通過4':'4角', ' Ave-3F':'Ave-3F', \
        ' PCI':'PCI','タイムS':'走破タイム', 'レースPCI':'RPCI',})

        # 欠損値nanを０に置換
        df_uma = df_uma.fillna(0)

        # ここから馬別実績データの前処理

        # 日付の前処理（年月日を分割し、一旦int型に変更したのち、str型に変更し、結合、datetime型に変換）
        df_uma['年'] = df_uma['日付'].str[:4]
        df_uma['年'] = df_uma['年'].astype(int)

        df_uma['月'] = df_uma['日付'].str[5:7]
        df_uma['月'] = df_uma['月'].astype(int)

        df_uma['日'] = df_uma['日付'].str[8:10]
        df_uma['日'] = df_uma['日'].astype(int)

        df_uma['年'] = df_uma['年'].astype(str)
        df_uma['月'] = df_uma['月'].astype(str)
        df_uma['日'] = df_uma['日'].astype(str)
        df_uma['日付'] = df_uma['年']+['-']+df_uma['月']+['-']+df_uma['日']

        df_uma['日付'] = pd.to_datetime(df_uma['日付'])

        # 走破タイムの前処理（走破タイムを分と秒に分け、秒換算
        df_uma['分'] = df_uma['走破タイム'].str[:1]
        df_uma['分'] = df_uma['分'].astype(int)
        df_uma['秒'] = df_uma['走破タイム'].str[2:7]
        df_uma['秒'] = df_uma['秒'].astype(float)
        df_uma['走破タイム'] = df_uma['分']*60 + df_uma['秒']

        # 走破タイムに'----'データが含まれる場合、および走破タイムが0の場合は0にする
        df_uma['走破タイム'] = pd.to_numeric(df_uma['走破タイム'], errors='coerce').fillna(0)
        # 着差に'----'データが含まれる場合、欠損値（NaN）に変換する
        df_uma['着差'] = pd.to_numeric(df_uma['着差'], errors='coerce')

        #　距離をint型に変換
        df_uma['距離'] = df_uma['距離'].astype(int)

        # 着順データ内の不純物を除去した後、int型に変換
        drop_index = df_uma[(df_uma['着順'] == '①') | (df_uma['着順'] == '②') | (df_uma['着順'] == '③') | (df_uma['着順'] == '④') | \
        (df_uma['着順'] == '消') | (df_uma['着順'] == '止') | (df_uma['着順'] == '外') | (df_uma['着順'] == '失')].index
        df_uma = df_uma.drop(drop_index)
        target_columns = ['人気', '着順', '1角', '2角', '3角', '4角']
        df_uma[target_columns] = df_uma[target_columns].astype(int)

        df_uma = df_uma.drop(['年', '月', '日', '分', '秒'], axis=1)

        df_uma['斤量'] = df_uma['斤量'].astype(str)
        df_uma['斤量'] = df_uma['斤量'].str.replace('[▲△★☆◆◇]', '', regex=True)
        df_uma['斤量'] = df_uma['斤量'].astype(float)

        df_uma['勝負服色'] = df_uma['勝負服色'].replace(0.0, '不明')


        # 馬名の取得
        hr_col = race_table_df.columns.get_loc('馬名')
        hr_name = race_table_df.iloc[i, hr_col]
        df_uma['馬名'] = hr_name

        # 馬名カラムをRPCIの次のカラムに移動
        columns = df_uma.columns.tolist()
        columns.remove('馬名')
        key_col = columns.index('RPCI')
        columns.insert(key_col + 1, '馬名')
        df_uma = df_uma[columns]

        # PostgreSQLとcsvに上書き保存
        df_uma.to_sql(f'Uma{i+1}', con=engine, if_exists = 'replace')

        # リストに各馬のdf_umaを追加
        list_uma_df.append(df_uma)

    # データフレームの最終化
    HorseRecords_df = pd.concat(list_uma_df, axis=0, ignore_index=True)

    return HorseRecords_df


#====================================================
# 調教データの取得
#====================================================

def Training_Data_Preparation():

    # 調教データ読み込み
    df_hanro = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/Hanro.csv', encoding = 'cp932')
    df_cw = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/CW.csv', encoding = 'cp932')

    # 坂路調教データの整形
    df_hanro = df_hanro.rename(columns = {'馬番/仮番(出馬表モード時のみ)':'番'})
    df_hanro['番'] = df_hanro['番'].astype(int)
    df_hanro = df_hanro[['年月日', '時刻', '番','馬名', '性別', '年齢', '調教師','Time1','Time2','Time3','Time4','Lap4','Lap3','Lap2','Lap1']]
    
    # すべてのラップが0より大きい行だけを残す
    lap_cols = ['Lap1', 'Lap2', 'Lap3', 'Lap4']
    df_hanro = df_hanro[(df_hanro[lap_cols] != 0).all(axis=1)]

    # CW調教データの整形
    df_cw = df_cw.rename(columns = {'馬番/仮番(出馬表モード時のみ)':'番'})
    df_cw['番'] = df_cw['番'].astype(int)
    df_cw = df_cw[['年月日', '時刻', '番','馬名', '性別', '年齢', '調教師','所属', '回り', '10F', '9F', '8F', '7F', '6F', '5F', '4F', '3F', '2F', '1F',
                'Lap9', 'Lap8', 'Lap7', 'Lap6', 'Lap5', 'Lap4', 'Lap3', 'Lap2', 'Lap1',]]

    # Lap6 ~ 1が0より大きい行だけを残す
    lap_cols = ['Lap1', 'Lap2', 'Lap3', 'Lap4', 'Lap5', 'Lap6']
    df_cw = df_cw[(df_cw[lap_cols] != 0).all(axis=1)]

    return df_hanro, df_cw


#====================================================
# 上記の各関数の実行（4つの基本データフレームの取得）
#====================================================

if __name__ == '__main__':

    # 出馬表の取得
    RaceTable_df = Race_Table_Preparation()

    # 全出走馬のレース実績データの取得
    HorseRecords_df = Horse_Records_Preparation(RaceTable_df)

    # 調教データの取得
    df_hanro, df_cw = Training_Data_Preparation()

    # PRISM分析に必要となるDFをPostgreSQLに保存
    RaceTable_df.to_sql('RaceTable', con=engine, if_exists = 'replace')
    HorseRecords_df.to_sql(f'HorseRecords', con=engine, if_exists = 'replace')
    df_hanro.to_sql('Hanro', con=engine, if_exists = 'replace')
    df_cw.to_sql('CW', con=engine, if_exists = 'replace')
