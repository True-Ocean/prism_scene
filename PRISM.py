#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
#=============================================================

#====================================================
# PRISM分析の準備
#====================================================

# ライブラリの準備
import os
import pandas as pd
import numpy as np
import math
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy.types import Date, Time, Integer, Float
from sqlalchemy import create_engine, text
from colorama import Fore, Back, Style
import time
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
import shutil

import warnings
warnings.filterwarnings('ignore')

# モジュールの準備
import My_Global as g
import PRISM_SCENE_Menu
import Data_Getter
import Data_Preparation
import PRISM_R
import PRISM_G
import PRISM_B

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
# PRISM_SCENE分析 メニュー画面表示：レース情報の取得
#====================================================

print(Fore.GREEN)
print('====================================================')
print('                 PRISM_SCENE 分析')
print('====================================================')
print(Fore.YELLOW)
print('ようこそ、PRISM_SCENE分析へ！')
print(Style.RESET_ALL)
print('これからPRISM_SCENE分析を実施します。')
print('メニュー画面にレース情報をインプットしてください。')

# メニュー画面からレース情報データを取得
PRISM_SCENE_Menu.PRISM_SCENE_Menu()

# メニューダイアログが消えるのを確実に待ち、フォーカスを戻す
#time.sleep(1.5) # PyAutoGUIが走る前に少し長めに待つ

# データフレームにレース情報を格納
col = ['日付', '競馬場', 'R番号', '年齢', 'クラス', 'TD', '距離', '状態', 'レース名']
r_info = [[g.race_date, g.stadium, g.r_num, g.age, g.clas, g.td, g.distance, g.cond, g.race_name]]
RaceInfo_df = pd.DataFrame(data = r_info, index = ['レース情報'], columns = col)

# csvとして保存
RaceInfo_df.to_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/RaceInfo.csv', index=False, encoding="utf-8")
# PostgreSQLに保存
RaceInfo_df.to_sql('RaceInfo', con=engine, if_exists = 'replace')

print(Fore.YELLOW)
print('今回のレース情報を取得しました。')
print(Style.RESET_ALL)


#====================================================
# PRISM_SCENE分析に必要なデータの自動取得
#====================================================

# pyautoguiによる自動マウス操作でTFJVから全ての必要データを取得
Data_Getter.Data_Getter()


#====================================================
# 取得したデータの整形・準備
#====================================================

print(Fore.GREEN)
print('====================================================')
print('  PRISM_SCENE分析に必要なデータの整形')
print('====================================================')
print(Style.RESET_ALL)
print('PRISM_SCENE分析に必要なデータを整形しています。')

# 出馬表の取得
RaceTable_df = Data_Preparation.Race_Table_Preparation()
# 全出走馬のレース実績データの取得
HorseRecords_df = Data_Preparation.Horse_Records_Preparation(RaceTable_df)
# 調教データの取得
Hanro_df, CW_df = Data_Preparation.Training_Data_Preparation()

# PRISM分析に必要となる3つの基本DFをPostgreSQLに保存
RaceTable_df.to_sql('RaceTable', con=engine, if_exists = 'replace')
HorseRecords_df.to_sql('HorseRecords', con=engine, if_exists = 'replace')
Hanro_df.to_sql('Hanro', con=engine, if_exists = 'replace')
CW_df.to_sql('CW', con=engine, if_exists = 'replace')

print(Fore.YELLOW)
print('PRISM_SCENE分析に必要なデータの整形が完了しました。')
print(Style.RESET_ALL)


#====================================================
# PRISM分析の実行
#====================================================

if g.exe_opt in [2, 4]:

    #====================================================
    # PRISM分析の実行
    #====================================================

    print(Fore.GREEN)
    print('====================================================')
    print('  PRISM分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('これより、PRISM分析を開始します。')

    #====================================================
    # PRISM_R分析の実行
    #====================================================

    print(Fore.GREEN)
    print('====================================================')
    print('  PRISM_R分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('PRISM_R分析を実行しています。')

    # 馬名リストの取得
    target_horses = RaceTable_df['馬名'].tolist()

    # 対象馬の過去成績をDBから取得
    query = f"SELECT * FROM \"HorseRecords\" WHERE \"馬名\" IN ({str(target_horses)[1:-1]})"
    horse_records_all = pd.read_sql(query, con=engine)

    # PRISM_Base：基礎偏差値の算出
    PRISM_Base_df = PRISM_R.PRISM_Base(engine, horse_records_all)
    # PRISM_R分析の実行
    PRISM_R_df = PRISM_R.PRISM_R_Analysis(PRISM_Base_df, RaceTable_df,)

    # PostgreSQLへの保存
    PRISM_Base_df.to_sql('PRISM_Base', con=engine, if_exists='replace', index=False)
    PRISM_R_df.to_sql('PRISM_R', con=engine, if_exists='replace', index=False)

    # PRISM_Rのビジュアル化実行
    PRISM_R.PRISM_R_Visualization(PRISM_R_df)

    print(Fore.YELLOW)
    print('PRISM_R分析が完了しました。')
    print(Style.RESET_ALL)


    #====================================================
    # PRISM_G分析の実行
    #====================================================

    print(Fore.GREEN)
    print('====================================================')
    print('  PRISM_G分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('PRISM_G分析を実行しています。')

    # マスターデータの読み込み
    MasterDataset_df = pd.read_sql(sql='SELECT * FROM "MasterDataset";', con=engine)

    # PRISM_Gの実行
    track_summary, intrinsic_baselines = PRISM_G.RPCI_Shift_Analysis(MasterDataset_df)

    # 環境・展開補正の適用
    PRISM_RG_df = PRISM_G.PRISM_G_Analysis(
        PRISM_R_df, 
        MasterDataset_df, 
        RaceTable_df, 
        track_summary, 
        intrinsic_baselines
    )

    # PostgreSQLに保存
    PRISM_RG_df.to_sql('PRISM_RG', con=engine, if_exists='replace')

    # PRISM_Gのビジュアル化実行
    PRISM_G.PRISM_G_Visualization(PRISM_RG_df)
    # PRISM_RGのビジュアル化実行
    PRISM_G.PRISM_RG_Visualization(PRISM_RG_df)

    print(Fore.YELLOW)
    print('PRISM_G分析が完了しました。')
    print(Style.RESET_ALL)


    #====================================================
    # PRISM_B分析の実行
    #====================================================

    print(Fore.GREEN)
    print('====================================================')
    print('  PRISM_B分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('PRISM_B分析を実行しています。')

    # PRISM_Bの実行
    PRISM_B_df = PRISM_B.PRISM_B_Analysis(RaceTable_df, HorseRecords_df, CW_df, Hanro_df, g.race_date)
    PRISM_B_df.to_sql('PRISM_B', con=engine, if_exists = 'replace')

    # PRISM_RGBの実行
    PRISM_RGB_df = PRISM_B.Calculate_PRISM_RGB(PRISM_RG_df, PRISM_B_df)
    PRISM_RGB_df.to_sql('PRISM_RGB', con=engine, if_exists = 'replace')

    print(Fore.YELLOW)
    print('PRISM_B分析が完了しました。')
    print(Style.RESET_ALL)

    # PRISM_Bのビジュアル化実行
    PRISM_B.PRISM_B_Visualization(PRISM_B_df, RaceTable_df)
    # PRISM_RGBのビジュアル化実行
    PRISM_B.PRISM_RGB_Visualization(PRISM_RGB_df)
    # 調教データのビジュアル化実行
    PRISM_B.Horse_Training_Visualization(RaceTable_df, CW_df, Hanro_df, g.race_date)

    print(Fore.RED)
    print('PRISM分析が完了しました!')
    print(Style.RESET_ALL)


    #====================================================
    # PRISM可視化データのアーカイブ
    #====================================================

    # アーカイブフォルダの設定
    race_dir = '/Users/trueocean/Desktop/PRISM_SCENE/Archive/' + g.race_date + '/' + g.stadium + '/' + g.r_num + '/'
    # 作業用フォルダの設定
    work_dir = '/Users/trueocean/Desktop/Python_Code/PRISM_SCENE/Media_files/'

    # 各画像データをアーカイブフォルダにコピー
    shutil.copy(f'{work_dir}PRISM_R.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_G.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_RG.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_B_Hanro.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_B_CW.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_B_Hanro_Time.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_B_Hanro_Lap.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_B_CW_Time.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_B_CW_Lap.png', race_dir)
    shutil.copy(f'{work_dir}PRISM_RGB.png', race_dir)


#====================================================
# SCENE分析の実行
#====================================================

if g.exe_opt in [3, 4]:

    print(Fore.GREEN)
    print('====================================================')
    print('  SCENE分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('これより、SCENE分析を開始します。')
    print('Now Under Construction...')
    print('')
