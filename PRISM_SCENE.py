#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM_SCENE分析
# (Performance Rating and Intelligent Scoring Model &
#  Systematic Character Extraction for Narrative Epilogue)
#=============================================================

#====================================================
# PRISM分析の準備
#====================================================

# ライブラリの準備
import os
import pandas as pd
import numpy as np
import math
import re
import random
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
from bs4 import BeautifulSoup # HTMLの整形
from itertools import combinations
from google import genai  # Gemini APIクライアント
from google.genai.errors import APIError # Gemini APIのエラー処理
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pydantic import BaseModel, Field # JSONスキーマの定義に利用
import asyncio # 並列処理
import edge_tts # Microsoftの音声ファイル生成

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
import SCENE_Script
import SCENE_Cast
import SCENE_Ensemble
import SCENE
import Race_Audio_Maker
import After_Story_Extractor

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

# アーカイブフォルダの設定
race_dir = '/Users/trueocean/Desktop/PRISM_SCENE/Archive/' + g.race_date + '/' + g.stadium + '/' + g.r_num + '/'
# 作業用フォルダの設定
work_dir = '/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/'
# メディアフォルダの設定
media_dir = '/Users/trueocean/Desktop/Python_Code/PRISM_SCENE/Media_files/'

# csvとして保存
RaceInfo_df.to_csv(f'{work_dir}RaceInfo.csv', index=False, encoding="utf-8")
# PostgreSQLに保存
RaceInfo_df.to_sql('RaceInfo', con=engine, if_exists = 'replace', index=False)

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

if g.exe_opt in [1, 2, 6]:

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

    # 出走馬の頭数をグローバル変数に格納
    g.hr_num = len(RaceTable_df)

    # PRISM分析に必要となる3つの基本DFをPostgreSQLに保存
    RaceTable_df.to_sql('RaceTable', con=engine, if_exists = 'replace', index=False)
    HorseRecords_df.to_sql('HorseRecords', con=engine, if_exists = 'replace', index=False)
    Hanro_df.to_sql('Hanro', con=engine, if_exists = 'replace', index=False)
    CW_df.to_sql('CW', con=engine, if_exists = 'replace', index=False)

    # SCENE分析に必要となるDFをcsvファイルとして保存
    RaceTable_df.to_csv(f'{work_dir}RaceTable.csv', index=False, encoding="utf-8")
    HorseRecords_df.to_csv(f'{work_dir}HorseRecords.csv', index=False, encoding="utf-8")

    # アーカイブフォルダにコピー
    shutil.copy(f'{work_dir}RaceTable.csv', race_dir)
    shutil.copy(f'{work_dir}HorseRecords.csv', race_dir)

    print(Fore.YELLOW)
    print('PRISM_SCENE分析に必要なデータの整形が完了しました。')
    print(Style.RESET_ALL)


#====================================================
# PRISM分析の実行
#====================================================

if g.exe_opt in [2, 6]:

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
    PRISM_R_df.to_csv(f'{work_dir}PRISM_R.csv', index=False, encoding="utf-8")
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

    # 保存
    PRISM_RG_df.to_csv(f'{work_dir}PRISM_RG.csv', index=False, encoding="utf-8")
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
    PRISM_B_df.to_csv(f'{work_dir}PRISM_B.csv', index=False, encoding="utf-8")
    PRISM_B_df.to_sql('PRISM_B', con=engine, if_exists = 'replace', index=False)

    # PRISM_RGBの実行
    PRISM_RGB_df = PRISM_B.Calculate_PRISM_RGB(PRISM_RG_df, PRISM_B_df)
    PRISM_RGB_df.to_csv(f'{work_dir}PRISM_RGB.csv', index=False, encoding="utf-8")
    PRISM_RGB_df.to_sql('PRISM_RGB', con=engine, if_exists = 'replace', index=False)

    print(Fore.YELLOW)
    print('PRISM_B分析が完了しました。')

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
    # PRISMデータのアーカイブ
    #====================================================

    # 各画像データをアーカイブフォルダにコピー
    shutil.copy(f'{media_dir}PRISM_R.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_G.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_RG.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_B_Hanro.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_B_CW.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_B_Hanro_Time.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_B_Hanro_Lap.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_B_CW_Time.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_B_CW_Lap.png', race_dir)
    shutil.copy(f'{media_dir}PRISM_RGB.png', race_dir)

    # 各CSVファイルをアーカイブフォルダにコピー
    shutil.copy(f'{work_dir}PRISM_R.csv', race_dir)
    shutil.copy(f'{work_dir}PRISM_RG.csv', race_dir)
    shutil.copy(f'{work_dir}PRISM_B.csv', race_dir)
    shutil.copy(f'{work_dir}PRISM_RGB.csv', race_dir)


#====================================================
# SCENE分析（キャラ設定・ライバル分析）の実行
#====================================================

if g.exe_opt in [3, 6]:

    print(Fore.GREEN)
    print('====================================================')
    print('  SCENE分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('これより、SCENE分析を開始します。')


    #====================================================
    # SCENE_Script分析の実行
    #====================================================

    print(Fore.GREEN)
    print('====================================================')
    print('  SCENE_Script分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('SCENE_Script分析を実行しています。')

    # データの読み込み
    PRISM_R = pd.read_csv(f'{work_dir}PRISM_R.csv', encoding = 'utf-8')
    PRISM_RG = pd.read_csv(f'{work_dir}PRISM_RG.csv', encoding = 'utf-8')
    PRISM_B = pd.read_csv(f'{work_dir}PRISM_B.csv', encoding = 'utf-8')
    PRISM_RGB = pd.read_csv(f'{work_dir}PRISM_RGB.csv', encoding = 'utf-8')

    HorseRecords_df = pd.read_csv(f'{work_dir}HorseRecords.csv', encoding = 'utf-8')
    RaceTable_df = pd.read_csv(f'{work_dir}RaceTable.csv', encoding = 'utf-8')

    SCENE_Script_df = SCENE_Script.SCENE_Script(PRISM_R, PRISM_RG, PRISM_B, PRISM_RGB, HorseRecords_df, RaceTable_df)

    print(Fore.YELLOW)
    print('SCENE_Script分析が完了しました!')
    print(Style.RESET_ALL)


    #====================================================
    # SCENE_Cast分析の実行
    #====================================================

    print(Fore.GREEN)
    print('====================================================')
    print('  SCENE_Cast分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('SCENE_Cast分析を実行しています。')
    print('')

    Race_Info = f'{g.stadium} {g.clas} {g.td} {g.distance}m {g.race_name}'
    SCENE_Cast_df = SCENE_Script_df
    g.hr_num = len(RaceTable_df)

    # HTMLファイルから抽出した血統情報の格納
    blood_info_list = []
    for i in range(g.hr_num):
        blood_file = f'{work_dir}Blood{(i+1):02d}.html'
        file_text, used_encoding = SCENE_Cast.read_text_with_fallback(blood_file)
        cleaned_text = SCENE_Cast.clean_html_for_analysis(file_text) 
        blood_info_list.append(cleaned_text)
    SCENE_Cast_df['血統情報'] = blood_info_list

    # 各馬のキャラ設定を実行
    SCENE_Cast_df = SCENE_Cast.process_all_horses_parallel(SCENE_Cast_df, max_workers=8)

    # 結果をPostgreSQLに保存
    SCENE_Cast_df.to_sql('SCENE_Cast', con=engine, if_exists='replace', index=False)
    # csvファイルとして保存
    SCENE_Cast_df.to_csv(f'{media_dir}SCENE_Cast.csv', index=False, encoding="utf-8")

    # SCENE分析で生成したデータをアーカイブフォルダにコピー
    shutil.copy(f'{media_dir}SCENE_Cast.csv', race_dir)

    print(Fore.YELLOW)
    print('SCENE_Cast分析が完了しました!')
    print(Style.RESET_ALL)


    #====================================================
    # SCENE_Ensemble分析の実行
    #====================================================

    print(Fore.GREEN)
    print('====================================================')
    print('  SCENE_Ensemble分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('SCENE_Ensemble分析を実行しています。')
    print('')

    # ライバル関係の抽出
    SCENE_Ensemble_df = SCENE_Ensemble.SCENE_Ensemble_Analysis(HorseRecords_df)
    # 結果をPostgreSQLに保存
    SCENE_Ensemble_df.to_sql('SCENE_Ensemble', con=engine, if_exists = 'replace', index=False)
    # csvファイルとして保存
    SCENE_Ensemble_df.to_csv(f'{media_dir}SCENE_Ensemble.csv', index=False, encoding="utf-8")

    # SCENE分析で生成したデータをアーカイブフォルダにコピー
    shutil.copy(f'{media_dir}SCENE_Ensemble.csv', race_dir)

    print(Fore.YELLOW)
    print('SCENE_Ensemble分析が完了しました!')
    print(Style.RESET_ALL)


#====================================================
# SCENE分析（レース・シミュレーション）の実行
#====================================================

if g.exe_opt in [4, 6]:

    print(Fore.GREEN)
    print('====================================================')
    print('  SCENE分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('SCENE分析を実行しています。')
    print('')

    if g.exe_opt == 4:

        # データフレームの読み込み
        RaceTable_df = pd.read_csv(f'{work_dir}RaceTable.csv', encoding = 'utf-8')
        SCENE_Cast_df = pd.read_csv(f'{media_dir}SCENE_Cast.csv', encoding = 'utf-8')
        SCENE_Ensemble_df = pd.read_csv(f'{media_dir}SCENE_Ensemble.csv', encoding = 'utf-8')

    # バックグラウンドシミュレーション実行
    all_ranks = SCENE.run_background_simulation_parallel(SCENE_Cast_df, SCENE_Ensemble_df, RaceTable_df, client, MODEL)

    # 最終結果統合
    final_report = SCENE.run_final_aggregation(all_ranks, SCENE_Cast_df)
    final_report.to_sql('FinalReport', con=engine, if_exists = 'replace', index=False)
    final_report.to_csv(f'{media_dir}Final_Report.csv', index=False, encoding="utf-8")

    # 馬印の付与
    final_mark = SCENE.assign_race_marks_advanced(final_report, SCENE_Ensemble_df)
    final_mark = final_mark[final_mark['印'] != ""][['印', '枠番', '番', '馬名']]
    final_mark.to_sql('FinalMark', con=engine, if_exists = 'replace', index=False)
    final_mark.to_csv(f'{media_dir}Final_Mark.csv', index=False, encoding="utf-8")

    # SCENE分析で生成したデータをアーカイブフォルダにコピー
    shutil.copy(f'{media_dir}Final_Report.csv', race_dir)
    shutil.copy(f'{media_dir}Final_Mark.csv', race_dir)


#====================================================
# PRISM_SCENE分析（物語・レース実況生成）の実行
#====================================================

if g.exe_opt in [5, 6]:

    print(Fore.GREEN)
    print('====================================================')
    print('  PRISM_SCENE分析')
    print('====================================================')
    print(Style.RESET_ALL)
    print('PRISM_SCENE分析を実行しています。')
    print('')

    if g.exe_opt == 5:

        # データフレームの読み込み
        SCENE_Cast_df = pd.read_csv(f'{media_dir}SCENE_Cast.csv', encoding = 'utf-8')
        SCENE_Ensemble_df = pd.read_csv(f'{media_dir}SCENE_Ensemble.csv', encoding = 'utf-8')
        final_report = pd.read_csv(f'{media_dir}Final_Report.csv', encoding = 'utf-8')
        final_mark = pd.read_csv(f'{media_dir}Final_Mark.csv', encoding = 'utf-8')


    # ファイナル・ドラマ生成
    final_story = SCENE.generate_final_drama(SCENE_Cast_df, SCENE_Ensemble_df, final_report, final_mark, client, MODEL)
    save_drama_name = f'{media_dir}Final_drama.txt'
    SCENE.save_text_to_file(final_story, save_drama_name)

    # レース実況テキスト生成（クレンジング前）
    broadcast_script_draft = SCENE.generate_race_broadcast(final_story, final_report, client, MODEL)

    # 最終レース実況テキスト・音声の生成・保存
    mp3_name = f"{media_dir}Broadcast.mp3"
    broadcast_name = f'{media_dir}Broadcast.txt'
    broadcast_script = asyncio.run(SCENE.save_race_audio(broadcast_script_draft, mp3_name))
    SCENE.save_text_to_file(broadcast_script, broadcast_name)

    # SCENE分析で生成したデータをアーカイブフォルダにコピー
    shutil.copy(f'{media_dir}Final_Drama.txt', race_dir)
    shutil.copy(f'{media_dir}Broadcast.txt', race_dir)
    shutil.copy(f'{media_dir}Broadcast.mp3', race_dir)

if g.exe_opt in [3, 4]:
    print(Fore.RED)
    print('SCENE分析が完了しました!')
    print(Style.RESET_ALL)

if g.exe_opt in [5, 6]:
    print(Fore.RED)
    print('PRISM_SCENE分析が完了しました!')
    print(Style.RESET_ALL)


#====================================================
# レース実況オーディオの再生成の実行
#====================================================

if g.exe_opt == 7:

    print(Fore.GREEN)
    print('====================================================')
    print('  レース実況オーディオ再生成')
    print('====================================================')
    print(Style.RESET_ALL)

    # ファイルパスを指定
    file_path = f'{media_dir}Broadcast.txt'

    Audio_Text = Race_Audio_Maker.audio_text_getter(file_path)

    # 最終レース実況テキスト・音声の生成・保存
    mp3_name = f"{media_dir}Broadcast.mp3"
    broadcast_script = asyncio.run(Race_Audio_Maker.save_race_audio(Audio_Text, mp3_name))

    # SCENE分析で生成したデータをアーカイブフォルダにコピー
    shutil.copy(f'{media_dir}Broadcast.txt', race_dir)
    shutil.copy(f'{media_dir}Broadcast.mp3', race_dir)

    print('')

#====================================================
# アクチュアル・ドラマ、アフター・ストーリーの生成
#====================================================

if g.exe_opt == 8:

    print(Fore.GREEN)
    print('====================================================')
    print('  アクチュアル・ドラマ、アフター・ストーリー生成')
    print('====================================================')
    print(Fore.RED)
    input('対象のレース結果を、TFJVから作業フォルダに保存済みであることを確認して Enter >> ')
    print(Style.RESET_ALL)

    # データフレームの読み込み
    SCENE_Cast_df = pd.read_csv(f'{media_dir}SCENE_Cast.csv', encoding = 'utf-8')
    SCENE_Ensemble_df = pd.read_csv(f'{media_dir}SCENE_Ensemble.csv', encoding = 'utf-8')
    RaceResult_df = pd.read_csv(f'{work_dir}RaceResult.csv', encoding = 'cp932')

    # ファイナル・ドラマ生成
    actual_drama = After_Story_Extractor.Actual_Race_Projector(SCENE_Cast_df, SCENE_Ensemble_df, RaceResult_df, client, MODEL)
    save_drama_name = f'{media_dir}Actual_Drama.txt'
    SCENE.save_text_to_file(actual_drama, save_drama_name)

    # アフター・ストーリー生成
    after_story = After_Story_Extractor.After_Story_Extractor(actual_drama, RaceResult_df, SCENE_Cast_df, SCENE_Ensemble_df, client, MODEL)
    save_story_name = f'{media_dir}After_Story.txt'
    SCENE.save_text_to_file(after_story, save_story_name)

    # 生成したデータをアーカイブフォルダにコピー
    shutil.copy(f'{media_dir}Actual_Drama.txt', race_dir)
    shutil.copy(f'{media_dir}After_Story.txt', race_dir)
    shutil.copy(f'{work_dir}RaceResult.csv', race_dir)