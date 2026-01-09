#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# SCENE分析（Systematic Character Extraction for Narrative Epilogue）
#=============================================================

#====================================================
# レース実況オーディオ生成
#====================================================


# ライブラリの準備
import os
from colorama import Fore, Back, Style
import shutil
import asyncio

# モジュールの準備
import My_Global as g
from SCENE import save_race_audio

def audio_text_getter(file_path):

    try:
        # 'r'は読み込みモード、encodingは日本語が含まれる場合は 'utf-8' を推奨
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("指定されたファイルが見つかりませんでした。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    
    return text

#====================================================
# 実行
#====================================================

if __name__ == "__main__":

    # ファイルパスを指定
    save_dir_path = '/Users/trueocean/Desktop/Python_Code/PRISM_SCENE/Media_files/'
    file_path = f'{save_dir_path}Broadcast.txt'

    Audio_Text = audio_text_getter(file_path)

    # 最終レース実況テキスト・音声の生成・保存
    mp3_name = f"{save_dir_path}Broadcast.mp3"
    broadcast_script = asyncio.run(save_race_audio(Audio_Text, mp3_name))

    # アーカイブフォルダの設定
    race_dir = '/Users/trueocean/Desktop/PRISM_SCENE/Archive/' + g.race_date + '/' + g.stadium + '/' + g.r_num + '/'
    # 作業用フォルダの設定
    work_dir = save_dir_path

    # SCENE分析で生成したデータをアーカイブフォルダにコピー
    shutil.copy(f'{work_dir}Broadcast.txt', race_dir)
    shutil.copy(f'{work_dir}Broadcast.mp3', race_dir)