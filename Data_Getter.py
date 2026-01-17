#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
# TFJVからのデータ取得
#=============================================================

# ライブラリの準備
import shutil
import pandas as  pd
import pyautogui as pgui
pgui.FAILSAFE = True
import os
import time
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
from colorama import Fore, Back, Style

# モジュールの準備
import My_Global as g

#====================================================
# TFJVウィンドウ座標取得（Mac専用のウィンドウ取得関数）
#====================================================

def get_target_rect():
    
    while True:
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        
        target_window = None
        for window in window_list:
            owner_name = window.get('kCGWindowOwnerName', '')
            
            # アプリ名が TFJV.EXE のものを探す
            if 'TFJV.EXE' in owner_name:
                bounds = window['kCGWindowBounds']
                target_window = (
                    int(bounds['X']),
                    int(bounds['Y']),
                    int(bounds['Width']),
                    int(bounds['Height'])
                )
                break # 見つかったらループを抜ける
        
        if target_window:
            return target_window
        
        # --- 見つからなかった場合の処理 ---
        print(Fore.RED + "\n[Error] TFJV.EXEが見つかりません。" + Style.RESET_ALL)
        input("TFJVの画面を表示させてから、Enterキーを押してください（再試行します）...")

# TFJVウィンドウの座標取得
original_x, original_y, original_w, original_h = get_target_rect()
g.original_x = original_x
g.original_y = original_y
g.original_w = original_w
g.original_h = original_h

if g.original_x < 0:
    g.mode = 'dual'
else:
    g.mode = 'single'


#====================================================
# ディスプレイモードによって座標辞書を選択
#====================================================

if g.mode == 'dual':
    TARGET_REL_POS = {
        "メニューバー|ファイル": (-1014, -1010),
        "テキストファイル出力": (-991, -992),
        "出馬表・★画面イメージ出力（CSV形式）": (-399, -724),
        "ダイアログ|OKボタン": (48, -646),
        "小ダイアログ|上書きボタン":(-178, -486),
        "1番目の馬名": (-649, -841),
        "★画面イメージCSV": (-647, -697),
        "各馬実績画面|⬇︎": (-723, -961),
        "メニューバー|閉じる": (-858, -990),
        "血統表ボタン": (4, -961),
        "血統表|出力ボタン": (-44, -960),
        "5大血統表・HTML形式出力": (13, -936),
        "血統表|⬇︎": (-706, -426),
        "調教一覧ボタン": (-75, -962),
        "坂路調教一覧": (-54, -936),
        "調教|全馬ボタン": (-585, -868),
        "調教期間選択の▼": (-651, -962),
        "365日追加": (-670, -870),
        "調教|出力ボタン": (113, -963),
        "ユーザー設定CSV": (222, -937),
        "ウッドC調教一覧": (-53, -896),
        "VSCode": (284, 52)
}
    
else:
    TARGET_REL_POS = {
        "メニューバー|ファイル": (116, 68),
        "テキストファイル出力": (142, 94),
        "出馬表・★画面イメージ出力（CSV形式）": (728, 364),
        "ダイアログ|OKボタン": (968, 351),
        "小ダイアログ|上書きボタン":(753, 517),
        "1番目の馬名": (491, 242),
        "★画面イメージCSV": (466, 382),
        "各馬実績画面|⬇︎": (407, 120),
        "メニューバー|閉じる": (273, 90),
        "血統表ボタン": (1140, 119),
        "血統表|出力ボタン": (1085, 120),
        "5大血統表・HTML形式出力": (1149, 142),
        "血統表|⬇︎": (425, 655),
        "調教一覧ボタン": (1054, 119),
        "坂路調教一覧": (1064, 146),
        "調教|全馬ボタン": (544, 214),
        "調教期間選択の▼": (480, 122),
        "365日追加": (453, 213),
        "調教|出力ボタン": (1245, 121),
        "ユーザー設定CSV": (1286, 145),
        "ウッドC調教一覧": (1065, 183),
        "VSCode": (284, 52)
}


# 待ち時間の設定
click_time = 0.2
short_wait_time = 0.3
wait_time = 0.75
long_wait_time = 10


#====================================================
# 事前にマウス座標を登録した辞書に基づきシングルクリック
#====================================================

def smart_click(name):

    # 辞書から相対座標を取得
    abs_x, abs_y = TARGET_REL_POS[name]
    
    # クリック実行
    pgui.click(abs_x, abs_y, duration=click_time)

    time.sleep(short_wait_time)


#====================================================
# 事前にマウス座標を登録した辞書に基づきダブルクリック
#====================================================

def smart_doubleclick(name):

    abs_x, abs_y = TARGET_REL_POS[name]
        
    # クリック実行
    pgui.doubleClick(abs_x, abs_y, duration=click_time)

    time.sleep(short_wait_time)

    

#====================================================
# 画面切り替え（VSCode => TFJV）
#====================================================

def focus_target():
    os.system('osascript -e "tell application \\"TFJV.EXE\\" to activate"')
    time.sleep(wait_time)

# 画面切り替え（TFJV => VSCode）
def focus_vscode():
    os.system('osascript -e "tell application \\"Code\\" to activate"')
    time.sleep(wait_time)


#====================================================
# PRISM_SCENE分析に必要なデータの自動取得実行
#====================================================

def Data_Getter():

    print(Fore.GREEN)
    print('====================================================')
    print('  PRISM_SCENE分析に必要なデータの自動取得')
    print('====================================================')
    print(Style.RESET_ALL)
    print('PRISM_SCENE分析に必要なデータを自動取得します。')

    # アーカイブフォルダの設定
    race_dir = '/Users/trueocean/Desktop/PRISM_SCENE/Archive/' + g.race_date + '/' + g.stadium + '/' + g.r_num + '/'
    # 作業用フォルダの設定
    work_dir = '/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/'
    # メディアフォルダの設定
    media_dir = f'/Users/trueocean/Desktop/Python_Code/PRISM_SCENE/Media_files/'

    # アーカイブフォルダがまだ存在していない場合
    if not os.path.exists(race_dir):

        focus_vscode()
        input(f'TFJVのメイン画面で、{Fore.YELLOW}対象のレースの出馬表を開いていること{Style.RESET_ALL}を確認してください。>> Enter')
        print('')
        time.sleep(wait_time)
        focus_target()

        # TFJVウィンドウの座標取得
        get_target_rect()

        # TFJV画面にフォーカス
        focus_target()

        # 出馬表データ取得
        print('出馬表データ取得...')
        smart_click('メニューバー|ファイル')
        smart_click('テキストファイル出力')
        smart_click('出馬表・★画面イメージ出力（CSV形式）')
        pgui.typewrite('RaceTable.csv')
        smart_click('ダイアログ|OKボタン')
        smart_click('小ダイアログ|上書きボタン')
        print('')

        # 出走頭数の取得
        df_race_table = pd.read_csv(f'{work_dir}RaceTable.csv', encoding = 'cp932')
        g.hr_num = len(df_race_table)

        # # デバグ用（マウス自動操作を止める）
        # focus_vscode()
        # input('>>')
        time.sleep(wait_time)
        focus_target()

        # 各馬実績データ取得
        print('実績データ取得...')
        smart_click('1番目の馬名')
        smart_doubleclick('1番目の馬名')
        for i in range(g.hr_num):
            print(f' {i+1}頭目の実績データを取得しています。')
            smart_click('メニューバー|ファイル')
            smart_click('テキストファイル出力')
            smart_click('★画面イメージCSV')
            pgui.typewrite(f'Uma{i+1}.csv')
            smart_click('ダイアログ|OKボタン')
            smart_click('小ダイアログ|上書きボタン')
            if i != g.hr_num - 1:
                smart_click('各馬実績画面|⬇︎')
        smart_click('メニューバー|閉じる')
        print('')

        # 各馬血統データ取得
        print('血統データ取得...')
        smart_click('1番目の馬名')
        smart_click('血統表ボタン')
        for i in range(g.hr_num):
            print(f' {i+1}頭目の血統データを取得しています。')
            smart_click('血統表|出力ボタン')
            smart_click('5大血統表・HTML形式出力')
            pgui.typewrite(f'Blood{(i+1):02d}.html')
            smart_click('ダイアログ|OKボタン')
            smart_click('小ダイアログ|上書きボタン')
            if i != g.hr_num - 1:
                smart_click('血統表|⬇︎')
        smart_click('メニューバー|閉じる')
        print('')

        # 調教データ取得
        print('調教データ取得...')
        # 坂路調教データ取得
        print(' 坂路調教データを取得しています。')
        smart_click('調教一覧ボタン')
        smart_click('坂路調教一覧')
        smart_click('調教|全馬ボタン')
        smart_click('調教期間選択の▼')
        smart_click('365日追加')
        time.sleep(long_wait_time)
        smart_click('調教|出力ボタン')
        smart_click('ユーザー設定CSV')
        pgui.typewrite('Hanro.csv')
        smart_click('ダイアログ|OKボタン')
        smart_click('小ダイアログ|上書きボタン')
        time.sleep(wait_time)
        smart_click('メニューバー|閉じる')

        # CW調教データ取得
        print(' CW調教データを取得しています。')
        smart_click('調教一覧ボタン')
        smart_click('ウッドC調教一覧')
        smart_click('調教|全馬ボタン')
        smart_click('調教期間選択の▼')
        smart_click('365日追加')
        time.sleep(long_wait_time)
        smart_click('調教|出力ボタン')
        smart_click('ユーザー設定CSV')
        pgui.typewrite('CW.csv')
        smart_click('ダイアログ|OKボタン')
        smart_click('小ダイアログ|上書きボタン')
        time.sleep(wait_time * 2)
        smart_click('メニューバー|閉じる')
        time.sleep(wait_time * 2)

        # アーカイブフォルダを作成
        os.makedirs(race_dir)

        # 出馬表データ（ファイル名：Race_Table.csv）をアーカイブフォルダにコピー
        shutil.copy(f'{work_dir}RaceInfo.csv', race_dir)
        shutil.copy(f'{work_dir}RaceTable.csv', race_dir)

        # 各馬実績データと血統データを作業フォルダーにコピー
        for i in range(g.hr_num):
            shutil.copy(f'{work_dir}Uma{i+1}.csv', race_dir)
            shutil.copy(f'{work_dir}Blood{(i+1):02d}.html', race_dir)
        # 調教データと血統データを作業フォルダーにコピー    
        shutil.copy(f'{work_dir}Hanro.csv', race_dir)
        shutil.copy(f'{work_dir}CW.csv', race_dir)

        print(Fore.YELLOW)
        print('PRISM_SCENE分析に必要な全てのデータをTFJVから取得しました。')
        print(Style.RESET_ALL)

    # 既にアーカイブフォルダに保存済みデータがある場合    
    else:
        # アーカイブフォルダに保存されているファイルを作業用フォルダにコピー
        shutil.copy(f'{race_dir}RaceInfo.csv', work_dir)
        shutil.copy(f'{race_dir}RaceTable.csv', work_dir)
        # 出走頭数の取得
        try:
            # まずはcp932 (Shift_JIS系) で読み込み
            df_race_table = pd.read_csv(f'{work_dir}RaceTable.csv', encoding = 'cp932')
        except UnicodeDecodeError:
            # cp932 でエラーが出た場合、utf-8 で読み込み
            df_race_table = pd.read_csv(f'{work_dir}RaceTable.csv', encoding='utf-8')

        g.hr_num = len(df_race_table)
        for i in range(g.hr_num):
            shutil.copy(f'{race_dir}Uma{i+1}.csv', work_dir)
            shutil.copy(f'{race_dir}Blood{(i+1):02d}.html', work_dir)
        shutil.copy(f'{race_dir}Hanro.csv', work_dir)
        shutil.copy(f'{race_dir}CW.csv', work_dir)

        if g.exe_opt in [3, 4, 5, 7, 8]:
            
            # レースフォルダから作業フォルダにコピー
            shutil.copy(f'{race_dir}PRISM_R.csv', work_dir)
            shutil.copy(f'{race_dir}PRISM_RG.csv', work_dir)
            shutil.copy(f'{race_dir}PRISM_B.csv', work_dir)
            shutil.copy(f'{race_dir}PRISM_RGB.csv', work_dir)
            shutil.copy(f'{race_dir}HorseRecords.csv', work_dir)

            # レースフォルダからメディアフォルダにコピー
            shutil.copy(f'{race_dir}PRISM_R.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_G.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_RG.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_RGB.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_B_CW.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_B_CW_Lap.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_B_CW_Time.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_B_Hanro.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_B_Hanro_Lap.png', media_dir)
            shutil.copy(f'{race_dir}PRISM_B_Hanro_Time.png', media_dir)

        if g.exe_opt in [4, 5, 7, 8]:
            # レースフォルダからメディアフォルダにコピー
            shutil.copy(f'{race_dir}SCENE_Cast.csv', media_dir)
            shutil.copy(f'{race_dir}SCENE_Ensemble.csv', media_dir)
        
        if g.exe_opt in [5, 7, 8]:
            # レースフォルダからメディアフォルダにコピー
            shutil.copy(f'{race_dir}Final_Report.csv', media_dir)
            shutil.copy(f'{race_dir}Final_Mark.csv', media_dir)
        
        if g.exe_opt in [7, 8]:
            shutil.copy(f'{race_dir}Final_Drama.txt', media_dir)
            shutil.copy(f'{race_dir}Broadcast.txt', media_dir)
            shutil.copy(f'{race_dir}Broadcast.mp3', media_dir)

        print(Fore.YELLOW)
        print('PRISM_SCENE分析に必要なデータをアーカイブフォルダから取得しました。')
        print(Style.RESET_ALL)
    
    focus_vscode()


if __name__ == '__main__':
    Data_Getter()