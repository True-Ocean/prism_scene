#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
# TFJVからのデータ取得
#=============================================================

# ライブラリの準備
import shutil
import pandas as  pd
import pyautogui as pgui
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
    """TARGET (TFJV.EXE) のウィンドウ位置とサイズを取得する"""
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    
    for window in window_list:
        owner_name = window.get('kCGWindowOwnerName', '')
        
        # アプリ名が TFJV.EXE のものを探す
        if 'TFJV.EXE' in owner_name:
            bounds = window['kCGWindowBounds']
            # x, y, width, height
            rect = (
                int(bounds['X']),
                int(bounds['Y']),
                int(bounds['Width']),
                int(bounds['Height'])
            )
            return rect
    return None

# TFJVウィンドウの座標取得
original_x, original_y, original_w, original_h = get_target_rect()
g.original_x = original_x
g.original_y = original_y
g.original_w = original_w
g.original_h = original_h


#====================================================
# ディスプレイモードによって座標辞書を選択
#====================================================

if g.mode == 'dual':
    TARGET_REL_POS = {
        "メニューバー|ファイル": (116, 33),
        "テキストファイル出力": (120, 56),
        "出馬表・★画面イメージ出力（CSV形式）": (681, 326),
        "ダイアログ|OKボタン": (1172, 398),
        "小ダイアログ|上書きボタン":(950, 565),
        "1番目の馬名": (489, 207),
        "★画面イメージCSV": (524, 350),
        "各馬実績画面|⬇︎": (408, 87),
        "メニューバー|閉じる": (270, 57),
        "血統表ボタン": (1135, 86),
        "血統表|出力ボタン": (1086, 86),
        "5大血統表・HTML形式出力": (1188, 114),
        "血統表|⬇︎": (425, 624),
        "調教一覧ボタン": (1057, 86),
        "坂路調教一覧": (1093, 112),
        "調教|全馬ボタン": (543, 181),
        "調教期間選択の▼": (482, 89),
        "坂路|365日追加": (470, 182),
        "調教|出力ボタン": (1243, 86),
        "ユーザー設定CSV": (1306, 115),
        "ウッドC調教一覧": (1093, 153),
        "VSCode": (1365, 1104)
}
    
else:
    TARGET_REL_POS = {
        "メニューバー|ファイル": (118, 33),
        "テキストファイル出力": (137, 58),
        "出馬表・★画面イメージ出力（CSV形式）": (734, 326),
        "ダイアログ|OKボタン": (967, 317),
        "小ダイアログ|上書きボタン":(748, 483),
        "1番目の馬名": (485, 207),
        "★画面イメージCSV": (490, 350),
        "各馬実績画面|⬇︎": (407, 86),
        "メニューバー|閉じる": (275, 56),
        "血統表ボタン": (1142, 82),
        "血統表|出力ボタン": (1085, 85),
        "5大血統表・HTML形式出力": (1138, 115),
        "血統表|⬇︎": (425, 624),
        "調教一覧ボタン": (1051, 83),
        "坂路調教一覧": (1100, 108),
        "調教|全馬ボタン": (546, 181),
        "調教期間選択の▼": (481, 83),
        "365日追加": (447, 178),
        "調教|出力ボタン": (1242, 84),
        "ユーザー設定CSV": (1307, 113),
        "ウッドC調教一覧": (1100, 150),
        "VSCode": (1365, 1104)
}


# 待ち時間の設定
click_time = 0.3
wait_time = 0.75
long_wait_time = 5


#====================================================
# 事前にマウス座標を登録した辞書に基づきシングルクリック
#====================================================

def smart_click(name):

    # 辞書から相対座標を取得
    rel_x, rel_y = TARGET_REL_POS[name]
        
    # 絶対座標を計算（ウィンドウの左上 + 相対距離）
    abs_x = g.original_x + rel_x
    abs_y = g.original_y + rel_y
    
    # クリック実行
    pgui.click(abs_x, abs_y, duration=click_time)


#====================================================
# 事前にマウス座標を登録した辞書に基づきダブルクリック
#====================================================

def smart_doubleclick(name):

    rel_x, rel_y = TARGET_REL_POS[name]
    
    # 絶対座標を計算（ウィンドウの左上 + 相対距離）
    abs_x = g.original_x + rel_x
    abs_y = g.original_y + rel_y
    
    # クリック実行
    pgui.doubleClick(abs_x, abs_y, duration=click_time)
    

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

# アーカイブフォルダがまだ存在していない場合、アーカイブフォルダを作成
if not os.path.exists(race_dir):
    os.makedirs(race_dir)

    input(f'TFJVのメイン画面で、{Fore.YELLOW}対象のレースの出馬表を開いていること{Style.RESET_ALL}を確認してください。>> Enter')
    print('')

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
    time.sleep(wait_time)
    smart_click('メニューバー|閉じる')

    focus_vscode()

    # 出馬表データ（ファイル名：Race_Table.csv）をアーカイブフォルダにコピー
    shutil.copy(f'{work_dir}RaceInfo.csv', race_dir)
    shutil.copy(f'{work_dir}RaceTable.csv', race_dir)
    # 出走頭数の取得
    df_race_table = pd.read_csv(f'{work_dir}RaceTable.csv', encoding = 'cp932')
    g.hr_num = len(df_race_table)
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
    df_race_table = pd.read_csv(f'{work_dir}RaceTable.csv', encoding = 'cp932')
    g.hr_num = len(df_race_table)
    for i in range(g.hr_num):
        shutil.copy(f'{race_dir}Uma{i+1}.csv', work_dir)
        shutil.copy(f'{race_dir}Blood{(i+1):02d}.html', work_dir)
    shutil.copy(f'{race_dir}Hanro.csv', work_dir)
    shutil.copy(f'{race_dir}CW.csv', work_dir)

    print(Fore.YELLOW)
    print('PRISM_SCENE分析に必要なデータをアーカイブフォルダから取得しました。')
    print(Style.RESET_ALL)
