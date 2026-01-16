#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
# メニュー画面の表示とレース情報の取得
#=============================================================

# ライブラリの準備
import os
import sys
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import time
import pandas as pd
import pyautogui as pgui

import warnings
# マッチするメッセージの警告を無視
warnings.filterwarnings("ignore", message=".*IMKCFRunLoopWakeUpReliable.*")

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from colorama import Fore, Back, Style

# モジュールの準備
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


# PRISM_SCENEメニュー画面のクラス
class RaceInfoApp:
    def __init__(self):
        # 初期データの読み込み
        self.load_default_data()
        
        # アプリウィンドウ設定
        self.app = Tk()
        self.app.attributes('-topmost', True)
        self.app.title('PRISM SCENE Analysis（プリズム・シーン分析） - ver.3.11')
        
        # Windows/Mac両方で見た目を安定させるためのスタイル設定
        self.style = ttk.Style()
        self.style.theme_use('classic')
        self.style.configure('Main.TFrame', background='#f0f0f0')
        self.style.configure('Pink.TLabel', foreground='#ff1493', font=('Helvetica', 10, 'bold'))
        self.style.configure('Action.TButton', font=('Helvetica', 10, 'bold'))

        # Tkinter変数の初期化
        self.init_variables()
        
        # ウィジェットの作成と配置
        self.create_widgets()
        

    def load_default_data(self):
        """初期表示データの読み込み"""
        try:
            path = '/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/RaceInfo.csv'
            df_Race_Info = pd.read_csv(path).fillna('')
            self.def_data = {
                'race_date': df_Race_Info.iat[0, 0],
                'stadium': df_Race_Info.iat[0, 1],
                'r_num': df_Race_Info.iat[0, 2],
                'age': '', 'clas': '', 'td': '', 'distance': '', 'cond': '', 'race_name': ''
            }
        except:
            self.def_data = {key: '' for key in ['race_date', 'stadium', 'r_num', 'age', 'clas', 'td', 'distance', 'cond', 'race_name']}

    def init_variables(self):
        """Tkinter変数の初期化"""
        self.var_race_date = StringVar(value=self.def_data['race_date'])
        self.var_stadium = StringVar(value=self.def_data['stadium'])
        self.var_r_num = StringVar(value=self.def_data['r_num'])
        self.var_race_name = StringVar()
        self.var_age = StringVar()
        self.var_clas = StringVar()
        self.var_td = StringVar()
        self.var_distance = StringVar()
        self.var_cond = StringVar()
        self.var_exe_opt = IntVar(value=6)
        self.scene_vars = {}

    def update_scene_state(self, *args):
        """分析オプションの値に応じてシーン選択の有効/無効を切り替える"""
        opt = self.var_exe_opt.get()
        # 5, 6, 8 のいずれかであれば 'normal'、それ以外は 'disabled'
        new_state = 'normal' if opt in [5, 6, 8] else 'disabled'
                
        # 全てのチェックボックスの状態を更新
        for cb in self.scene_checkboxes:
            cb.configure(state=new_state)

    def create_widgets(self):
            """画面レイアウトの構築"""
            # メインコンテナ
            container = ttk.Frame(self.app, padding="15 15 15 15")
            container.grid(row=0, column=0, sticky=(N, S, E, W))
            self.app.columnconfigure(0, weight=1)
            container.columnconfigure(0, weight=1)

            # --- Section 1: 基本情報 ---
            f1 = ttk.Labelframe(container, text=' レース基本情報 ', padding=10)
            f1.grid(row=0, column=0, sticky='ew', pady=5)
            for i in range(4): f1.columnconfigure(i, weight=1)

            ttk.Label(f1, text='日付', style='Pink.TLabel').grid(row=0, column=0, pady=2)
            ttk.Entry(f1, textvariable=self.var_race_date, width=12, justify='center').grid(row=1, column=0, padx=5)

            ttk.Label(f1, text='競馬場', style='Pink.TLabel').grid(row=0, column=1, pady=2)
            ttk.Combobox(f1, textvariable=self.var_stadium, width=6, values=['札幌','函館','福島','新潟','中山','東京','中京','京都','阪神','小倉'], justify='center').grid(row=1, column=1, padx=5)

            ttk.Label(f1, text='レース番号', style='Pink.TLabel').grid(row=0, column=2, pady=2)
            ttk.Combobox(f1, textvariable=self.var_r_num, width=5, values=[f'{i}R' for i in range(1, 13)], justify='center').grid(row=1, column=2, padx=5)

            ttk.Label(f1, text='レース名').grid(row=0, column=3, pady=2)
            ttk.Entry(f1, textvariable=self.var_race_name, width=15, justify='center').grid(row=1, column=3, padx=5)

            # --- Section 2: レース条件 ---
            f2 = ttk.Labelframe(container, text=' レース条件 ', padding=10)
            f2.grid(row=1, column=0, sticky='ew', pady=5)
            for i in range(5): f2.columnconfigure(i, weight=1)

            cond_labels = [('年齢限定', self.var_age, ['２歳','３歳','３上','４上']),
                        ('クラス', self.var_clas, ['Ｇ１','Ｇ２','Ｇ３', 'OP(L)', 'ｵｰﾌﾟﾝ','3勝','2勝','1勝', '未勝利']),
                        ('トラック', self.var_td, ['芝','ダ']),
                        ('距離', self.var_distance, ['1000','1150','1200','1300','1400','1500','1600','1700','1800','1900','2000','2100','2200','2300','2400','2500','2600','3000','3200','3400','3600']),
                        ('馬場状態', self.var_cond, ['良','稍','重','不'])]

            for i, (label, var, vals) in enumerate(cond_labels):
                l_style = 'Pink.TLabel' if label == '馬場状態' else 'TLabel'
                ttk.Label(f2, text=label, style=l_style).grid(row=0, column=i, pady=2)
                ttk.Combobox(f2, textvariable=var, values=vals, width=8, justify='center').grid(row=1, column=i, padx=3)

            # --- Section 3: 分析・実行オプション ---
            f3 = ttk.Labelframe(container, text=' 分析・実行オプション選択 ', padding=10)
            f3.grid(row=2, column=0, sticky='ew', pady=5)

            opts = [ ('1. データ取得のみ実行', 1), 
                    ('2. PRISM分析まで実行', 2), 
                    ('3. SCENE分析（キャラ設定・ライバル分析）のみ実行', 3), 
                    ('4. SCENE分析（レース・シミュレーション）のみ実行', 4), 
                    ('5. PRISM_SCENE分析（物語・レース実況オーディオ生成）のみ実行', 5), 
                    ('6. 全モジュール一括実行', 6),
                    ('7. レース実況オーディオ再生成（/Media Files/Broadcast.txtから）', 7),
                    ('8. アクチュアル・ドラマ と アフター・ストーリー生成', 8)]
            for i, (txt, val) in enumerate(opts):
                ttk.Radiobutton(f3, text=txt, value=val, variable=self.var_exe_opt, command=self.update_scene_state).grid(row=i, column=0, sticky=W, pady=1)

            # --- Section 4: シーン選択 ---
            # 3列表示のチェックボックスの場合
            self.f4 = ttk.Labelframe(container, text=' シーン選択 （5, 6, 8 実行時のみチェック）', padding=10)
            self.f4.grid(row=3, column=0, sticky='ew', pady=5)
            
            # 3列分の重みを設定
            for col in range(3):
                self.f4.columnconfigure(col, weight=1)


            scene_opts = [ ('スタート', 0), ('先行争い', 1), ('直線', 2), ('最初のコーナー', 3), 
                ('ホームストレッチ', 4), ('第1コーナー', 5), ('第2コーナー', 6), ('向正面', 7), 
                ('第3コーナー', 8), ('第4コーナー', 9), ('最終直線', 10), ('ゴール', 11), ('エピローグ', 12)]
                    
            # デフォルトでチェックを入れたいIDのリスト
            default_checked = [0, 8, 9, 10, 11, 12]
            self.scene_vars = {}
            self.scene_checkboxes = [] # ★ チェックボックスのウィジェットを保持するリスト

            for i, (txt, val) in enumerate(scene_opts):
                # 現在のvalがリストにあれば True、なければ False を初期値にする
                is_checked = True if val in default_checked else False
                var = tk.BooleanVar(value=is_checked) 
                self.scene_vars[val] = var
                
                # # 2列配置の計算
                # row_idx = i // 2
                # col_idx = i % 2
                
                # ttk.Checkbutton(f4, text=txt, variable=var).grid(row=row_idx, column=col_idx, sticky=tk.W, padx=20, pady=1)

                # 3列配置の計算
                row_idx = i // 3  # 3個ごとに次の行へ
                col_idx = i % 3   # 0, 1, 2 の繰り返し
                
                cb = ttk.Checkbutton(self.f4, text=txt, variable=var)
                cb.grid(row=row_idx, column=col_idx, sticky=tk.W, padx=10, pady=1)
                self.scene_checkboxes.append(cb) # リストに追加

            # ★ 初期化完了後に現在の選択状態を反映させる
            self.update_scene_state()


            # --- Section 5: 実行ボタン ---
            btn_frame = ttk.Frame(container, padding="0 10 0 0")
            btn_frame.grid(row=4, column=0, sticky='ew') # rowを4に変更
            btn_frame.columnconfigure(0, weight=1)
            btn_frame.columnconfigure(1, weight=1)

            ttk.Button(btn_frame, text='情報を更新', width=20, command=self.race_info_get).grid(row=0, column=0, padx=10, pady=10, sticky=E)
            ttk.Button(btn_frame, text='スタート', width=20, style='Action.TButton', command=self.analysis_start).grid(row=0, column=1, padx=10, pady=10, sticky=W)

    def race_info_get(self):
        """外部ファイルから情報を再読み込み"""
        path = f'/Users/trueocean/Desktop/PRISM_SCENE/Archive/{self.var_race_date.get()}/{self.var_stadium.get()}/{self.var_r_num.get()}/RaceInfo.csv'
        df = pd.read_csv(path).fillna('')
        self.var_age.set(df.iat[0, 3]); self.var_clas.set(df.iat[0, 4])
        self.var_td.set(df.iat[0, 5]); self.var_distance.set(df.iat[0, 6])
        self.var_cond.set(df.iat[0, 7]); self.var_race_name.set(df.iat[0, 8])

        # --- シーン選択の自動連動ロジック ---
        dist = int(df.iat[0, 6])
        
        # 全レース共通の必須シーン
        # 0:スタート, 10:最終直線, 11:ゴール, 12:エピローグ
        target_scenes = [0, 10, 11, 12]
        
        if dist == 1000:
            # 新潟千直
            target_scenes += [2]
        elif dist <= 1800:
            # 短距離〜マイル：コーナーが少ないためシンプルに
            target_scenes += [1, 8, 9] # 最初のコーナーと第3,第4コーナー
        elif dist <= 2000:
            # 中距離：向正面を含める
            target_scenes += [1, 3, 7, 8, 9] 
        elif dist <= 2400:
            # 中距離（2400m以上）：ほぼ全ての展開を描写
            target_scenes += [1, 5, 6, 7, 8, 9]
        else:
            # 中距離（2400m以上）：ほぼ全ての展開を描写
            target_scenes += [1, 3, 4, 5, 6, 7, 8, 9]

        # UIのチェックボックスを更新
        for val, var in self.scene_vars.items():
            if val in target_scenes:
                var.set(True)
            else:
                var.set(False)
        
        print(f"情報更新完了: {self.var_race_name.get()} (距離:{dist}m に合わせてシーンを最適化しました)")

    def analysis_start(self):
        # 入力値の取得
        race_date = self.var_race_date.get()
        stadium = self.var_stadium.get()
        r_num = self.var_r_num.get()
        race_name = self.var_race_name.get()
        td = self.var_td.get()
        distance_val = self.var_distance.get()
        age = self.var_age.get()
        clas = self.var_clas.get()
        cond = self.var_cond.get()

        # --- シーン選択の結果を取得 ---
        # シーンIDと名称の対応辞書
        scene_map = {
            0: 'スタート', 1: '先行争い', 2: '直線', 3: '最初のコーナー',
            4: 'ホームストレッチ', 5: '第1コーナー', 6: '第2コーナー',
            7: '向正面', 8: '第3コーナー', 9: '第4コーナー',
            10: '最終直線', 11: 'ゴール', 12: 'エピローグ'
        }
        # チェックされている名称だけをリスト化
        selected_scenes = [scene_map[val] for val, var in self.scene_vars.items() if var.get()]

        # 必須入力チェック
        required_fields = {
            "日付": race_date,
            "競馬場": stadium,
            "レース番号": r_num,
            "トラック": td,
            "距離": distance_val,
            "年齢限定": age,
            "クラス": clas,
            "馬場状態": cond
        }

        empty_fields = [label for label, val in required_fields.items() if not str(val).strip()]

        if empty_fields:
            print(Fore.RED + "\n[入力エラー] 以下の項目が未入力です。")
            print(f"対象項目: {', '.join(empty_fields)}")
            print("全ての情報を入力してから「分析スタート」を押してください。" + Style.RESET_ALL)
            return

        # シーンが一つも選ばれていない場合の警告
        if not selected_scenes:
            print(Fore.RED + "\n[入力エラー] シーンが一つも選択されていません。" + Style.RESET_ALL)
            return

        """終了処理とグローバル変数への代入"""
        g.race_date = race_date
        g.stadium = stadium
        g.r_num = r_num
        g.race_name = race_name
        g.td = td
        g.distance = int(distance_val)
        g.age = age
        g.clas = clas
        g.cond = cond
        g.exe_opt = self.var_exe_opt.get()
        
        # --- シーンリストをグローバル変数に格納 ---
        g.selected_scenes = selected_scenes
        
        g.filename = f"{g.race_date}{g.stadium}{g.r_num}"

        # --- ウィンドウ破棄処理 ---
        self.app.withdraw()
        self.app.update()
        self.app.destroy()
        self.app.quit()

    def run(self):
        
        # 左上の「×」ボタンが押された時の処理を登録
        # "WM_DELETE_WINDOW" はウィンドウの閉じるボタン（×）を指す
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)

        # まずウィンドウを非表示にする
        self.app.withdraw()
        
        # 座標を計算するために一度情報を更新
        self.app.update_idletasks()
        
        # 3. サイズを取得して中央座標を計算
        w = self.app.winfo_width()
        h = self.app.winfo_height()
        x = (self.app.winfo_screenwidth() // 2) - (w // 2)
        y = (self.app.winfo_screenheight() // 2) - (h // 2)
        
        # 座標を設定
        self.app.geometry(f'+{x}+{y}')
        
        # ここで初めて表示させる
        self.app.deiconify()
        
        # マウスを「情報を更新」ボタンに配置
        pgui.moveTo(650, 830)

        # メインループ開始
        self.app.mainloop()

        # mainloopを抜けた（quitが呼ばれた）直後に、外側から確実に破棄する
        try:
            self.app.destroy()
        except:
            pass

    def on_closing(self):
        """×ボタンが押された時に実行される処理"""
        print(Fore.YELLOW + "\n[中断] ユーザーによってウィンドウが閉じられました。プログラムを終了します。" + Style.RESET_ALL)
        print('')
        
        # Tkinterのリソースを安全に解放
        self.app.destroy()
        
        # プログラム全体を終了させる
        # これを入れないと、呼び出し元のメインコードが動き続けてしまいます
        os._exit(0)


def PRISM_SCENE_Menu():
    
    app = RaceInfoApp()
    app.run()

    # ここで念のため確実に消えるのを待つ
    time.sleep(1)


if __name__ == '__main__':

    print(Fore.GREEN)
    print('====================================================')
    print('                 PRISM_SCENE 分析')
    print('====================================================')
    print(Fore.YELLOW)
    print('ようこそ、PRISM_SCENE分析へ！')
    print(Style.RESET_ALL)
    print('これからPRISM_SCENE分析を実施します。')
    print('メニュー画面にレース情報をインプットしてください。')
    
    PRISM_SCENE_Menu()

    # データフレームにレース情報を格納
    col = ['日付', '競馬場', 'R番号', '年齢', 'クラス', 'TD', '距離', '状態', 'レース名']
    r_info = [[g.race_date, g.stadium, g.r_num, g.age, g.clas, g.td, g.distance, g.cond, g.race_name]]
    RaceInfo_df = pd.DataFrame(data = r_info, index = ['レース情報'], columns = col)

    # csvとして保存
    RaceInfo_df.to_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/RaceInfo.csv', index=False, encoding="utf-8")
    # postgreSQLに保存
    RaceInfo_df.to_sql('RaceInfo', con=engine, if_exists = 'replace')

    # シーン選択表示
    print(g.selected_scenes)

    print(Fore.YELLOW)
    print('今回のレース情報を取得しました。')
    print(Style.RESET_ALL)
