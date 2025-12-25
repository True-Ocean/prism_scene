#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
# メニュー画面の表示とレース情報の取得
#=============================================================

# ライブラリの準備
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import time
import pandas as pd
import pyautogui as pgui
from colorama import Fore, Back, Style

# モジュールの準備
import My_Global as g

# PRISM_SCENEメニュー画面のクラス
class RaceInfoApp:
    def __init__(self):
        # 初期データの読み込み
        self.load_default_data()
        
        # アプリウィンドウ設定
        self.app = Tk()
        self.app.attributes('-topmost', True)
        self.app.title('PRISM SCENE Analysis（プリズム・シーン分析） - ver 1.01β')
        
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
        
        # マウスを「分析スタート」ボタン付近へ移動
        pgui.moveTo(x=200, y=550)

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
        self.var_exe_opt = IntVar(value=0)
        self.var_scr_opt = BooleanVar(value=True)

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
        # 横に並べるためにサブフレームを作成
        option_frame = ttk.Frame(container)
        option_frame.grid(row=2, column=0, sticky='ew', pady=5)
        option_frame.columnconfigure(0, weight=3)
        option_frame.columnconfigure(1, weight=1)

        f3 = ttk.Labelframe(option_frame, text=' 分析データ取得設定 ', padding=10)
        f3.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        opts = [('全データ取得', 0), ('馬体重・オッズ優先', 1), ('オッズのみ', 2), ('ローカルデータのみ', 3)]
        for i, (txt, val) in enumerate(opts):
            ttk.Radiobutton(f3, text=txt, value=val, variable=self.var_exe_opt).grid(row=i, column=0, sticky=W, pady=1)

        f4 = ttk.Labelframe(option_frame, text=' 環境設定 ', padding=10)
        f4.grid(row=0, column=1, sticky='nsew')
        ttk.Checkbutton(f4, text='デュアルモニター', variable=self.var_scr_opt).pack(anchor=W, pady=5)

        # --- Section 4: 実行ボタン ---
        btn_frame = ttk.Frame(container, padding="0 10 0 0")
        btn_frame.grid(row=3, column=0, sticky='ew')
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text='情報を更新', width=20, command=self.race_info_get).grid(row=0, column=0, padx=10, pady=10, sticky=E)
        ttk.Button(btn_frame, text='分析スタート', width=20, style='Action.TButton', command=self.analysis_start).grid(row=0, column=1, padx=10, pady=10, sticky=W)

    def race_info_get(self):
        """外部ファイルから情報を再読み込み"""
        path = f'/Users/trueocean/Desktop/PRISM_SCENE/Archive/{self.var_race_date.get()}/{self.var_stadium.get()}/{self.var_r_num.get()}/RaceInfo.csv'
        df = pd.read_csv(path).fillna('')
        self.var_age.set(df.iat[0, 3]); self.var_clas.set(df.iat[0, 4])
        self.var_td.set(df.iat[0, 5]); self.var_distance.set(df.iat[0, 6])
        self.var_cond.set(df.iat[0, 7]); self.var_race_name.set(df.iat[0, 8])

    def analysis_start(self):
        """終了処理とグローバル変数への代入"""
        g.race_date = self.var_race_date.get()
        g.stadium = self.var_stadium.get()
        g.r_num = self.var_r_num.get()
        g.race_name = self.var_race_name.get()
        g.td = self.var_td.get()
        g.distance = self.var_distance.get()
        g.age = self.var_age.get()
        g.clas = self.var_clas.get()
        g.cond = self.var_cond.get()
        g.exe_opt = self.var_exe_opt.get()
        g.scr_opt = self.var_scr_opt.get()
        g.filename = f"{g.race_date}{g.stadium}{g.r_num}"

        self.app.quit()
        self.app.withdraw()
        self.app.update_idletasks()
        self.app.destroy()
        time.sleep(0.3)

    def run(self):
            # 1. まずウィンドウを非表示にする
            self.app.withdraw()
            
            # 2. 座標を計算するために一度情報を更新
            self.app.update_idletasks()
            
            # 3. サイズを取得して中央座標を計算
            w = self.app.winfo_width()
            h = self.app.winfo_height()
            x = (self.app.winfo_screenwidth() // 2) - (w // 2)
            y = (self.app.winfo_screenheight() // 2) - (h // 2)
            
            # 4. 座標を設定
            self.app.geometry(f'+{x}+{y}')
            
            # 5. ここで初めて表示させる
            self.app.deiconify()
            
            self.app.mainloop()

def PRISM_SCENE_Menu():
    app = RaceInfoApp()
    app.run()

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
    df_Race_Info = pd.DataFrame(data = r_info, index = ['レース情報'], columns = col)

    # csvとして保存
    df_Race_Info.to_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/RaceInfo.csv', index=False, encoding="utf-8")

    print(Fore.YELLOW)
    print('今回のレース情報を取得しました。')
    print(Style.RESET_ALL)
