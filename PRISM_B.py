#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
# PRISM_B分析
#=============================================================

#====================================================
# PRISM_B分析の準備
#====================================================

# ライブラリの準備
import os
import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from dotenv import load_dotenv
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

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


#====================================================
# PRISM_B分析の準備
#====================================================

# 調教データ読み込み
df_hanro = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/Hanro.csv', encoding = 'cp932')
df_cw = pd.read_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/CW.csv', encoding = 'cp932')

# 坂路調教データの整形
df_hanro = df_hanro.rename(columns = {'馬番/仮番(出馬表モード時のみ)':'番'})
df_hanro['番'] = df_hanro['番'].astype(int)
df_hanro = df_hanro[['年月日', '時刻', '番','馬名', '性別', '年齢', '調教師','Time1','Time2','Time3','Time4','Lap4','Lap3','Lap2','Lap1']]

# CW調教データの整形
df_cw = df_cw.rename(columns = {'馬番/仮番(出馬表モード時のみ)':'番'})
df_cw['番'] = df_cw['番'].astype(int)
df_cw = df_cw[['年月日', '時刻', '番','馬名', '性別', '年齢', '調教師','所属', '回り', '10F', '9F', '8F', '7F', '6F', '5F', '4F', '3F', '2F', '1F',
               'Lap9', 'Lap8', 'Lap7', 'Lap6', 'Lap5', 'Lap4', 'Lap3', 'Lap2', 'Lap1',]]


#====================================================
# PRISM_B分析（調教データによる成長・劣化分析）
#====================================================

def PRISM_B_Analysis(race_table_df, horse_records_df, df_cw, df_hanro, target_race_date):
    cw = df_cw.copy()
    hanro = df_hanro.copy()
    records = horse_records_df.copy()

    def safe_to_datetime(series):
        return pd.to_datetime(series.astype(str).str.replace(r'\.0$', '', regex=True), errors='coerce')

    target_date = pd.to_datetime(target_race_date)
    records['datetime'] = safe_to_datetime(records['日付'])
    cw['datetime'] = safe_to_datetime(cw['年月日'])
    hanro['datetime'] = safe_to_datetime(hanro['年月日'])

    for df in [cw, hanro, records, race_table_df]:
        df['馬名'] = df['馬名'].astype(str).str.strip()

    for df, cols in [(cw, ['6F', 'Lap1', 'Lap2']), (hanro, ['Time1', 'Lap1', 'Lap2'])]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    target_horses = race_table_df['馬名'].unique()
    results = []
    REST_THRESHOLD_DAYS = 70

    for horse in target_horses:
        past_history = records[(records['馬名'] == horse) & (records['datetime'] < target_date)].sort_values('datetime', ascending=False)
        row_data = {'馬名': horse}

        if not past_history.empty:
            last_race_date = past_history.iloc[0]['datetime']
            interval_days = (target_date - last_race_date).days
            row_data['中何週'] = interval_days // 7
            hist_dates = sorted(past_history['datetime'].unique(), reverse=True)
            analysis_dates = [target_date] + hist_dates
        else:
            row_data['中何週'] = np.nan
            analysis_dates = [target_date]

        labels = ["今回", "前走", "前々走"]
        for i, label in enumerate(labels):
            if i < len(analysis_dates):
                ref_date = analysis_dates[i]
                start_date = ref_date - pd.Timedelta(days=21)
                
                for mode, df, time_col in [('CW', cw, '6F'), ('坂路', hanro, 'Time1')]:
                    t_df = df[(df['馬名'] == horse) & (df['datetime'] >= start_date) & (df['datetime'] < ref_date)].copy()
                    if not t_df.empty:
                        valid = t_df[t_df[time_col] > 0].sort_values(time_col).head(3)
                        if not valid.empty:
                            row_data[f'{label}_{mode}_最良'] = valid[time_col].min()
                            row_data[f'{label}_{mode}_L1平均'] = valid['Lap1'].mean()
                            # 最大加速幅を取得
                            accel_series = valid['Lap2'] - valid['Lap1']
                            row_data[f'{label}_{mode}_最大加速'] = accel_series.max()

        results.append(row_data)
    
    return pd.DataFrame(results)

#====================================================
# PRISM_BからPRISM_RGBへの統合（最終偏差値算出コード）
#====================================================

def Calculate_PRISM_RGB(prism_rg_df, training_stats_df):
    df_b = training_stats_df.copy()
    
    # --- 指標算出 ---
    for mode in ['CW', '坂路']:
        # 1. 末脚のキレ補正 (上限 1.5)
        prev_L1 = df_b[[f'前走_{mode}_L1平均', f'前々走_{mode}_L1平均']].mean(axis=1)
        df_b[f'{mode}_キレ補正'] = (prev_L1 - df_b[f'今回_{mode}_L1平均']).clip(lower=0, upper=1.5)
        
        # 2. 最大加速補正 (上限 1.5)
        df_b[f'{mode}_加速補正'] = df_b[f'今回_{mode}_最大加速'].clip(lower=0, upper=1.5)
        
        # 3. 時計更新補正 (上限 2.0)
        prev_best = df_b[[f'前走_{mode}_最良', f'前々走_{mode}_最良']].min(axis=1)
        df_b[f'{mode}_時計補正'] = (prev_best - df_b[f'今回_{mode}_最良']).clip(lower=0, upper=2.0)

    # 調教データが全くない馬を判定するためのフラグ
    # (今回調教の「最良」が CW/坂路 ともに NaN ならデータなしとみなす)
    df_b['has_data'] = df_b[['今回_CW_最良', '今回_坂路_最良']].notna().any(axis=1)

    # スコア合算
    df_b['PRISM_B_Score'] = (
        (df_b['CW_キレ補正'].fillna(0) + df_b['坂路_キレ補正'].fillna(0)) * 1.0 +
        (df_b['CW_加速補正'].fillna(0) + df_b['坂路_加速補正'].fillna(0)) * 1.2 +
        (df_b['CW_時計補正'].fillna(0) + df_b['坂路_時計補正'].fillna(0)) * 0.2
    )

    # --- 平均値による補完ロジック ---
    # 1. データがある馬だけの平均スコアを算出
    avg_score = df_b.loc[df_b['has_data'] == True, 'PRISM_B_Score'].mean()
    # もし全馬データなしという極端な状況なら 0 にする
    if pd.isna(avg_score): avg_score = 0.0
    
    # 2. データがない馬(has_data == False)に平均値を代入
    df_b.loc[df_b['has_data'] == False, 'PRISM_B_Score'] = avg_score
    
    # 3. 最終スコアのキャップ（最大3.5程度に抑えてRGとの主従を維持）
    df_b['PRISM_B_Score'] = df_b['PRISM_B_Score'].clip(upper=3.5)

    # --- RGB統合 ---
    merged_df = pd.merge(prism_rg_df, df_b[['馬名', 'PRISM_B_Score']], on='馬名', how='left')
    
    # 万が一結合でNaNが出た場合も平均値で埋める
    merged_df['PRISM_B_Score'] = merged_df['PRISM_B_Score'].fillna(avg_score)
    
    # 最終偏差値算出
    merged_df['PRISM_RGB_Score'] = merged_df['PRISM_RG_Score'] + merged_df['PRISM_B_Score']
    
    return merged_df.sort_values('PRISM_RGB_Score', ascending=False)


#====================================================
# PRISM_Bのビジュアル化
#====================================================

def PRISM_B_Visualization(prism_b_df, race_table_df):
    """
    path_effectsを使用して、重なりに強い（縁取り文字）ビジュアル化を行います。
    """

    # Mac用フォント設定
    plt.rcParams['font.family'] = ['Hiragino Sans']
    plt.rcParams['axes.unicode_minus'] = False

    df = prism_b_df.copy()
    
    # 馬番マッピング
    if '馬番' in race_table_df.columns:
        mapping = race_table_df.set_index('馬名')['馬番'].to_dict()
    else:
        mapping = {name: i+1 for i, name in enumerate(df['馬名'])}

    # 分析指標の計算
    for mode in ['CW', '坂路']:
        prev_l1 = df[[f'前走_{mode}_L1平均', f'前々走_{mode}_L1平均']].mean(axis=1)
        df[f'{mode}_キレ変化'] = (prev_l1 - df[f'今回_{mode}_L1平均'])
        prev_acc = df[[f'前走_{mode}_最大加速', f'前々走_{mode}_最大加速']].mean(axis=1)
        df[f'{mode}_加速変化'] = (df[f'今回_{mode}_最大加速'] - prev_acc)

    modes = [('CW', 'royalblue'), ('坂路', 'forestgreen')]
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 縁取りスタイルの定義（白縁に黒文字）
    pe = [path_effects.withStroke(linewidth=3, foreground="white")]

    for i, (mode, color) in enumerate(modes):
        ax = axes[i]
        m_df = df[df[f'今回_{mode}_最良'].notna()].copy()
        
        if m_df.empty:
            ax.set_title(f"{mode}：データなし")
            continue

        # 1. 散布図
        ax.scatter(m_df[f'{mode}_キレ変化'], m_df[f'{mode}_加速変化'], 
                   s=300, color=color, alpha=0.7, edgecolors='black', zorder=3)

        # 2. 馬名ラベル（path_effectsを適用）
        for _, row in m_df.iterrows():
            num = int(mapping.get(row['馬名'], 0))
            label = f"{num} {row['馬名']}"
            
            # 簡易的な重なり回避：馬番の奇数/偶数でラベルを上下に振り分ける
            offset = 0.05 if num % 2 == 0 else -0.08
            va = 'bottom' if num % 2 == 0 else 'top'
            
            ax.text(row[f'{mode}_キレ変化'], 
                    row[f'{mode}_加速変化'] + offset, 
                    label, 
                    fontsize=11, 
                    fontweight='bold',
                    ha='center', 
                    va=va,
                    path_effects=pe, # 縁取りを適用
                    zorder=4)

        # 十字のガイドライン
        ax.axhline(0, color='gray', lw=1.5, ls='--', zorder=1)
        ax.axvline(0, color='gray', lw=1.5, ls='--', zorder=1)
        
        # 領域ラベル
        ax.text(0.98, 0.98, '【成長・充実】', transform=ax.transAxes, 
                fontsize=18, color='red', alpha=0.6, fontweight='bold',
                ha='right', va='top', path_effects=pe)
        
        ax.text(0.02, 0.02, '【劣化・停滞】', transform=ax.transAxes, 
                fontsize=18, color='blue', alpha=0.6, fontweight='bold',
                ha='left', va='bottom', path_effects=pe)

        ax.set_title(f"【{mode}】調教パフォーマンス変化分析", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("末脚の鋭さ（L1タイム）の更新秒数 ※右ほど良化", fontsize=13)
        ax.set_ylabel("加速の持続性（L2-L1）の更新秒数 ※上ほど良化", fontsize=13)
        ax.grid(True, alpha=0.2, zorder=0)

        # 軸範囲の自動調整
        padding = 0.4
        ax.set_xlim(m_df[f'{mode}_キレ変化'].min() - padding, m_df[f'{mode}_キレ変化'].max() + padding)
        ax.set_ylim(m_df[f'{mode}_加速変化'].min() - padding, m_df[f'{mode}_加速変化'].max() + padding)

    plt.tight_layout()
    plt.savefig('./Media_files/PRISM_B.png', dpi=150, bbox_inches='tight')
    # plt.show()

    return df


#====================================================
# PRISM_RGBのビジュアル化
#====================================================

def PRISM_RGB_Visualization(prism_rgb_df):
    """
    PRISM_R, PRISM_RG, PRISM_RGB の3指標を横一列に並べて表示
    """
    # Mac環境向けのフォント設定
    plt.rcParams['font.family'] = ['Hiragino Sans']
    
    df = prism_rgb_df.copy()
    # グラフ表示用のラベル作成（馬番 + 馬名）
    if '番' in df.columns:
        df['表示名'] = df['番'].astype(str) + ' ' + df['馬名']
    else:
        df['表示名'] = df['馬名']

    # 1行3列のレイアウト設定
    fig, axes = plt.subplots(1, 3, figsize=(25, 12), layout="constrained")

    # タイトルの設定（gオブジェクトが定義されている前提）
    try:
        race_title = f"{g.stadium} {g.distance}m ({g.cond})"
    except NameError:
        race_title = "対象レース"
        
    fig.suptitle(f"PRISM 最終統合分析：{race_title}\n(R:地力 / RG:適性補正 / RGB:調教・成長補正)", 
                 fontsize=28, fontweight='bold')

    # 可視化する設定：(カラム名, タイトル, 色)
    plot_configs = [
        ('PRISM_R_Score',   '① PRISM_R：原能力', 'crimson'),
        ('PRISM_RG_Score',  '② PRISM_RG：条件補正後', 'forestgreen'),
        ('PRISM_RGB_Score', '③ PRISM_RGB：最終偏差値', 'royalblue')
    ]

    # X軸の最大値を統一して比較しやすくする
    max_val = df[['PRISM_R_Score', 'PRISM_RG_Score', 'PRISM_RGB_Score']].max().max() * 1.15

    for idx, (col, title, color) in enumerate(plot_configs):
        ax = axes[idx]
        
        # 各グラフごとにその指標の降順で並び替え
        df_sub = df.sort_values(col, ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(df_sub))
        
        bars = ax.barh(y_pos, df_sub[col], color=color, alpha=0.7)
        
        # サブタイトルの設定
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        # Y軸（馬名）の設定
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sub['表示名'], fontsize=12)
        
        # X軸（偏差値）の設定
        ax.set_xlim(0, max_val)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

        # バーの右端に数値を表示
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max_val * 0.01), bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}', va='center', ha='left', fontsize=11, fontweight='bold')

    # 画像保存
    plt.savefig("./Media_files/PRISM_RGB.png", bbox_inches='tight')
    # plt.show()


#====================================================
# 調教データのビジュアル化
#====================================================

# Mac用フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

def Horse_Training_Visualization(race_table_df, df_cw, df_hanro, target_race_date):
    horse_names = race_table_df['馬名'].str.strip().unique()
    
    # --- 1. 横軸（期間）の設定 ---
    race_dt = pd.to_datetime(target_race_date)
    fixed_start = race_dt - pd.Timedelta(days=14)
    fixed_end = race_dt - pd.Timedelta(days=1)
    date_range = pd.date_range(fixed_start, fixed_end)
    date_to_idx = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(date_range)}
    date_labels = [d.strftime('%m/%d') for d in date_range]

    # --- 2. データフィルタリング関数の定義 ---
    def get_filtered_data(df, is_cw=False):
        tmp = df.copy()
        # 日付変換
        tmp['dt_eval'] = pd.to_datetime(tmp['年月日'].astype(str).str.replace(r'\.0$', '', regex=True), errors='coerce')
        
        if is_cw:
            time_col = '6F'
            lap_cols = [f'Lap{i}' for i in range(1, 7)]
            time_limit = 100
        else:
            time_col = 'Time1'
            lap_cols = [f'Lap{i}' for i in range(1, 5)]
            time_limit = 80
        
        # 期間フィルタ
        tmp = tmp[(tmp['dt_eval'] >= fixed_start) & (tmp['dt_eval'] <= fixed_end)].copy()
        
        # 数値変換
        tmp[time_col] = pd.to_numeric(tmp[time_col], errors='coerce')
        for lc in lap_cols:
            tmp[lc] = pd.to_numeric(tmp[lc], errors='coerce')
        
        # 条件除外：全体時計
        tmp = tmp[tmp[time_col] <= time_limit]
        
        # 【修正】各ラップを20秒以下に制限
        for lc in lap_cols:
            tmp = tmp[tmp[lc] <= 20]
            
        return tmp.dropna(subset=[time_col] + lap_cols)

    # フィルタリング済みデータの取得
    h_period = get_filtered_data(df_hanro, is_cw=False)
    c_period = get_filtered_data(df_cw, is_cw=True)

    # 統計・レンジ算出
    def get_stats(df, time_col, lap_cols):
        if df.empty: return [], [], None, None
        times = df[time_col].tolist()
        lap1s = df['Lap1'].tolist()
        return times, lap1s, np.mean(times), np.mean(lap1s)

    h_times, _, h_time_avg, h_lap1_avg = get_stats(h_period, 'Time1', [f'Lap{i}' for i in range(1, 5)])
    c_times, _, c_time_avg, c_lap1_avg = get_stats(c_period, '6F', [f'Lap{i}' for i in range(1, 7)])

    # レンジ決定
    h_t_range = (min(h_times)-0.5 if h_times else 45, 80)
    c_t_range = (min(c_times)-1.0 if c_times else 60, 100)
    # ラップレンジは10秒〜20秒に固定
    lap_range = (10, 20)

    categories = [
        {'title': '坂路：全体(4F)時計推移', 'type': 'h_time', 'filename': './Media_files/PRISM_B_Hanro_Time.png', 'y_range': h_t_range, 'x_type': 'date', 'avg': h_time_avg},
        {'title': '坂路：Best3_Lap構成推移', 'type': 'h_lap', 'filename': './Media_files/PRISM_B_Hanro_Lap.png', 'y_range': lap_range, 'x_type': 'lap4', 'avg': h_lap1_avg},
        {'title': 'CW：全体(6F)時計推移', 'type': 'c_time', 'filename': './Media_files/PRISM_B_CW_Time.png', 'y_range': c_t_range, 'x_type': 'date', 'avg': c_time_avg},
        {'title': 'CW：Best3_Lap構成推移', 'type': 'c_lap', 'filename': './Media_files/PRISM_B_CW_Lap.png', 'y_range': lap_range, 'x_type': 'lap6', 'avg': c_lap1_avg}
    ]

    for cat in categories:
        num_horses = len(horse_names)
        cols = 3 
        rows = (num_horses + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), layout="constrained")
        fig.suptitle(f"【{target_race_date}直前14日間】 {cat['title']}", fontsize=24, fontweight='bold')
        axes_flat = axes.flatten() if num_horses > 1 else [axes]

        for i, horse in enumerate(horse_names):
            ax = axes_flat[i]
            data = h_period[h_period['馬名'] == horse].copy() if cat['type'].startswith('h') else c_period[c_period['馬名'] == horse].copy()
            
            if not data.empty:
                data['dt_str'] = data['dt_eval'].dt.strftime('%Y-%m-%d')
                
                if cat['type'] in ['h_time', 'c_time']:
                    val_col = 'Time1' if cat['type'] == 'h_time' else '6F'
                    d = data.sort_values('dt_eval')
                    x_vals = [date_to_idx[ds] for ds in d['dt_str'] if ds in date_to_idx]
                    y_vals = [v for ds, v in zip(d['dt_str'], d[val_col]) if ds in date_to_idx]
                    ax.plot(x_vals, y_vals, marker='o', color='royalblue', lw=2)

                elif cat['type'] == 'h_lap':
                    top3 = data.sort_values('Time1').head(3)
                    for _, row in top3.iterrows():
                        laps = [row[f'Lap{k}'] for k in range(4, 0, -1)]
                        ax.plot([0, 1, 2, 3], laps, marker='s', label=row['dt_eval'].strftime('%m/%d'))

                elif cat['type'] == 'c_lap':
                    top3 = data.sort_values('6F').head(3)
                    for _, row in top3.iterrows():
                        laps = [row[f'Lap{k}'] for k in range(6, 0, -1)]
                        ax.plot(range(6), laps, marker='s', label=row['dt_eval'].strftime('%m/%d'))

            if cat['avg'] is not None:
                ax.axhline(y=cat['avg'], color='red', linestyle='--', alpha=0.6, label='全体平均')

            # 【修正】Y軸は反転せず、通常通り（下小→上大）
            ax.set_ylim(cat['y_range'][0], cat['y_range'][1])
            
            if cat['x_type'] == 'date':
                ax.set_xlim(-0.5, 13.5)
                ax.set_xticks(range(0, 14, 2))
                ax.set_xticklabels(date_labels[::2], rotation=45)
            elif cat['x_type'] == 'lap4':
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels(['Lap4', 'Lap3', 'Lap2', 'Lap1'])
            elif cat['x_type'] == 'lap6':
                ax.set_xticks(range(6))
                ax.set_xticklabels([f'Lap{k}' for k in range(6, 0, -1)])
            
            ax.set_title(f"【{horse}】", fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')

        for j in range(i + 1, len(axes_flat)): fig.delaxes(axes_flat[j])
        plt.savefig(cat['filename'], bbox_inches='tight', dpi=150)
    
    return True


#====================================================
# PRISM_B分析の実行
#====================================================

if __name__ == "__main__":

    HorseRecords_df = pd.read_sql(sql = f'SELECT * FROM "HorseRecords";', con=engine)
    RaceTable_df = pd.read_sql(sql = f'SELECT * FROM "RaceTable";', con=engine)

    # PRISM_Bの実行
    PRISM_B_df = PRISM_B_Analysis(RaceTable_df, HorseRecords_df, df_cw, df_hanro, g.race_date)
    PRISM_B_df.to_sql('PRISM_B', con=engine, if_exists = 'replace')

    # PRISM_RGの読み込み
    PRISM_RG_df = pd.read_sql(sql = f'SELECT * FROM "PRISM_RG";', con=engine)

    # PRISM_RGBの実行
    PRISM_RGB_df = Calculate_PRISM_RGB(PRISM_RG_df, PRISM_B_df)
    PRISM_RGB_df.to_sql('PRISM_RGB', con=engine, if_exists = 'replace')

    # PRISM_Bのビジュアル化実行
    PRISM_B_Visualization(PRISM_B_df, RaceTable_df)
    
    # PRISM_RGBのビジュアル化実行
    PRISM_RGB_Visualization(PRISM_RGB_df)

    # 調教データのビジュアル化実行
    Horse_Training_Visualization(RaceTable_df, df_cw, df_hanro, g.race_date)