#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
# PRISM_R分析
#=============================================================

#====================================================
# PRISM_R分析の準備
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
# PRISM_Base分析
#====================================================

def PRISM_Base(engine, horse_records_df, base_weight=58.0):
    """
    基準タイムマスターを使用して絶対偏差値を算出する
    """
    # 1. DBから基準データを読み込む
    standards_df = pd.read_sql('SELECT * FROM race_standards', con=engine)
    
    # 2. 分析対象馬のデータと基準データを結合
    # 走破タイム0を除外（海外レース等でデータがない場合の実績を除外）し、今回のレースのTD（芝かダート）でフィルタリング
    horse_records_df = horse_records_df[horse_records_df['走破タイム'] > 0]
    horse_records_df = horse_records_df[horse_records_df['TD'] == g.td]

    pdf = horse_records_df.copy()
    pdf = pd.merge(pdf, standards_df, on=['場所', 'TD', '距離', 'クラス名'], how='left')
    
    # 3. 世代指数（山型ロジック）
    def get_generation_index(age):
        if age <= 2: return 20
        if age == 3: return 35
        if age == 4: return 45
        if age == 5: return 44
        if age == 6: return 40
        return 35
    pdf['世代指数'] = pdf['年齢'].apply(get_generation_index)

    # --- 【新規】上がり3Fの全体基準を算出 ---
    # 全データ（過去20年の芝G1出走馬）の上がり3Fから、偏差値計算用の定数を作成
    # 理想は条件別ですが、まずは全体の分布を基準にします
    all_3f_mean = pdf['上り3F'].replace(0, np.nan).mean()
    all_3f_std = pdf['上り3F'].replace(0, np.nan).std()

    # 4. 絶対偏差値の算出
    pdf['PRISM偏差値'] = (
        (pdf['平均タイム'] - pdf['走破タイム']) / pdf['標準偏差'].replace(0, np.nan) * 10 + 50
    ).fillna(50)

    # --- 【新規】上がりの絶対偏差値を算出 ---
    # 速い（数値が小さい）ほど偏差値が高くなるように計算
    pdf['上がり偏差値'] = (
        50 - ((pdf['上り3F'] - all_3f_mean) / (all_3f_std if all_3f_std > 0 else 1.0) * 10)
    ).fillna(50)
    
    # 5. 斤量：1kg差 ＝ 2.0（タイム補正 + 物理的な斤量補正）
    pdf['斤量補正'] = (pdf['斤量'] - base_weight) * 2.0 
    
    # 世代：ピーク(45)との差を補正
    pdf['世代補正'] = (45 - pdf['世代指数']) * 0.5
    
    # グレード補正ロジック
    def calculate_grade_bonus(row):
        class_name = str(row['クラス名'])
        rank = row['着順']
        if rank > 3: return 0.0
        
        if 'Ｇ１' in class_name:
            return 4.0 if rank == 1 else (2.0 if rank == 2 else 1.0)
        elif 'Ｇ２' in class_name:
            return 1.5 if rank == 1 else (1.0 if rank == 2 else 0.5)
        elif 'Ｇ３' in class_name:
            return 0.6 if rank == 1 else (0.4 if rank == 2 else 0.2)
        return 0.0

    pdf['グレード補正値'] = pdf.apply(calculate_grade_bonus, axis=1)
    
    # 6. 最終偏差値の合算
    pdf['最終補正偏差値'] = (
        pdf['PRISM偏差値'] + 
        pdf['斤量補正'] + 
        pdf['世代補正'] + 
        pdf['グレード補正値']
    )

    PRISM_Base_df = pdf

    return PRISM_Base_df

#====================================================
# PRISM_R分析
#====================================================

def PRISM_R_Analysis(prism_base_df, race_table_df):
    final_results = []
    ref_date = pd.to_datetime(g.race_date)

    # 安定化のため、xを1/1000にして計算
    def quadratic_func(x, a, b, c):
        return a * (x**2) + b * x + c

    def fit_with_regularization(x_data, y_data, weights):
        def objective(params):
            a, b, c = params
            y_pred = quadratic_func(x_data, a, b, c)
            error = np.sum(weights * (y_data - y_pred)**2)
            penalty = 10000 * (a**2)
            return error + penalty

        from scipy.optimize import minimize
        res = minimize(objective, x0=[-0.1, 0.5, 50.0], 
                       bounds=[(-10.0, 0), (-np.inf, np.inf), (0, 100)])
        return res.x

    # --- '番' カラムと '年齢' の情報をマッピング ---
    if isinstance(race_table_df, pd.DataFrame):
        # データフレームから情報を抽出
        horse_info = race_table_df[['馬名', '番', '年齢']].to_dict('records')
    else:
        # リスト形式の場合（※'番'がない可能性への考慮）
        horse_list = race_table_df
        # prism_base_dfから最新の年齢を取得
        age_map = prism_base_df.groupby('馬名')['年齢'].max().to_dict()
        horse_info = [{'馬名': h, '番': 0, '年齢': age_map.get(h, 0)} for h in horse_list]

    # horse_info（辞書のリスト）をループ
    for info in horse_info:
        horse = info['馬名']
        horse_num = info['番']
        current_age = info['年齢']
        
        h_data = prism_base_df[prism_base_df['馬名'] == horse].copy()
        if h_data.empty: continue

        # --- 外れ値のパージ ---
        h_data = h_data[h_data['最終補正偏差値'] >= 30.0].copy()
        if len(h_data) < 2:
            h_data = prism_base_df[prism_base_df['馬名'] == horse].copy()

        # --- 距離の正規化 ---
        h_data['dist_scaled'] = h_data['距離'] / 1000.0
        target_x = g.distance / 1000.0

        # --- 時間重みとスランプ補正 ---
        h_data['日付'] = pd.to_datetime(h_data['日付'])
        h_data['days_diff'] = (ref_date - h_data['日付']).dt.days
        h_data['weight'] = np.exp(-h_data['days_diff'] / 350) 

        recent_2 = h_data.sort_values('日付', ascending=False).head(2)
        avg_rank = recent_2['着順'].mean()
        
        if avg_rank > 10:
            h_data.loc[h_data['days_diff'] > 180, '最終補正偏差値'] *= 0.95
            h_data['weight'] *= 0.3
        elif avg_rank > 9:
            h_data.loc[h_data['days_diff'] > 180, '最終補正偏差値'] *= 0.92
            h_data['weight'] *= 0.2
        elif avg_rank > 6:
            h_data.loc[h_data['days_diff'] > 180, '最終補正偏差値'] *= 0.96

        # --- 形状制約付き回帰 ---
        if len(h_data) >= 3:
            try:
                popt = fit_with_regularization(
                    h_data['dist_scaled'].values, 
                    h_data['最終補正偏差値'].values, 
                    h_data['weight'].values
                )
                raw_score = quadratic_func(target_x, *popt)
            except:
                raw_score = h_data['最終補正偏差値'].mean()
        else:
            raw_score = h_data['最終補正偏差値'].mean()

        # --- 上限設定とペナルティ ---
        bonus = 0.0
        if avg_rank <= 5:
            if current_age <= 3: bonus = 2.5
            elif current_age <= 5: bonus = 1.0        

        recent_3y = h_data[h_data['days_diff'] <= 1095]
        estimated_score = min(raw_score, recent_3y['最終補正偏差値'].max() + bonus) if not recent_3y.empty else raw_score

        # --- 下方修正とクラス基準 ---
        if avg_rank > 9:
            estimated_score = max(raw_score * 0.94 - 1.0, 42.0)
        elif avg_rank > 6:
            estimated_score = max(raw_score * 0.98, 46.0)
        else:
            estimated_score = max(raw_score, 48.0)

        # 距離経験ペナルティ
        exp_filter = (h_data['days_diff'] <= 1095) & (h_data['距離'].between(g.distance-250, g.distance+250))
        if h_data[exp_filter].empty:
            estimated_score -= 1.0

        # --- 脚質判定 & EPI ---
        for col in ['1角', '2角', '3角', '4角']:
            h_data[col] = pd.to_numeric(h_data[col], errors='coerce').replace(0, np.nan)
        
        p_list = [(h_data[c] / h_data['出走頭数'].replace(0, np.nan)).mean() for c in ['1角', '2角', '3角', '4角']]
        p1, p2, p3, p4 = p_list

        if np.all(np.isnan(p_list)):
            style, epi = "不明", 0.50
        else:
            p12_val = np.nanmean([p1, p2]) if not (np.isnan(p1) and np.isnan(p2)) else (p3 if not np.isnan(p3) else p4)
            p12_val = np.nan_to_num(p12_val, nan=0.5)
            p3_val = p3 if not np.isnan(p3) else p4
            p4_val = p4 if not np.isnan(p4) else p3_val
            
            if p12_val > 0.6 and p4_val < 0.35: style = "ﾏｸﾘ"
            elif p12_val < 0.15: style = "逃げ"
            elif p12_val <= 0.4: style = "先行"
            elif 0.4 < p12_val <= 0.7 and p4_val < p12_val: style = "差し"
            else: style = "追込"
            epi = round(1 - ((p12_val * 0.6) + (p3_val * 0.2) + (p4_val * 0.2)), 2)

        # --- 上がり偏差値 ---
        h_3f_data = h_data[h_data['上り3F'] > 0]
        if not h_3f_data.empty:
            valid_3f = h_3f_data
            w_mean_3f = np.sum(valid_3f['上り3F'] * valid_3f['weight']) / valid_3f['weight'].sum()
            last3f_final = max(30.0, min(70.0, 50 + (34.8 - w_mean_3f) * 5))
        else:
            last3f_final = 50.0

        # --- 最終期待値 ---
        base_s = estimated_score
        f3f_s = last3f_final
        f3f_gap = f3f_s - 50
        
        if style in ["差し", "追込", "ﾏｸﾘ"]:
            expectancy = base_s + (f3f_gap * 0.4)
        elif style == "逃げ":
            adj = 0.2 if f3f_gap > 0 else 0.1
            expectancy = base_s + (f3f_gap * adj)
        else:
            expectancy = base_s + (f3f_gap * 0.3)

        if current_age >= 7: expectancy -= 0.5
        elif current_age == 3: expectancy += 0.5

        # --- '番' を辞書に追加 ---
        final_results.append({
            '馬名': horse,
            '番': horse_num,  # ← ここに追加
            'Base_Score': round(base_s, 2),
            'Last3F_Score': round(f3f_s, 2),
            'PRISM_R_Score': round(expectancy, 2),
            'EPI': epi,
            '脚質': style,
            '年齢': int(current_age),
            '安定度': round(h_data['最終補正偏差値'].std(), 2) if len(h_data) > 1 else 0.0
        })
    
    return pd.DataFrame(final_results)


#====================================================
# PRISM_R分析結果のビジュアル化
#====================================================

# Mac用の日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False 

def PRISM_R_Visualization(df):
    
    plt.figure(figsize=(14, 9))
    sns.set_style("whitegrid", {'font.family': 'Hiragino Sans'})
    
    # 脚質ごとに色を明示的に指定
    custom_palette = {
        '逃げ': '#FF4B00', '先行': '#F6AA00', '差し': '#03AF7A',
        '追込': '#005AFF', 'ﾏｸﾘ': '#9370DB', '不明': '#C0C0C0'
    }
    order = ['逃げ', '先行', '差し', '追込', 'ﾏｸﾘ', '不明']
    custom_markers = {
        '逃げ': '>', '先行': 'o', '差し': 's', '追込': 'D', 'ﾏｸﾘ': '<', '不明': 'X'
    }

    # 散布図のプロット
    scatter = sns.scatterplot(
        data=df, 
        x='EPI', 
        y='PRISM_R_Score', 
        hue='脚質', 
        style='脚質', 
        markers=custom_markers,
        s=120, 
        palette=custom_palette,
        hue_order=order,
        style_order=order,
        edgecolor='black',
        alpha=0.8
    )
    
    # 各点に「番 + 馬名」を表示
    for i in range(df.shape[0]):
        label_text = f"{int(df.番[i])} {df.馬名[i]}"
        
        txt = plt.text(
            df.EPI[i] + 0.005, 
            df.PRISM_R_Score[i] + 0.15, 
            label_text, 
            fontsize=9, 
            ha='left',
            va='bottom'
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white', alpha=0.8)])
    
    plt.title(f"PRISM_R 分析  ( {g.race_date} {g.stadium} {g.td} {g.distance}m  {g.race_name} )", fontsize=16)
    plt.xlabel("EPI ( 先行指数： 左 <= 後 、右 => 前 )", fontsize=12)
    plt.ylabel("馬本来の基礎能力 (偏差値)", fontsize=12)
    
    # --- 平均線の描画とテキスト追加 ---
    r_mean = df['PRISM_R_Score'].mean()
    # labelを削除して凡例に載らないようにする
    plt.axhline(r_mean, color='red', linestyle='--', alpha=0.4)
    plt.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
    
    # グラフ上の平均線付近に数値を表示
    x_max = df['EPI'].max()
    txt_mean = plt.text(
        x_max + 0.02, # グラフの右端付近に配置
        r_mean, 
        f'平均基礎能力:{r_mean:.1f}', 
        color='red', 
        fontsize=10, 
        fontweight='bold',
        va='center', 
        ha='left'
    )
    txt_mean.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white', alpha=0.8)])
        
    # 凡例の設定（平均能力を除外）
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        borderaxespad=0,
        fontsize=9,
        markerscale=1.2,
        labelspacing=1.5,
        handletextpad=1.2,
        borderpad=1.0,
        title_fontsize=10,
        frameon=True,
        edgecolor='gray'
    )

    plt.tight_layout()

    # 画像保存
    plt.savefig('./Media_files/PRISM_R.png', bbox_inches='tight', dpi=150)


#====================================================
# PRISM_R分析の実行
#====================================================

if __name__ == "__main__":

    RaceTable_df = pd.read_sql(sql = f'SELECT * FROM "RaceTable";', con=engine)

    # 馬名リストの取得
    target_horses = RaceTable_df['馬名'].tolist()

    # 対象馬の過去成績をDBから取得
    query = f"SELECT * FROM \"HorseRecords\" WHERE \"馬名\" IN ({str(target_horses)[1:-1]})"
    horse_records_all = pd.read_sql(query, con=engine)

    # PRISM_Base：基礎偏差値の算出
    PRISM_Base_df = PRISM_Base(engine, horse_records_all)

    # PRISM_R分析の実行
    PRISM_R_df = PRISM_R_Analysis(PRISM_Base_df, RaceTable_df,)

    # csvとして保存
    PRISM_R_df.to_csv('/Users/trueocean/Desktop/PRISM_SCENE/TFJV_Data/PRISM_R.csv', index=False, encoding="utf-8")

    # PostgreSQLへの保存
    PRISM_R_df.to_sql('PRISM_R', con=engine, if_exists='replace', index=False)
    

    # 実行
    PRISM_R_Visualization(PRISM_R_df)
