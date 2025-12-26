#=============================================================
# プリズム・シーン理論（PRISM-SCENE Theory）
# PRISM分析（Performance Rating and Intelligent Scoring Model）
# PRISM_G分析
#=============================================================

#====================================================
# PRISM_G分析の準備
#====================================================

# ライブラリの準備
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import norm
from scipy.stats import norm
from dotenv import load_dotenv
from sqlalchemy import create_engine

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
# PRISM_G分析の準備
#====================================================

# 競馬場毎に馬場状態別のRPCIシフト量と基準偏差を算出

def RPCI_Shift_Analysis(MasterDataset_df):
    """
    競馬場ごとに馬場状態別のRPCIシフト量と基準偏差を算出
    """
    # 当該競馬場の芝コースデータを抽出
    place_data = MasterDataset_df[
        (MasterDataset_df['場所'] == g.stadium) & 
        (MasterDataset_df['TD'] == g.td) & 
        (MasterDataset_df['RPCI'] > 0)
    ].copy()
    
    # 距離ごとの「良馬場」統計
    ryo_all_dist = place_data[place_data['馬場状態'] == '良']
    dist_stats = ryo_all_dist.groupby('距離')['RPCI'].agg(['median', 'std', 'count'])
    
    valid_stds = dist_stats[dist_stats['count'] >= 2]['std']
    intrinsic_baselines = {
        'median': ryo_all_dist['RPCI'].median(),
        'std': valid_stds.median() if not valid_stds.empty else 4.0
    }
    
    distance_medians = dist_stats['median'].to_dict()
    place_data['RPCI_Diff'] = place_data.apply(
        lambda x: x['RPCI'] - distance_medians.get(x['距離'], np.nan), axis=1
    )
    
    track_summary = place_data.dropna(subset=['RPCI_Diff']).groupby('馬場状態').agg({
        'RPCI_Diff': ['count', 'mean', 'median', 'std']
    }).reset_index()
    track_summary.columns = ['馬場状態', 'サンプル数', '平均シフト', '中央値シフト', '標準偏差']
    
    return track_summary, intrinsic_baselines


#====================================================
# PRISM_G分析
#====================================================

def PRISM_G_Analysis(prism_r_df, MasterDataset_df, race_table_df, track_summary, intrinsic_baselines):
    """
    PRISM_G 最適化版: 展開シミュレーションと環境補正の統合
    """
    
    # 1. 舞台設定（良馬場基準）の特定
    course_ryo = MasterDataset_df[
        (MasterDataset_df['場所'] == g.stadium) & 
        (MasterDataset_df['距離'] == g.distance) & 
        (MasterDataset_df['TD'] == g.td) &
        (MasterDataset_df['馬場状態'] == g.cond)
    ]
    
    b_median = course_ryo['RPCI'].median() if not course_ryo.empty else intrinsic_baselines['median']
    b_std = course_ryo['RPCI'].std() if len(course_ryo) > 1 else intrinsic_baselines['std']

    # 2. 馬場状態シフトの適用
    cond_stats = track_summary[track_summary['馬場状態'] == g.cond]
    s_val = cond_stats['中央値シフト'].values[0] if not cond_stats.empty else 0.0
    s_mult = (cond_stats['標準偏差'].values[0] / intrinsic_baselines['std']) if (not cond_stats.empty and not pd.isna(cond_stats['標準偏差'].values[0])) else 1.0

    # レース想定RPCI分布
    current_rpci_median = b_median + s_val
    current_rpci_std = b_std * s_mult
    
    # 逃げ先行馬の頭数による動的ペースシフト
    high_epi_count = prism_r_df[prism_r_df['EPI'] >= 0.75].shape[0]
    current_rpci_median += (high_epi_count - 2) * -0.5 

    # 3. シミュレーション範囲設定
    rpci_range_full = np.arange(current_rpci_median - (current_rpci_std * 3.0), current_rpci_median + (current_rpci_std * 3.0), 0.1)
    rpci_range_real = np.arange(current_rpci_median - (current_rpci_std * 1.0), current_rpci_median + (current_rpci_std * 1.0), 0.1)
    
    prism_g_results = []

    for _, horse in prism_r_df.iterrows():
        name = horse['馬名']
        r_score = horse['PRISM_R_Score']
        hist = MasterDataset_df[MasterDataset_df['馬名'] == name]
        
        # 個体適性(PCI)の算出
        is_imputed = False
        # 修正：データが0件、または有効な統計が取れない（1件のみ）場合を考慮
        if len(hist) <= 1 or (hist['PCI'] == 0).all():
            # 1件あるが不十分な場合、その1件のPCIを参考にしつつ、
            # 標準偏差(sigma)は安全側に大きく見積もる（または補完モードにする）
            if len(hist) == 1:
                # 1走だけデータがある場合：そのPCIを目標にしつつ、sigmaは広めに設定
                ideal_pci = hist['PCI'].iloc[0]
                sigma = 4.0  # 補完時と同じ広めの範囲
                is_imputed = True # ガードレールを適用するためにTrueにする
            else:
                # 0件の場合
                ideal_pci, sigma, is_imputed = current_rpci_median, 4.0, True
        else:
            # 2件以上ある場合（通常ロジック）
            top_runs = hist.nsmallest(min(5, len(hist)), '着順')
            ideal_pci = top_runs['PCI'].mean()
            raw_sigma = top_runs['PCI'].std()
            
            # ここでも念のため、stdがNaN（2走だが1走分しかPCIがない等）の場合をガード
            sigma = np.clip(raw_sigma if pd.notnull(raw_sigma) else 3.0, 1.5, 5.0)

        # 適合度シミュレーション
        weights_full = norm.pdf(rpci_range_full, current_rpci_median, current_rpci_std)
        responses_full = np.exp(- (rpci_range_full - ideal_pci)**2 / (2 * sigma**2))
        g_avg_raw = np.sum(responses_full * weights_full) / np.sum(weights_full)

        responses_real = np.exp(- (rpci_range_real - ideal_pci)**2 / (2 * sigma**2))
        g_peak_raw = max(g_avg_raw, np.max(responses_real))
        g_risk_raw = min(g_avg_raw, np.min(responses_real))

        # ガードレール（Rスコア連動型）
        g_floor = np.clip(0.60 + (r_score / 250), 0.65, 0.85)
        g_ceiling = 0.95 if is_imputed else np.clip(1.02 - (sigma * 0.02), 0.88, 0.98)

        g_avg = np.clip(g_avg_raw, g_floor, g_ceiling)
        g_peak = np.clip(g_peak_raw, g_floor, g_ceiling)
        g_risk = np.clip(g_risk_raw, g_floor, g_ceiling)
        if is_imputed: g_avg = max(g_avg, 0.85)

        prism_g_results.append({'馬名': name, 'G_Avg': g_avg, 'G_Peak': g_peak, 'G_Risk': g_risk})

    # 4. 枠順・脚質の統合
    df_g = pd.DataFrame(prism_g_results).merge(race_table_df[['馬名', '枠番', '番']], on='馬名')
    df_g = df_g.merge(prism_r_df[['馬名', 'PRISM_R_Score', 'EPI']], on='馬名')
    
    for i, row in df_g.iterrows():
        gate_factor = 1.0
        if row['枠番'] <= 3: gate_factor = 1.02 if row['EPI'] >= 0.5 else 0.98
        elif row['枠番'] >= 7: gate_factor = 0.98 if row['EPI'] >= 0.5 else 1.02
        
        df_g.at[i, 'G_Avg'] *= gate_factor
        df_g.at[i, 'G_Peak'] *= max(gate_factor, 1.0)
        df_g.at[i, 'G_Risk'] *= min(gate_factor, 1.0)

    df_g['PRISM_RG_Score'] = df_g['PRISM_R_Score'] * df_g['G_Avg']

    # データフレームの最終化
    PRISM_RG_df = df_g

    return PRISM_RG_df


#====================================================
# PRISM_G分析結果のビジュアル化
#====================================================

def PRISM_G_Visualization(PRISM_RG_df):
    """
    各馬の能力(R)と適合(G)をプロット。縦軸を%表記に変換。
    """
    # 描画用データのコピーと変換
    plot_df = PRISM_RG_df.copy()
    plot_df['G_Avg_Percent'] = plot_df['G_Avg'] * 100  # 1.0 -> 100%
    
    # 日本語フォント設定
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 散布図のプロット
    scatter = sns.scatterplot(
        data=plot_df,
        x='PRISM_R_Score',
        y='G_Avg_Percent',
        size='PRISM_RG_Score',
        hue='PRISM_RG_Score',
        palette='viridis',
        sizes=(100, 600),
        alpha=0.7,
        ax=ax
    )
    
    # 馬名ラベルの追加（例：1 ジャスティンパレス）
    for i in range(plot_df.shape[0]):
        ax.text(
            plot_df.PRISM_R_Score.iloc[i], 
            plot_df.G_Avg_Percent.iloc[i] + 0.3, # 少し上にずらす
            f"{int(plot_df.番.iloc[i])} {plot_df.馬名.iloc[i]}",
            fontsize=8,
            #fontweight='bold',
            ha='center',
            va='bottom'
        )

    # 凡例のカスタマイズ
    plt.legend(
        title='PRISM_RG スコア',
        bbox_to_anchor=(1.05, 1), # グラフの外側に配置
        loc='upper left',
        fontsize=9,               # 文字を小さく
        title_fontsize=10,        # タイトルも少し小さく
        labelspacing=1.2,         # 行間を広げてゆったりさせる
        borderpad=1.0,            # 枠内の余白を広げる
        frameon=True,             # 枠線を表示
        edgecolor='gray'          # 枠線の色
    )    

    # 軸の設定
    ax.yaxis.set_major_formatter(mtick.PercentFormatter()) # 単位を%に
    
    # 平均値の計算
    r_mean = plot_df['PRISM_R_Score'].mean()
    g_mean = plot_df['G_Avg_Percent'].mean()

    # --- 平均線の描画と数値ラベルの追加 ---
    # 横軸の平均線 (垂直線)
    ax.axvline(r_mean, color='red', linestyle='--', alpha=0.4)
    ax.text(r_mean, ax.get_ylim()[1], f'平均基礎能力:{r_mean:.1f}', 
            color='red', fontsize=10, ha='center', va='bottom', fontweight='bold')

    # 縦軸の平均線 (水平線)
    ax.axhline(g_mean, color='blue', linestyle='--', alpha=0.4)
    ax.text(ax.get_xlim()[1], g_mean, f'平均適合率:{g_mean:.1f}%', 
            color='blue', fontsize=10, ha='left', va='center', fontweight='bold')
    
    # グラフの装飾
    plt.title(f'PRISM_G 分析  ( {g.race_date} {g.stadium} {g.td} {g.distance}m  {g.race_name} )', fontsize=16, pad=20)
    plt.xlabel('基礎能力（PRISM_R スコア）', fontsize=12)
    plt.ylabel('レース条件 (馬場状態・枠順・脚質・展開)適合率  %', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    
    # グラフ右側の余白を確保（凡例が切れないように）
    plt.subplots_adjust(right=0.8)    
    
    # # 例：能力スコアを40-70、適合度を70%-100%で固定する場合
    # plt.xlim(40, 70)
    # plt.ylim(75, 90)

    # 自動スケーリング
    plt.tight_layout()

    plt.savefig("./Media_files/PRISM_G.png", bbox_inches='tight') # 余白を自動調整して保存

    # plt.show()


#====================================================
# PRISM_RG のビジュアル化
#====================================================

def PRISM_RG_Visualization(prism_rg_df):
    """
    ダッシュボード可視化：タイトル重なり防止修正版
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
    
    df = prism_rg_df.copy()
    df['表示名'] = df['番'].astype(str) + ' ' + df['馬名']
    df['Peak_Score'] = df['PRISM_R_Score'] * df['G_Peak']
    df['Risk_Score'] = df['PRISM_R_Score'] * df['G_Risk']
    
    # layout="constrained" を使用（現在の推奨記法）
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), layout="constrained")

    race_title = f"{g.stadium} {g.td} {g.distance}m {g.race_name} ({g.cond})"
    
    # y座標の指定を消し、fontsizeを調整。
    # layout="constrained" が自動でグラフを下げてタイトル用のスペースを作ります。
    fig.suptitle(f"PRISM_RG 統合分析ダッシュボード：{race_title}", fontsize=26, fontweight='bold')

    plot_configs = [
        ('PRISM_R_Score', '① PRISM_R：原能力', 'crimson', 0, 0),
        ('PRISM_RG_Score',      '② PRISM_G：期待値', 'forestgreen',    0, 1),
        ('Peak_Score',    '③ 上振れ (Peak)',   'orange',  1, 0),
        ('Risk_Score',    '④ 下振れ (Risk)',   'teal',   1, 1)
    ]

    max_val = df[['PRISM_R_Score', 'Peak_Score']].max().max() * 1.15

    for col, title, color, row_idx, col_idx in plot_configs:
        ax = axes[row_idx, col_idx]
        df_sub = df.sort_values(col, ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(df_sub))
        
        bars = ax.barh(y_pos, df_sub[col], color=color, alpha=0.7)
        
        # サブタイトルの重なり防止のため pad（余白）を少し入れる
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20, 
                     color='crimson' if row_idx == 0 and col_idx == 0 else 'black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sub['表示名'], fontsize=12)
        ax.set_xlim(0, max_val)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max_val * 0.01), bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}', va='center', ha='left', fontsize=11, fontweight='bold')

    plt.savefig("./Media_files/PRISM_RG.png", bbox_inches='tight') # 余白を自動調整して保存
    # plt.show()


#====================================================
# PRISM_G 実行：Rの結果をGへ
#====================================================

if __name__ == "__main__":
    # 1. PRISM_R で計算された最新の偏差値データを読み込み
    # (PRISM_R_df には '馬名', 'PRISM_R_Score', 'EPI' が含まれている前提)
    PRISM_R_df = pd.read_sql(sql='SELECT * FROM "PRISM_R";', con=engine)
    
    # 2. 過去データとレース情報（枠順など）の読み込み
    MasterDataset_df = pd.read_sql(sql='SELECT * FROM "MasterDataset";', con=engine)
    RaceTable_df = pd.read_sql(sql='SELECT * FROM "RaceTable";', con=engine)
    
    # 3. PRISM_G 分析の実行
    # 馬場状態のシフト分析
    track_summary, intrinsic_baselines = RPCI_Shift_Analysis(MasterDataset_df)
    
    # 環境・展開補正の適用
    PRISM_RG_df = PRISM_G_Analysis(
        PRISM_R_df, 
        MasterDataset_df, 
        RaceTable_df, 
        track_summary, 
        intrinsic_baselines
    )
    
    # 4. 結果を PostgreSQL の "PRISM_RG" テーブルに保存
    PRISM_RG_df.to_sql('PRISM_RG', con=engine, if_exists='replace')
    
    # 5. PRISM_G可視化グラフの生成
    PRISM_G_Visualization(PRISM_RG_df)
    
    # 6. PRISM_RG可視化ダッシュボードの生成
    PRISM_RG_Visualization(PRISM_RG_df)