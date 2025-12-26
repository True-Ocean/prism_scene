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
    g.cond（今回の馬場状態）に合わせて基準値を動的に調整し、
    馬場状態の影響を考慮した統計値を返す。
    """
    # 1. 基礎データの抽出
    place_data = MasterDataset_df[
        (MasterDataset_df['場所'] == g.stadium) & 
        (MasterDataset_df['TD'] == g.td) & 
        (MasterDataset_df['RPCI'] > 0)
    ].copy()

    # 2. 距離ごとの「良馬場」中央値を計算（これは全ての基準点となる）
    ryo_all_dist = place_data[place_data['馬場状態'] == '良']
    distance_medians = ryo_all_dist.groupby('距離')['RPCI'].median().to_dict()
    
    # 3. RPCIシフト（各レースRPCI - 距離別良馬場基準）を算出
    place_data['RPCI_Diff'] = place_data.apply(
        lambda x: x['RPCI'] - distance_medians.get(x['距離'], np.nan), axis=1
    )

    # 4. 今回の馬場状態（g.cond）に応じた統計母集団の決定（マージロジック）
    MIN_SAMPLES = 10
    cond_order = ['良', '稍', '重', '不']
    current_idx = cond_order.index(g.cond) if g.cond in cond_order else 0
    
    # マージ範囲の定義：[今回のみ, 前後1つ, 全馬場]
    expansion_steps = [
        [current_idx, current_idx],
        [max(0, current_idx-1), min(3, current_idx+1)],
        [0, 3]
    ]

    final_subset = pd.DataFrame()
    used_conds = []

    for step in expansion_steps:
        target_conds = cond_order[step[0] : step[1]+1]
        subset = place_data[place_data['馬場状態'].isin(target_conds)].dropna(subset=['RPCI_Diff'])
        if len(subset) >= MIN_SAMPLES:
            final_subset = subset
            used_conds = target_conds
            break
    else:
        # 万が一、全馬場でもMIN_SAMPLESに達しない場合
        final_subset = place_data.dropna(subset=['RPCI_Diff'])
        used_conds = cond_order

    # 5. intrinsic_baselines（今回のレース基準値）の決定
    # 良馬場基準のRPCI中央値に、今回の馬場での「平均シフト量」を加算して補正する
    base_median = ryo_all_dist['RPCI'].median() if not ryo_all_dist.empty else 50.0
    shift_amount = final_subset['RPCI_Diff'].mean() if not final_subset.empty else 0.0
    
    # 標準偏差も今回の馬場グループのものを使用
    valid_stds = final_subset.groupby('距離')['RPCI'].std()
    
    intrinsic_baselines = {
        'median': base_median + shift_amount, # 馬場状態を考慮した補正済み基準RPCI
        'std': valid_stds.median() if not valid_stds.empty else 4.0,
        'shift': shift_amount # どの程度馬場に引っ張られているか
    }

    # 6. track_summary の更新
    # 全体の統計に加え、今回の分析で「採用された基準」という行を追加
    track_summary = place_data.dropna(subset=['RPCI_Diff']).groupby('馬場状態').agg({
        'RPCI_Diff': ['count', 'mean', 'median', 'std']
    }).reset_index()
    track_summary.columns = ['馬場状態', 'サンプル数', '平均シフト', '中央値シフト', '標準偏差']
    
    # 「今回採用基準」をサマリーの最後に追加
    current_summary = pd.DataFrame([{
        '馬場状態': f'★今回採用({g.cond})',
        'サンプル数': len(final_subset),
        '平均シフト': shift_amount,
        '中央値シフト': final_subset['RPCI_Diff'].median(),
        '標準偏差': final_subset['RPCI_Diff'].std()
    }])
    track_summary = pd.concat([track_summary, current_summary], ignore_index=True)

    return track_summary, intrinsic_baselines


#====================================================
# PRISM_G分析
#====================================================

def PRISM_G_Analysis(prism_r_df, MasterDataset_df, race_table_df, track_summary, intrinsic_baselines):
    
    # 1. 舞台設定の特定
    # 「今回の馬場状態(g.cond)」での当該コースデータを直接狙う
    course_target = MasterDataset_df[
        (MasterDataset_df['場所'] == g.stadium) & 
        (MasterDataset_df['距離'] == g.distance) & 
        (MasterDataset_df['TD'] == g.td) &
        (MasterDataset_df['馬場状態'] == g.cond) # ← ここを今回の馬場にする
    ]
    
    # 2. 基準RPCIの決定（フォールバック付き）
    if len(course_target) >= 5:
        # 今回の馬場のデータが十分（5件以上）あれば、その実測値を使う
        current_rpci_median = course_target['RPCI'].median()
        current_rpci_std = course_target['RPCI'].std()
    else:
        # データが足りない場合は、良馬場基準 + シフト量で推測する
        course_ryo = MasterDataset_df[
            (MasterDataset_df['場所'] == g.stadium) & 
            (MasterDataset_df['距離'] == g.distance) & 
            (MasterDataset_df['TD'] == g.td) &
            (MasterDataset_df['馬場状態'] == '良')
        ]
        
        # 良のデータすらない場合は、競馬場全体の平均(intrinsic_baselines)を使う
        base_m = course_ryo['RPCI'].median() if not course_ryo.empty else intrinsic_baselines['median']
        base_s = course_ryo['RPCI'].std() if len(course_ryo) > 1 else intrinsic_baselines['std']
        
        # 前段で計算したシフト量を加算
        s_val = intrinsic_baselines.get('shift', 0.0)
        current_rpci_median = base_m + s_val

        # --- 【ここがポイント：track_summaryの活用】 ---
        # 競馬場全体の馬場状態による標準偏差の変化率を適用する
        # 「★今回採用」行から、今回の馬場条件下での標準偏差を取得
        current_cond_stats = track_summary[track_summary['馬場状態'] == f'★今回採用({g.cond})']
        
        if not current_cond_stats.empty:
            cond_std = current_cond_stats['標準偏差'].values[0]
            # 良馬場の基準偏差に対して、今回の馬場グループの偏差がどう違うかを倍率で適用
            # 欠損値でなければ、偏差の広がりを反映させる
            if pd.notnull(cond_std) and intrinsic_baselines['std'] > 0:
                std_multiplier = cond_std / intrinsic_baselines['std']
                current_rpci_std = base_s * std_multiplier
            else:
                current_rpci_std = base_s
        else:
            current_rpci_std = base_s
    
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
        
        # --- 1. 個体適性(PCI)の算出（道悪考慮型） ---
        is_imputed = False  # ここで初期化しておくことでエラーを防ぐ
        
        if len(hist) <= 1 or (hist['PCI'] == 0).all():
            if len(hist) == 1:
                ideal_pci, sigma, is_imputed = hist['PCI'].iloc[0], 4.0, True
            else:
                # 0件の場合
                ideal_pci, sigma, is_imputed = current_rpci_median, 4.0, True
        else:
            # 2件以上ある場合
            target_runs = hist[hist['馬場状態'].isin(['稍', '重', '不'])] if g.cond in ['稍', '重', '不'] else hist
            if target_runs.empty: target_runs = hist
            
            top_runs = target_runs.nsmallest(min(5, len(target_runs)), '着順')
            ideal_pci = top_runs['PCI'].mean()
            
            raw_sigma = top_runs['PCI'].std()
            sigma = np.clip(raw_sigma if pd.notnull(raw_sigma) else 3.0, 1.5, 5.0)
            
        # --- 2. 適合度シミュレーション（生の確率計算） ---
        # 全範囲（3σ）での平均適合度
        weights_full = norm.pdf(rpci_range_full, current_rpci_median, current_rpci_std)
        responses_full = np.exp(- (rpci_range_full - ideal_pci)**2 / (2 * sigma**2))
        g_avg_raw = np.sum(responses_full * weights_full) / np.sum(weights_full)

        # 実効範囲（1σ）でのピークとリスク
        responses_real = np.exp(- (rpci_range_real - ideal_pci)**2 / (2 * sigma**2))
        g_peak_raw = max(g_avg_raw, np.max(responses_real))
        g_risk_raw = min(g_avg_raw, np.min(responses_real))

        # --- 3. レンジのスケーリング（0.88 〜 1.00 への圧縮） ---
        scale = 0.12
        offset = 0.88

        g_avg_mapped = offset + (g_avg_raw * scale)
        g_peak_mapped = offset + (g_peak_raw * scale)
        g_risk_mapped = offset + (g_risk_raw * scale)

        # --- 4. ガードレール（Rスコア連動の地力保証） ---
        # Rスコアが50の馬で 0.88 程度の最低保証
        g_floor = np.clip(0.86 + (r_score / 2000), 0.88, 0.92)
        g_ceiling = 1.0 if not is_imputed else 0.95

        g_avg = np.clip(g_avg_mapped, g_floor, g_ceiling)
        g_peak = np.clip(g_peak_mapped, g_floor, g_ceiling)
        g_risk = np.clip(g_risk_mapped, g_floor, g_ceiling)

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
    plt.title(f'PRISM_G 分析  {g.race_date} {g.stadium} {g.td} {g.distance}m {g.race_name} ({g.cond}))', fontsize=16, pad=20)
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

    # 馬場状態検証用：良、稍、重、不のいずれかを記述して確認してください。
    g.cond = '良'

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