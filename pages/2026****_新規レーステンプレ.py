# ライブラリの準備
import streamlit as st
from google.cloud import storage
import pandas as pd
import io
import os

# --- キャッシュ機能の定義 ---
@st.cache_data(ttl=3600)  # 1時間はキャッシュを保持

# 画像ファイルを読み込み、キャッシュ
@st.cache_data(ttl=3600)  # Streamlitのキャッシュ機能を追加
def load_all_images_from_gcs(bucket_name, sub_dir, file_names_list):
    """
    GCSから指定されたサブフォルダ内の複数の画像を読み込む
    bucket_name: "prism_scene_data_storage"
    sub_dir: "new" または "archive"
    file_names_list: 読み込みたいファイル名のリスト
    """
    client = storage.Client.from_service_account_info(st.secrets["gcp_service_account"])
    bucket = client.bucket(bucket_name)
    
    loaded_images = {}
    for name in file_names_list:
        # サブフォルダ名とファイル名を結合してフルパス（blob名）を作る
        full_path = f"{sub_dir}/{name}"
        blob = bucket.blob(full_path)
        
        if blob.exists():
            loaded_images[name] = blob.download_as_bytes()
        else:
            loaded_images[name] = None
            
    return loaded_images

# テキストファイル（.txt）を読み込みキャッシュ
def load_text_from_gcs(bucket_name, file_path):
    client = storage.Client.from_service_account_info(st.secrets["gcp_service_account"])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    if blob.exists():
        # テキストとしてダウンロード（UTF-8でデコードされます）
        return blob.download_as_text(encoding="utf-8")
    else:
        return "ファイルが見つかりませんでした。"

# 音声ファイル（.mp3）を読み込みキャッシュ
def load_audio_from_gcs(bucket_name, file_path):
    client = storage.Client.from_service_account_info(st.secrets["gcp_service_account"])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    if blob.exists():
        return blob.download_as_bytes()
    else:
        return None

# GCSからCSVを読み込んでDataFrameにする関数
def load_csv_from_gcs(bucket_name, file_path):
    client = storage.Client.from_service_account_info(st.secrets["gcp_service_account"])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    if blob.exists():
        csv_bytes = blob.download_as_bytes()
        # バイトデータをファイルのような形式（io.BytesIO）に変換してpandasで読み込む
        df = pd.read_csv(io.BytesIO(csv_bytes))
        return df
    else:
        return None


# --- データの読み込み ---
# 読み込みたいファイル名のリスト
target_files = [
    "PRISM_R.png",
    "PRISM_G.png",
    "PRISM_B_CW.png",
    "PRISM_B_Hanro.png",
    "PRISM_B_CW_Time.png",
    "PRISM_B_CW_Lap.png",
    "PRISM_B_Hanro_Time.png",
    "PRISM_B_Hanro_Lap.png",
    "PRISM_RGB.png",
]

# Google Cloud Storage のバケット名
dir_name = "prism_scene_data_storage"

# Google Cloud Storage のバケット名
sub_dir_name = "archive/2026****_レース名" # ================================= ここを書き換え =====================================


# キャッシュ関数を呼び出し（2回目以降はここが一瞬で終わります）
images = load_all_images_from_gcs(dir_name, sub_dir_name, target_files)

# --- 1. ページ全体の基本設定 ---
st.set_page_config(page_title="PRISM_SCENE Report", layout="wide")

# --- サイドメニューの設定 ---
with st.sidebar:
    st.header("1. カテゴリ")
    category = st.selectbox(
        "カテゴリ",
        ["プリズム分析", "シーン分析", "プリズム・シーン"]
    )

    st.header("2. サブカテゴリ")
    
    if category == "プリズム分析":
        sub_menu = st.selectbox(
            "プリズム分析の項目",
            ["基礎能力と先行指数", "レース条件適合率", "調教成長度", "最終期待値"]
        )
    elif category == "シーン分析":
        sub_menu = st.selectbox(
            "シーン分析の項目",
            ["キャラクター紹介", "ライバル関係", "シミュレーション結果", "注目キャラ"]
        )
    elif category == "プリズム・シーン":
        sub_menu = st.selectbox(
            "プリズム・シーンの項目",
            ["とある世界線の物語", "とある世界線のレース実況"]
        )

# --- メイン画面の表示 ---
st.title('🐎 2026年**月**日 レース名 分析レポート') # ここを変更 =================================================================
st.divider()
st.title(f"{category}：{sub_menu}")

# 共通の画像表示用関数（画像が見つからない場合の処理を追加）
def display_gcs_image(image_key, caption_text):
    if images.get(image_key):
        st.image(images[image_key], caption=caption_text, width="content")
    else:
        st.warning(f"画像 {image_key} がGCS上に見当たりません（パス: {sub_dir_name}/{image_key}）")

# --- メインコンテンツの分岐 ---
if sub_menu == "基礎能力と先行指数":
    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        display_gcs_image("PRISM_R.png", "PRISM_R")

    st.divider()
    st.subheader(f"💡 グラフ解説")
    st.write(f"このグラフは、PRISM_SCENE分析の中の、PRISM_R分析による結果を表しています。")
    st.write(f"【横軸】 EPI: Early Position Index（先行指数）は、過去のレース実績のAve-3F（前半600m換算タイム）をもとに評価された指標で、数値が高いほど序盤で逃げ・先行に向かう傾向があることを示します。")
    st.write(f"【縦軸】 基礎能力（偏差値）は、JRAで開催された過去20年間の同種レースの統計データから得られる標準スコアをベースとした、各馬の偏差値を示します。")
    st.write(f"【凡例】各馬の過去レースにおける決め手から確認される典型的な脚質を示します。※ EPIとの関連性は高いですが、常に一致するわけではありません。")


elif sub_menu == "レース条件適合率":
    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        display_gcs_image("PRISM_G.png", "PRISM_G")

    st.divider()
    st.subheader(f"💡 グラフ解説")
    st.write(f"このグラフは、PRISM_SCENE分析の中の、PRISM_G分析による結果を表しています。")
    st.write(f"【横軸】 PRISM_R分析で算出された各馬の基礎能力（偏差値）を示します。")
    st.write(f"【縦軸】 今回のレース条件（馬場状態、先行指数と枠順、予想される展開）に対する、各馬の適合率を示します。高いほど、今回のレースに対する適性が高いことを示しています。")
    st.write(f"【凡例】円が大きく明るい色になるほど、PRISM_G分析までの結果（PRISM_RGスコア）が高いことを示しています。")

elif sub_menu == "調教成長度":
    st.write("")
    st.write("")
    st.divider()

    st.subheader("🐎 CW調教による成長度")
    st.write("")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        display_gcs_image("PRISM_B_CW.png", "PRISM_B_CW")

    st.divider()

    st.subheader("🐎 坂路調教による成長度")
    st.write("")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        display_gcs_image("PRISM_B_Hanro.png", "PRISM_B_Hanro")

    st.divider()
    st.subheader(f"💡 グラフ解説")
    st.write(f"2つのグラフは、PRISM_SCENE分析の中の、PRISM_B分析による結果を表しています。")
    st.write(f"【横軸】 前走、前々走での調教における最終ラップタイム（L1）からの変化を示します。数値が高いほど、末脚の鋭さが上がっていることを示します。")
    st.write(f"【縦軸】 前走、前々走での調教における最終加速ラップ（L2-L1）からの変化を示します。数値が高いほど、加速の持続性が高くなっていることを示します。")

    st.divider()

    st.subheader("【参考】🐎 各調教データグラフ")
    st.write("")

    selected_training = st.selectbox(
        "調教選択",
        ["",
        "CW調教時計",
        "CW調教ラップ",
        "坂路調教時計",
        "坂路調教ラップ"]
    )

    if selected_training == "":
        st.write("見たい調教を選択すると、その調教に関するグラフがここに表示されます。")

    elif selected_training == "CW調教時計":
        st.write("")
        st.subheader("CW調教：6F時計推移（過去最大1年間）")
        st.write("")
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            display_gcs_image("PRISM_B_CW_Time.png", "PRISM_B_CW_Time")

        st.write("")
        st.divider()
        st.subheader(f"💡 グラフ解説")
        st.write(f"このグラフは、CW調教における各馬の6F時計の推移を示しています。")
        st.write(f"【横軸】 調教日付を示します。過去の調教データが左側、最新の調教データが右側に表示されます。")
        st.write(f"【縦軸】 各ラップタイムの時間（秒）を示します。小さくなるほど速いタイムを示します。調教の質が向上している場合、タイムが短縮される傾向があります。")


    elif selected_training == "CW調教ラップ":
        st.write("")
        st.subheader("CW調教：6Fラップ推移（最近2週間 最大ベスト3）")
        st.write("")
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            display_gcs_image("PRISM_B_CW_Lap.png", "PRISM_B_CW_Lap")

    elif selected_training == "坂路調教時計":
        st.write("")
        st.subheader("坂路調教：4F時計推移（過去最大1年間）")
        st.write("")
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            display_gcs_image("PRISM_B_Hanro_Time.png", "PRISM_B_Hanro_Time")

        st.write("")
        st.divider()
        st.subheader(f"💡 グラフ解説")
        st.write(f"このグラフは、今回のレース前14日間のCW調教における6F時計トップ3のラップ推移を示しています。")
        st.write(f"【横軸】 6Fの各ラップ（Lap6 ~ Lap1）を示します。")
        st.write(f"【縦軸】 各ラップタイムの時間（秒）を示します。小さくなるほど速いタイムを示します。調教の質が向上している場合、各ラップタイムが短縮される傾向があります。")
        st.write(f"右下がりのグラフは、終い重点型の調教を示し、特に、Lap2からLap1にかけての急激なタイム短縮は、鋭い末脚を発揮する可能性を示唆します。")


        st.write("")
        st.divider()
        st.subheader(f"💡 グラフ解説")
        st.write(f"このグラフは、坂路調教における各馬の4F時計の推移を示しています。")
        st.write(f"【横軸】 調教日付を示します。過去の調教データが左側、最新の調教データが右側に表示されます。")
        st.write(f"【縦軸】 各ラップタイムの時間（秒）を示します。小さくなるほど速いタイムを示します。調教の質が向上している場合、ラップタイムが短縮される傾向があります。")

    elif selected_training == "坂路調教ラップ":
        st.write("")
        st.write("坂路調教：4Fラップベスト3推移（最近2週間 最大ベスト3）")
        st.write("")
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            display_gcs_image("PRISM_B_Hanro_Lap.png", "PRISM_B_Hanro_Lap")

        st.write("")
        st.divider()
        st.subheader(f"💡 グラフ解説")
        st.write(f"このグラフは、今回のレース前14日間の坂路調教における4F時計トップ3のラップ推移を示しています。")
        st.write(f"【横軸】 4Fの各ラップ（Lap4 ~ Lap1）を示します。")
        st.write(f"【縦軸】 各ラップタイムの時間（秒）を示します。小さくなるほど速いタイムを示します。調教の質が向上している場合、各ラップタイムが短縮される傾向があります。")
        st.write(f"右下がりのグラフは、終い重点型の調教を示し、特に、Lap2からLap1にかけての急激なタイム短縮は、鋭い末脚を発揮する可能性を示唆します。")


elif sub_menu == "最終期待値":
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        display_gcs_image("PRISM_RGB.png", "PRISM_RGB")

    st.divider()
    st.subheader(f"💡 グラフ解説")
    st.write(f"このグラフは、PRISM分析の最終結果を表しています。")
    st.write(f"【左の赤いグラフ】各馬の基礎能力（PRISM_Rスコア）を示します。")
    st.write(f"【中央緑のグラフ】PRISM_Rスコアにレース条件適合率を乗算した偏差値（PRISM_RGスコア）を示しています。")
    st.write(f"【右の青いグラフ】PRISM_RGスコアに調教成長度を加算した偏差値（PRISM_RGBスコア）を示しています。")


elif sub_menu == "キャラクター紹介":
    st.write("")
    st.write("")
    st.write("ここでは、SCENE分析の中の、SCENE_Script分析 および SCENE_Cast分析 による結果を表示します。")
    st.divider()
    SCENE_Cast_df = load_csv_from_gcs(dir_name, f"{sub_dir_name}/SCENE_Cast.csv")

    selected_columns = ['番', '馬名', '血統分析', 'キャラ設定', '自己紹介']
    
    if SCENE_Cast_df is not None:
        
        horse_list = SCENE_Cast_df['馬名'].tolist()
        selected_horse = st.selectbox(
            "登場キャラ選択",
            options=horse_list,
            index=None,
            placeholder="見たいキャラを選択してください。"
        )

        if selected_horse:
            horse_data = SCENE_Cast_df[SCENE_Cast_df['馬名'] == selected_horse].iloc[0]

            st.subheader(f"🏡 {selected_horse} の血筋・家柄")
            st.write(horse_data["血統分析"])
            st.subheader(f"🐎 {selected_horse} の特徴")
            st.write(horse_data["キャラ設定"])
            st.subheader(f"🐴 {selected_horse} の自己紹介")
            st.write(horse_data["自己紹介"])
        else:
            st.write("キャラを選択すれば、そのキャラの「血筋・家柄」「特徴」「自己紹介」を見ることができます。")

        st.divider()

        st.subheader(f"📝 登場キャラ一覧")
        st.write("見切れているセルは、ダブルクリックすれば全文を確認できます。")
        st.dataframe(
            SCENE_Cast_df[selected_columns],
            hide_index=True,
            width="content",
            height=680,
            column_config={
                "番": st.column_config.TextColumn("番", width=30), # 数値ではなくテキストとして扱うことで左寄せに
                "馬名": st.column_config.TextColumn("馬名", width=180), # 幅を広げて1行に収まりやすくする
                "血統分析": st.column_config.TextColumn("血統分析", width=300),
                "キャラ設定": st.column_config.TextColumn("キャラ設定", width=300),
                "自己紹介": st.column_config.TextColumn("自己紹介", width=300),
            }
        )

    else:
        st.error("CSVファイルが見つかりませんでした。")

    st.divider()
    st.subheader(f"💡 解説")
    st.write(f"【SCENE_Script分析】各馬の多様なデータとPRISM分析の結果をもとに、各馬の「特徴」をテキスト情報として整理")
    st.write(f"【SCENE_Cast分析】SCENE_Script分析と各馬の5代血統表をもとに、各馬の「血統分析」「キャラ設定」「自己紹介」を生成（by Gemini API）")

elif sub_menu == "ライバル関係":
    st.write("")
    st.write("")
    st.write("ここでは、SCENE分析の中の、SCENE_Ensemble分析 による結果を表示します。")
    st.divider()

    scene_ensemble_df = load_csv_from_gcs(dir_name, f"{sub_dir_name}/SCENE_Ensemble.csv")

    SCENE_Ensemble_df = scene_ensemble_df.rename(columns = {'馬名_A':'キャラA', '馬名_B':'キャラB', 'Total_Matches':'対戦', 'conclusion_type':'タイプ', 'narrative_summary':'対戦内容',\
                                                             'turning_point_race':'注目レース', 'current_dominance':'優位性', 'dominance_reason':'優位性の根拠', 'ライバル関係':'ライバル関係'})

    if SCENE_Ensemble_df is not None:

        horse_list_A = SCENE_Ensemble_df['キャラA'].unique().tolist()
        selected_horse_A = st.selectbox(
            "キャラA選択",
            options=horse_list_A,
            index=None,
            placeholder="キャラAを選択してください。"
        )

        if selected_horse_A:
            horse_A_df = SCENE_Ensemble_df[SCENE_Ensemble_df['キャラA'] == selected_horse_A]

            horse_list_B = horse_A_df['キャラB'].unique().tolist()
            selected_horse_B = st.selectbox(
                "キャラB選択",
                options=horse_list_B,
                index=None,
                placeholder="キャラBを選択してください。"
        )

            if selected_horse_B:
                horse_AB_data = horse_A_df[horse_A_df['キャラB'] == selected_horse_B].iloc[0]
                st.subheader(f"⚔️ {selected_horse_A} vs {selected_horse_B} のライバル関係")
                st.write(horse_AB_data["ライバル関係"])

            else:
                st.write("2人のキャラを選択すれば、そのキャラ同士の「ライバル関係」を見ることができます。")

        else:
            st.write("キャラAを選択すれば、キャラBの選択肢が表示され、その2人の「ライバル関係」を見ることができます。")

    st.write
    st.divider()

    selected_columns = ['キャラA', 'キャラB', '対戦', 'タイプ', '対戦内容', '注目レース', '優位性', '優位性の根拠', 'ライバル関係']

    if SCENE_Ensemble_df is not None:

        st.subheader(f"📝 本レースにおけるライバル関係一覧")
        st.write(f"💡ライバル関係として注目度の高い上位最大10件をピックアップ‼️")
        st.write("見切れているセルは、ダブルクリックすれば全文を確認できます。")
        st.dataframe(
            SCENE_Ensemble_df[selected_columns],
            hide_index=True,
            width="content",
            height=400,
            column_config={
                "キャラA": st.column_config.TextColumn("キャラA", width=180), 
                "キャラB": st.column_config.TextColumn("キャラB", width=180), 
                "対戦": st.column_config.TextColumn("対戦", width=50), 
                "タイプ": st.column_config.TextColumn("タイプ", width=100),
                "対戦内容": st.column_config.TextColumn("対戦内容", width=300),
                "注目レース": st.column_config.TextColumn("注目レース", width=200),
                "優位性": st.column_config.TextColumn("優位性", width=200),
                "優位性の根拠": st.column_config.TextColumn("優位性の根拠", width=300),
                "ライバル関係": st.column_config.TextColumn("ライバル関係", width=400),
            }
        )
        st.write("横にスクロールできます。>>>>>")

    else:
        st.error("CSVファイルが見つかりませんでした。")

    st.divider()
    st.subheader(f"💡 解説")
    st.write(f"【SCENE_Ensemble分析】各馬の過去のレース実績から、各馬の「ライバル関係」を抽出し、注目度の高い順に並べて、テキスト情報として整理")


elif sub_menu == "シミュレーション結果":
    st.write("")
    st.write("")
    st.write("ここでは、SCENE分析によるバックグラウンド・シミュレーション結果を表示します。")
    st.divider()
    Final_Report_df = load_csv_from_gcs(dir_name, f"{sub_dir_name}/Final_Report.csv")

    if Final_Report_df is not None:

        st.subheader(f"🌏 全世界線のシミュレーション結果")
        st.write("")
        st.write(f"💡各キャラが主人公となる（出来るだけ上位に入着する）世界線を、{len(Final_Report_df)}キャラ分シミュレーションした結果")
        st.write("")
        st.dataframe(
            Final_Report_df,
            hide_index=True,
            width="content",
            height=680,
            column_config={
                "枠": st.column_config.TextColumn("枠番", width=30), 
                "番": st.column_config.TextColumn("番", width=30),
                "馬名": st.column_config.TextColumn("馬名", width=180),
                "平均着順": st.column_config.TextColumn("平均着順", width=100),
                "勝率": st.column_config.TextColumn("勝率", width=100),
                "連対率": st.column_config.TextColumn("連対率", width=100),
                "複勝率": st.column_config.TextColumn("複勝率", width=100),
                "最高位": st.column_config.TextColumn("最高位", width=100),
                "最下位": st.column_config.TextColumn("最下位", width=100),
            }
        )

    else:
        st.error("CSVファイルが見つかりませんでした。")

    st.divider()
    st.subheader(f"💡 解説")
    st.write(f"【SCENE分析】SCENE_Script分析とSCENE_Cast分析によって生成された「キャラ設定」と、SCENE_Ensemble分析によって抽出された「ライバル関係」を踏まえ、\
             各キャラが主役となるバックグラウンド・シミュレーションをキャラの数だけ実施し、各キャラの平均着順、勝率、連対率、複勝率、最高位、最下位を算出しています。")


elif sub_menu == "注目キャラ":
    st.write("")
    st.write("")
    st.write("ここでは、SCENE分析によって確認された注目キャラを発表します。")
    st.divider()
    Final_Mark_df = load_csv_from_gcs(dir_name, f"{sub_dir_name}/Final_Mark.csv")

    if Final_Mark_df is not None:

        st.subheader(f"📢 注目キャラの発表")
        st.write("")
        st.write(f"💡 全登場キャラの世界線を踏まえた、今回のレースの注目キャラは以下の{len(Final_Mark_df)}キャラとなりました。")
        st.write("")
        st.dataframe(
            Final_Mark_df,
            hide_index=True,
            width="content",
            height=250,
            column_config={

                "印": st.column_config.TextColumn("印", width=100), # 数値ではなくテキストとして扱うことで左寄せに
                "枠": st.column_config.TextColumn("枠番", width=50), # 数値ではなくテキストとして扱うことで左寄せに
                "番": st.column_config.TextColumn("番", width=50), # 数値ではなくテキストとして扱うことで左寄せに
                "馬名": st.column_config.TextColumn("馬名", width=180), # 幅を広げて1行に収まりやすくする

            }
        )
        st.divider()
        st.subheader(f"💡 印の意味")
        st.write('')
        st.write('全世界線におけるシミュレーションの集計結果をベースに、以下の基準で注目キャラを選出しています。')
        st.markdown("""
        【 本  名 】: 平均着順が最も優秀なキャラ  
        【 対  抗 】: 平均着順が2番手のキャラ  
        【 単  穴 】: 残ったキャラの中で、勝率が最も高いキャラ  
        【 ドラマ 】: 残ったキャラの中で、最高位が "3着以内" かつ "「ライバル関係」を有する" キャラ  
        【 ロマン 】: 残ったキャラの中で、最高位が "1着" または 複勝率がトップ のキャラ  
        【ドリーム】: 残ったキャラの中で、最高位が "3着以内" かつ "最も勝率が低い" キャラ
        """)


    else:
        st.error("CSVファイルが見つかりませんでした。")


elif sub_menu == "とある世界線の物語":
    st.write("")
    st.write("")
    st.write("『とある世界線の物語』をお楽しみください。")
    st.divider()
    final_drama = load_text_from_gcs(dir_name, f"{sub_dir_name}/Final_Drama.txt")
    st.text(final_drama)

    st.divider()
    st.subheader(f"💡 解説")
    st.write(f"全世界線シミュレーションの最終結果をベースに、各キャラの「キャラ設定」や「ライバル関係」を踏まえて、今回のレースにおけるドラマチックな物語を生成しています。（by Gemini API）")


elif sub_menu == "とある世界線のレース実況":

    st.write("")
    st.write("『とある世界線のレース実況』をお楽しみください。")
    st.write("")
    # 音声データの取得
    audio_bytes = load_audio_from_gcs(dir_name, f"{sub_dir_name}/Broadcast.mp3")

    if audio_bytes:
        # Streamlit標準のオーディオプレーヤーを表示
        st.audio(audio_bytes, format="audio/mpeg")
    else:
        st.error("音声ファイルが見つかりませんでした。")

    st.divider()
    st.subheader(f"📻 レース実況テキスト")
    st.write('')
    broadcast = load_text_from_gcs(dir_name, f"{sub_dir_name}/Broadcast.txt")

    replace_dict = {
        "コオナー": "コーナー",
        "メートル": "m",
        "はじける": "弾ける",
        "じゅうしょう": "重賞",
        "せんこうば": "先行馬",
        "さいこうほう": "最後方",
        "むこうじょうめん": "向正面",
        "すうばしん": "数馬身",
        "まえめ": "前目",
        "そとそと": "外々",
        "おおそと": "大外",
        "さいうち": "最内",
        "すえあし": "末脚",
        "こうスタート": "好スタート", 
        "あしいろ": "脚色", 
        "せんこうぜい": "先行勢",
        "せんこう": "先行",
        "追い込み": "追込",
        "うち": "内",
        "そと": "外",
        "急ざか": "急坂",
        "18ばん": "18番",
        "ばぐん": "馬群",
        "おも": "重",
        "ややおも": "稍重",
        "ゴールばん": "ゴール板",
        "ステークス": "S",
        "カップ": "C",
        "トロフィー": "T" 
    }

    # 一括置換の実行
    for old, new in replace_dict.items():
        broadcast = broadcast.replace(old, new)

    st.write(broadcast)

    st.divider()
    st.subheader(f"💡 解説")
    st.write(f"「とある世界線の物語」のレース展開をベースに、純粋なレース実況音声を生成しています。（by Gemini API）")
