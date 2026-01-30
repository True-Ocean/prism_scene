import streamlit as st
import streamlit.components.v1 as components

# 基本設定
st.set_page_config(page_title="PRISM_SCENE")

# SNS共有用のメタタグ埋め込み
components.html(
    """
    <head>
        <title>PRISM_SCENE</title>
        <meta property="og:type" content="website">
        <meta property="og:url" content="https://prism-scene.streamlit.app">
        <meta property="og:title" content="PRISM_SCENE">
        <meta property="og:description" content="プリズム・シーンへようこそ！定量分析レポートから『とある世界線の物語』、レース後の『アフター・ストーリー』まで、競馬コンテンツが満載のプラットフォームです。">
    </head>
    """,
    height=0, # 画面には表示させない
)

# --- 各ページの中身（関数） ---
def show_home():
    st.title('🏠 ようこそ！')
    with open("pages/Home.py", encoding="utf-8") as f:
            code = compile(f.read(), "pages/Home.py", 'exec')
            exec(code, globals())

def show_new_auth():
    # 1. 認証状態の初期化
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # 2. 認証済みの場合：分析ページの中身だけを表示
    if st.session_state.authenticated:
        st.title('🐎 2025年12月28日 有馬記念')# タイトルを変更 =============================================
        if st.button("ログアウト"):
            st.session_state.authenticated = False
            st.rerun()

        st.divider()
        # 🔑 スペシャルコンテンツの内容を表示
        with open("pages/20251228_有馬記念.py", encoding="utf-8") as f: # .pyファイル名を指定 =============================================
            code = compile(f.read(), "pages/20251228_有馬記念.py", 'exec')
            exec(code, globals())

    # 3. 未認証の場合：パスコード入力欄を表示
    else:
        st.title('🔐 パスコード認証')
        password = st.text_input("パスコードを入力してください", type="password")
        
        if st.button("認証する"):
            # 🔐 Secrets からパスコードを読み込んで比較
            if password == st.secrets["APP_PASSCORD"]: # Streamlitのウェブ画面右下「Manage app」のメニューから、Setting > Secrets の一番上の記載を変更 ================================
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("パスコードが違います。")
        
        st.write('note の記事内にパスコードを無料公開しています。noteでシャノワールをフォローしていただければ嬉しいです。')
        st.write('note のリンク先はこちら 👉 https://note.com/prism_scene')

def show_race_content(file_name):
    try:
        path = f"pages/{file_name}"
        with open(path, encoding="utf-8") as f:
            code = compile(f.read(), path, 'exec')
            exec(code, globals())
    except FileNotFoundError:
        st.error(f"ファイル {file_name} が見つかりませんでした。")

def load_archive_content(file_name):
    try:
        path = f"pages/{file_name}"
        with open(path, encoding="utf-8") as f:
            code = compile(f.read(), path, 'exec')
            exec(code, globals())
    except FileNotFoundError:
        st.error(f"ファイル {file_name} が見つかりませんでした。")

# --- 今週のレースを選択・表示する関数 ---
def show_races():
    st.title("📚 今週のレース")
    
    # ドロップダウンの選択肢を作成（表示名：ファイル名）
    race_options = {
        "選択してください": None,
        "2026/2/1 根岸S（G3）": "20260201_根岸S.py", #=============================== ファイル名を変更 =====================================
        "2026/2/1 シルクロードS（G3）": "20260201_シルクロードS.py",  #=============================== ファイル名を変更 =====================================
        }

    selected_label = st.selectbox("今週のレースレポートを選択", options=list(race_options.keys()))

    file_name = race_options[selected_label]
    if file_name:
        st.divider()
        load_archive_content(file_name)
    else:
        st.info("見たいレースを上のメニューから選んでください。")

# --- アーカイブを選択・表示する関数 ---
def show_archives():
    st.title("📚 過去のG1レース")
    
    # ドロップダウンの選択肢を作成（表示名：ファイル名）
    archive_options = {
        "選択してください": None,
        "2025/12/27 ホープフルS": "20251227_ホープフルS.py",
        "2025/11/30 ジャパンカップ": "20251130_ジャパンカップ.py"
    }
    
    selected_label = st.selectbox("過去のレースレポートを選択", options=list(archive_options.keys()))
    
    file_name = archive_options[selected_label]
    if file_name:
        st.divider()
        load_archive_content(file_name)
    else:
        st.info("見たいレースを上のメニューから選んでください。")

# --- 1. ページの定義 ---
home_page = st.Page(show_home, title="ホーム", icon="🏠")
new_page = st.Page(show_new_auth, title="2025/12/28 有馬記念", icon="🔥") # タイトルを変更 =================================
race_page = st.Page(show_races, title="今週の注目レース", icon="🏇")
archive_page = st.Page(show_archives, title="過去のG1レース", icon="📂") # 1つに統合

# --- 2. ナビゲーションの定義 ---
pg = st.navigation({
    "PRISM_SCENE": [home_page],
    "スペシャルコンテンツ": [new_page],
    "フリーコンテンツ": [race_page],
    "アーカイブ": [archive_page] # サイドバーには1項目だけ表示される
})

with st.sidebar:
    st.caption("PRISM_SCENE v3.1.2")

pg.run()