import streamlit as st

# 基本設定
st.set_page_config(page_title="PRISM_SCENE")

# アプリ全体の見出しサイズやモバイル対応を一括管理します
st.markdown("""
    <style>
    /* 1. 全画面共通の見出しサイズ抑制 */
    h1 { font-size: 2.0rem !important; }

    /* 2. モバイル端末（iPhone等）向けの最適化 */
    @media screen and (max-width: 850px) {
        /* メインタイトル (st.title) */
        h1 { font-size: 1.6rem !important; line-height: 1.2 !important; }
        /* 中見出し (st.header / ##) */
        h2 { font-size: 1.4rem !important; }
        /* 小見出し (st.subheader / ###) */
        h3 { font-size: 1.2rem !important; }
        /* 本文やウィジェットのラベル */
        p, .stText, label { font-size: 0.9rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

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

        st.title('🐎 2026年5月10日 NHKマイルカップ')# タイトルを変更 =============================================

        if st.button("ログアウト"):
            st.session_state.authenticated = False
            st.rerun()

        st.divider()
        # 🔑 スペシャルコンテンツの内容を表示
        with open("pages/20260510_NHKマイルカップ.py", encoding="utf-8") as f: # .pyファイル名を指定 =============================================
            code = compile(f.read(), "pages/20260510_NHKマイルカップ.py", 'exec') # .pyファイル名を指定 =============================================
            exec(code, globals())

    # 3. 未認証の場合：パスコード入力欄を表示
    else:

        st.title('🔐 パスコード認証')
        password = st.text_input("パスコードを入力してください", type="password")
        
        if st.button("認証する"):
            # 🔐 Secrets からパスコードを読み込んで比較
            # if password == "Tak_Miya": # ローカル検証用 Github同期の際には、下の行に切り替えること！ =================================================
            if password == st.secrets["APP_PASSCODE"]: # Streamlitのウェブ画面右下「Manage app」のメニューから、Setting > Secrets の一番上の記載を変更 ================================
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("パスコードが違います。")
        
        st.write('note の記事内にパスコードを無料公開しています。')
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

    st.title("📚 今週の注目レース")
    
    # ドロップダウンの選択肢を作成（表示名：ファイル名）
    race_options = {
        "選択してください": None,
        # "2026/3/22 阪神大賞典": "20260322_阪神大賞典.py", #=============================== ファイル名を変更 =====================================
        # "2026/3/7 中山牝馬S（G3）": "20260307_中山牝馬S.py", #=============================== ファイル名を変更 =====================================
        # "2026/3/8 弥生賞ディープインパクト記念（G2）": "20260308_弥生賞.py", #=============================== ファイル名を変更 =====================================
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
        "2026/5/3 天皇賞（春）": "20260503_天皇賞春.py",
        "2026/4/19 皐月賞": "20260419_皐月賞.py",
        "2026/4/12 桜花賞": "20260412_桜花賞.py",
        "2026/4/5 大阪杯": "20260405_大阪杯.py",
        "2026/3/29 高松宮記念": "20260329_高松宮記念.py",
        "2026/2/22 フェブラリーS": "20260222_フェブラリーS.py",
        "2025/12/27 ホープフルS": "20251227_ホープフルS.py",
        "2025/12/28 有馬記念": "20251228_有馬記念.py",
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
new_page = st.Page(show_new_auth, title="2026/5/10 NHKマイルカップ", icon="🔥") # タイトルを変更 =================================
# race_page = st.Page(show_races, title="今週の注目レース", icon="🏇") # G1以外の注目レースがある場合は、この行をアクティブにすること =================================
archive_page = st.Page(show_archives, title="過去のG1レース", icon="📂") # 1つに統合

# --- 2. ナビゲーションの定義 ---
pg = st.navigation({
    "PRISM_SCENE": [home_page],
    "スペシャルコンテンツ": [new_page],
    # "フリーコンテンツ": [race_page],
    "アーカイブ": [archive_page]
})

with st.sidebar:
    st.caption("PRISM_SCENE v3.1.3")

pg.run()