import streamlit as st

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
        st.title('🐎 2025年12月28日 有馬記念 分析レポート')
        if st.button("ログアウト"):
            st.session_state.authenticated = False
            st.rerun()

        st.divider()
        # 🔑 会員限定コンテンツの内容を表示
        with open("pages/20251228_有馬記念.py", encoding="utf-8") as f: # ファイル名を指定 =============================================
            code = compile(f.read(), "pages/20251228_有馬記念.py", 'exec')
            exec(code, globals())

    # 3. 未認証の場合：パスワード入力欄を表示
    else:
        st.title('🔐 会員認証')
        password = st.text_input("パスワードを入力してください（会員限定）", type="password")
        
        if st.button("認証する"):
            if password == "1234": # パスワードを指定 =============================================
                st.session_state.authenticated = True
                st.rerun() # 🚀 ここで画面を書き換える
            else:
                st.error("パスワードが違います。")
    pass

def load_archive_content(file_name):
    try:
        path = f"pages/{file_name}"
        with open(path, encoding="utf-8") as f:
            code = compile(f.read(), path, 'exec')
            exec(code, globals())
    except FileNotFoundError:
        st.error(f"ファイル {file_name} が見つかりませんでした。")

# --- アーカイブを選択・表示する関数 ---
def show_archives():
    st.title("📚 フリーアーカイブ")
    
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
new_page = st.Page(show_new_auth, title="2025/12/28 有馬記念", icon="🔥")
archive_page = st.Page(show_archives, title="フリーアーカイブ", icon="📂") # 1つに統合

# --- 2. ナビゲーションの定義 ---
pg = st.navigation({
    "PRISM_SCENE": [home_page],
    "会員限定コンテンツ": [new_page],
    "アーカイブ": [archive_page] # サイドバーには1項目だけ表示される
})

with st.sidebar:
    st.caption("PRISM_SCENE v2.0")

pg.run()