import streamlit as st

# --- 各ページの中身（関数） ---
def show_home():
    st.title('🏠 ようこそ！')
    with open("pages/Home.py", encoding="utf-8") as f:
            code = compile(f.read(), "pages/Home.py", 'exec')
            exec(code, globals())

# パスワード入力
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

def load_archive_content(file_name):
    # ファイル読み込みに失敗しないよう、少し丁寧に記述
    try:
        path = f"pages/{file_name}"
        with open(path, encoding="utf-8") as f:
            code = compile(f.read(), path, 'exec')
            exec(code, globals())
    except FileNotFoundError:
        st.error(f"ファイル {file_name} が見つかりませんでした。")


# --- 1. ページの定義 ---
home_page = st.Page(show_home, title="ホーム", icon="🏠")
new_page = st.Page(show_new_auth, title="2025/12/28 有馬記念", icon="🔥")

# アーカイブページを個別に定義
# url_path を追加して、それぞれ別の名前を付けます
archive_hopeful = st.Page(
    lambda: load_archive_content("20251227_ホープフルS.py"), 
    title="2025/12/27 ホープフルS", 
    icon="🐎",
    url_path="hopeful_2025" # ← ここを追加（英数字とハイフンのみ推奨）
)

archive_japan_cup = st.Page(
    lambda: load_archive_content("20251130_ジャパンカップ.py"), 
    title="2025/11/30 ジャパンカップ", 
    icon="🏆",
    url_path="japan_cup_2025" # ← ここを追加
)

# --- 2. ナビゲーションをセクション分けして定義 ---
# この辞書の「キー（太字部分）」がセクションのタイトルになります
pg = st.navigation({
    "PRISM_SCENE": [home_page],
    "会員限定コンテンツ": [new_page],
    "フリーアーカイブ": [
        archive_hopeful, 
        archive_japan_cup
    ]
})

# --- 3. サイドバーの装飾 ---
with st.sidebar:
    st.caption("PRISM_SCENE v2.0")

pg.run()
