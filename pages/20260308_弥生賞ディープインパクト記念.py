# ライブラリの準備
import streamlit as st

# --- メイン画面の表示 ---
st.markdown("""
    <style>
    /* 1. 画面幅に関わらず、すべての大きな見出しを一律で少し抑える設定 */
    h1 {
        font-size: 2.0rem !important;
    }
    
    /* 2. iPhone 13 Proを含むモバイル端末向けの設定（判定幅を広げました） */
    @media screen and (max-width: 850px) {
        /* メインタイトル (st.title) */
        h1 {
            font-size: 1.6rem !important;
            line-height: 1.2 !important;
        }
        /* 中見出し (st.header / ##) */
        h2 {
            font-size: 1.4rem !important;
        }
        /* 小見出し (st.subheader / ###) */
        h3 {
            font-size: 1.2rem !important;
        }
        /* 本文やウィジェットのラベル */
        p, .stText, label {
            font-size: 0.9rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🐎 2026年3月8日 弥生賞ディープインパクト記念') # ここを変更 =================================================================
st.divider()
st.write("準備中です。しばらくお待ちください。")
