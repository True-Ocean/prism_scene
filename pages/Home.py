

import streamlit as st

st.set_page_config(page_title="PRISM_SCENE Home", layout="wide")

st.title("🐎 PRISM_SCENE 分析結果 公開プラットフォーム")

st.divider()
st.markdown("""
### このサイトについて
PRISM_SCENEは、競走馬のパフォーマンスデータを多角的に分析し、
定量的なスコアとGeminiによる定性的なナラティブ（物語）を統合するシステムです。

            
### **主な機能:**
- **プリズム（PRISM）分析**:（定量分析）出走各馬の多様なデータと過去20年間の競馬データベースをもとに定量的に分析した結果を、わかりやすくグラフで可視化しています。
- **シーン（SCENE）分析**:（定性分析）出走各馬の多様なデータから割り出した特徴、キャラ、ライバル関係をもとに、全馬の世界線でレースをシミュレーションしています。
- **プリズム・シーン（PRISM_SCENE）分析**:（総合評価）最終的に割り出されたレース展開を、『とある世界線の物語』として描き出し、さらにレース実況音声を生成しています。
""")

st.info("👈　左側のサイドバーから見たいレースを選択して、分析結果を確認しよう ‼️")

st.divider()
st.write("")
st.markdown("""
### PRISM_SCENE の由来:
#### <b>Performance Rating and Intelligent Scoring Model  &  Systematic Character Extraction for Narrative Epilogue  の略称</b>
- <b>「プリズム」</b>は、光を分散させて多様な色を生み出す装置であり、競走馬の多面的なデータ分析を象徴しています。プリズム分析を構成するPRISM_R,PRISM_G,PRISM_Bの3つ分析は、色の三元色に準えた名称となっています。
- <b>「シーン」</b>は、物語の一場面を指し、競走馬の特徴や関係性を通じてレースのドラマを描き出すことを意味しています。シーン分析を構成するSCENE_Script,SCENE_Cast,SCENE_Ensembleの3つ分析は、映画や演劇の制作過程に準えた名称となっています。
""", unsafe_allow_html=True)
st.write("")
st.divider()
st.write("")
st.markdown("""
### 📝 更新履歴
- 2026/1/4: 京都金杯 の分析結果を公開
- 2026/1/4: 中山金杯 の分析結果を公開
- 2025/12/30: ジャパンC の分析結果を公開
- 2025/12/30: ホープフルS の分析結果を公開
- 2025/12/30: 有馬記念 の分析結果を公開
- 2025/12/30: ホームページをリリース
""")
st.write("")
st.divider()

st.markdown("""
Disclaimer: This platform is for educational and informational purposes only. It does not provide financial or betting advice. Users are responsible for their own decisions.<br>
免責事項：本プラットフォームは教育および情報提供を目的としており、金融または賭博に関するアドバイスを提供するものではありません。利用者は自己の判断に基づいて行動する責任があります。
""", unsafe_allow_html=True)

st.write("©︎ 2025 PRISM_SCENE Project. All rights reserved.")