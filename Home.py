import streamlit as st

st.set_page_config(
    page_title="CardioVis",
    page_icon="🫀"
)

pg = st.navigation([
    st.Page("Pages/About.py", title="About", icon="🏠"),
    st.Page("Pages\Preprocessing.py", title="Preprocessing", icon="1️⃣"),
    st.Page("Pages\Segmentation.py", title="Segmentation", icon="2️⃣"),
    st.Page("Pages\CTR Calculation.py", title="CTR Calculation", icon="3️⃣")
])
pg.run()
