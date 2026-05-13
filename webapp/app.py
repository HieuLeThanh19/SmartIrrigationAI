"""SmartIrrigationAI — Entry point Streamlit App"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import webapp.session as session_state
from webapp.components.sidebar_nav import show_sidebar_nav

st.set_page_config(
    page_title="SmartIrrigationAI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

session_state.init_session()
if hasattr(session_state, "sync_budget_state"):
    session_state.sync_budget_state()

show_sidebar_nav()

st.title("SmartIrrigationAI")
st.subheader("Hệ thống phân bổ nước tưới tiêu thông minh sử dụng AI")
st.markdown("""
Hệ thống sử dụng **4 thuật toán tối ưu hóa** để phân bổ nguồn nước hạn chế
cho **10 thửa ruộng** với các loại cây trồng và nhu cầu nước khác nhau,
nhằm **tối thiểu hóa chi phí bơm** và **đảm bảo mọi thửa ruộng được tưới đủ**.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số thuật toán", "4")
c2.metric("Số thửa ruộng", "10")
c3.metric("Ngân sách nước hiện tại", f"{st.session_state['W_total']:.0f} m³/ngày")
c4.metric("Thuật toán", "GA · SA · PSO · Hybrid")

st.divider()
if st.button("Bắt đầu ngay →", type="primary", use_container_width=True):
    st.switch_page("pages/01_input.py")
