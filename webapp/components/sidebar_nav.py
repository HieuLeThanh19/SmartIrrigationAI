"""Sidebar dùng chung cho toàn bộ ứng dụng Streamlit."""
import streamlit as st


def hide_streamlit_default_nav():
    """Ẩn menu multipage mặc định để tránh trùng với điều hướng tiếng Việt."""
    st.markdown(
        """
<style>
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] h2 { font-size: 1.1rem; margin-bottom: .15rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: .35rem; font-size: .88rem; }
[data-testid="stSidebar"] hr { margin: .6rem 0; }
[data-testid="stSidebar"] [data-testid="stPageLink"] a { padding: .28rem .4rem; min-height: 2rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { font-size: .78rem; }
</style>
""",
        unsafe_allow_html=True,
    )


def show_sidebar_nav():
    hide_streamlit_default_nav()
    with st.sidebar:
        st.markdown("## SmartIrrigationAI")
        st.caption("Phân bổ nước tưới tiêu")
        st.divider()
        st.page_link("app.py", label="Trang chủ")
        st.page_link("pages/01_input.py", label="Nhập dữ liệu")
        st.page_link("pages/02_run.py", label="Demo từng thuật toán")
        st.page_link("pages/03_all_algorithms.py", label="Demo tất cả thuật toán")
        st.page_link("pages/03_result.py", label="Kết quả")
        st.divider()
        st.caption("AI • GA • SA • PSO • Hybrid")
