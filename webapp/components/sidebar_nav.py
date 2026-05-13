"""Sidebar dùng chung cho toàn bộ ứng dụng Streamlit."""
import streamlit as st


def hide_streamlit_default_nav():
    """Ẩn menu multipage mặc định để tránh trùng với điều hướng tiếng Việt."""
    st.markdown(
        """
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""",
        unsafe_allow_html=True,
    )


def show_sidebar_nav():
    hide_streamlit_default_nav()
    with st.sidebar:
        st.markdown("## SmartIrrigationAI")
        st.markdown("Tối ưu phân bổ nước tưới tiêu bằng AI")
        st.divider()
        st.markdown("**Điều hướng**")
        st.page_link("app.py", label="Trang chủ")
        st.page_link("pages/01_input.py", label="Nhập dữ liệu")
        st.page_link("pages/02_run.py", label="Demo thuật toán")
        st.page_link("pages/03_result.py", label="Kết quả phân tích")
        st.divider()
        st.caption("Môn: Trí Tuệ Nhân Tạo\nGA • SA • PSO • Hybrid")
