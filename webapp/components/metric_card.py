# webapp/components/metric_card.py — Widget hiển thị kết quả từng thuật toán
import streamlit as st


def show_algo_card(algo_name: str, best_fitness: float, mean_fitness: float,
                   std: float, runtime: float, is_best: bool = False):
    """Hiển thị card kết quả thuật toán."""
    badge = "Tốt nhất" if is_best else ""
    border = "#2E7D32" if is_best else "#37474F"
    bg     = "linear-gradient(135deg,#E8F5E9,#F1F8E9)" if is_best else "linear-gradient(135deg,#ECEFF1,#F5F5F5)"

    st.markdown(f"""
    <div style="border:2px solid {border};border-radius:12px;padding:14px;
                background:{bg};text-align:center;margin:4px;">
        <div style="font-size:16px;font-weight:700;color:#1A237E">{algo_name} {badge}</div>
        <hr style="margin:6px 0;border-color:{border}">
        <div style="font-size:13px;color:#424242">
            <b>Tốt nhất:</b> {best_fitness:.4f}<br>
            <b>Trung bình:</b> {mean_fitness:.4f} ± {std:.4f}<br>
            <b>Thời gian:</b> {runtime:.3f}s
        </div>
    </div>
    """, unsafe_allow_html=True)
