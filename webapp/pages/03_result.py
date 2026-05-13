"""Trang 3 — Hiển thị kết quả đầy đủ."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import pandas as pd
from webapp.session import init_session, get_results
from webapp.components.sidebar_nav import show_sidebar_nav
from webapp.components.metric_card import show_algo_card
from webapp.components.result_table import show_allocation_table, show_summary_metrics
from webapp.components.chart_widget import show_chart
from analysis.comparator import build_summary_table, rank_algorithms, compute_improvement, get_best_allocation
from analysis.statistics import full_statistical_report, t_test_compare, compute_effect_size
from visualization.convergence import plot_convergence
from visualization.allocation  import plot_water_allocation
from visualization.boxplot     import plot_fitness_boxplot

init_session()
show_sidebar_nav()
st.title("Kết quả phân tích")

results = get_results()
if results is None:
    st.warning("Chưa có kết quả. Vui lòng chạy thuật toán trước.")
    if st.button("Đi đến trang chạy"):
        st.switch_page("pages/02_run.py")
    st.stop()

fields_data = st.session_state["fields_data"]
algo_names  = list(results.keys())

# ── Section 1: Metric cards ───────────────────────────────────────────────────
st.header("1. Tổng kết từng thuật toán")
ranked      = rank_algorithms(results)
best_algo   = ranked[0]
cols        = st.columns(len(algo_names))
for col, algo in zip(cols, algo_names):
    runs  = results[algo]
    fits  = [r["best_fitness"] for r in runs]
    rts   = [r["runtime"] for r in runs]
    with col:
        show_algo_card(
            algo_name    = algo,
            best_fitness = min(fits),
            mean_fitness = float(np.mean(fits)),
            std          = float(np.std(fits)),
            runtime      = float(np.mean(rts)),
            is_best      = (algo == best_algo),
        )

# ── Section 2: Bảng so sánh ──────────────────────────────────────────────────
st.header("2. Bảng so sánh chi tiết")
summary_df = build_summary_table(results)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("Tải CSV", csv, "summary.csv", "text/csv")

improvements = compute_improvement(results)
st.caption("% cải thiện so với GA: " +
           " | ".join(f"{k}: {v:+.1f}%" for k, v in improvements.items() if k != "GA"))

# ── Section 3: Biểu đồ ───────────────────────────────────────────────────────
st.header("3. Biểu đồ phân tích")
tab1, tab2, tab3 = st.tabs(["Hội tụ", "Phân bổ nước", "Boxplot"])

with tab1:
    histories = {algo: min(results[algo], key=lambda r: r["best_fitness"])["history"]
                 for algo in algo_names}
    fig = plot_convergence(histories, algo_names)
    show_chart(fig, caption="Đường hội tụ fitness theo iteration")

with tab2:
    best_alloc = get_best_allocation(results)
    sol        = best_alloc["solution"]
    fig2 = plot_water_allocation(sol, fields_data["demand_min"],
                                 fields_data["demand_max"],
                                 fields_data["field_names"])
    show_chart(fig2, caption=f"Phân bổ nước tốt nhất ({best_alloc['algo_name']})")

with tab3:
    all_fits = {a: [r["best_fitness"] for r in runs] for a, runs in results.items()}
    fig3 = plot_fitness_boxplot(all_fits, algo_names)
    show_chart(fig3, caption="Phân phối fitness qua các lần chạy")

# ── Section 4: Chi tiết phương án tốt nhất ───────────────────────────────────
st.header("4. Phân tích phương án tốt nhất")
best_alloc = get_best_allocation(results)
st.markdown(f"**Thuật toán:** {best_alloc['algo_name']} | **Điểm tối ưu:** {best_alloc['fitness']:.4f}")
show_allocation_table(best_alloc["solution"], fields_data)
show_summary_metrics(best_alloc["solution"], fields_data, fields_data["W_total"])

# ── Section 5: Thống kê ───────────────────────────────────────────────────────
st.header("5. Kết luận thống kê")
if len(algo_names) >= 2:
    report = full_statistical_report(results)
    st.code(report, language=None)

    if "GA" in results and "Hybrid" in results:
        fa = [r["best_fitness"] for r in results["GA"]]
        fb = [r["best_fitness"] for r in results["Hybrid"]]
        tt = t_test_compare(fa, fb)
        d  = compute_effect_size(fa, fb)
        sig_text = "Có ý nghĩa thống kê" if tt["significant"] else "Không có ý nghĩa thống kê"
        st.success(f"Hybrid vs GA: p={tt['p_value']:.4f} — {sig_text} | Cohen's d={d}")
else:
    st.info("Cần ít nhất 2 thuật toán để so sánh thống kê.")
