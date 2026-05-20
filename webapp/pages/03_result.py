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
from webapp.ai_explainer import render_comparison_ai_section
from analysis.comparator import build_summary_table, rank_algorithms, compute_improvement, get_best_allocation
from visualization.convergence import plot_convergence
from visualization.allocation  import plot_water_allocation
from visualization.boxplot     import plot_fitness_boxplot


def allocation_metrics(solution, fields_data):
    sol = np.array(solution, dtype=float)
    dmin = np.array(fields_data["demand_min"], dtype=float)
    dmax = np.array(fields_data["demand_max"], dtype=float)
    prices = np.array(fields_data["prices"], dtype=float)
    w_total = float(fields_data["W_total"])
    total = float(np.sum(sol))
    cost = float(np.dot(prices, sol))
    shortage = float(np.sum(np.maximum(0.0, dmin - sol)))
    field_surplus = float(np.sum(np.maximum(0.0, sol - dmax)))
    over_budget = float(max(0.0, total - w_total))
    target = float(min(w_total, np.sum(dmax)))
    under_target = float(max(0.0, target - total))
    return {
        "total": total,
        "cost": cost,
        "avg_cost": cost / max(total, 1e-9),
        "shortage": shortage,
        "surplus": field_surplus + over_budget,
        "under_target": under_target,
    }


def best_run_for_algo(results, algo):
    return min(results[algo], key=lambda r: r["best_fitness"])


def build_rank_explanations(results, ranked, fields_data):
    rows = []
    best_by_algo = {algo: best_run_for_algo(results, algo) for algo in ranked}
    metrics = {
        algo: allocation_metrics(best_by_algo[algo]["best_solution"], fields_data)
        for algo in ranked
    }
    best_cost_algo = min(ranked, key=lambda a: metrics[a]["cost"])
    best_water_algo = min(ranked, key=lambda a: metrics[a]["total"])
    best_clean_algo = min(ranked, key=lambda a: metrics[a]["shortage"] + metrics[a]["surplus"] + metrics[a]["under_target"])

    for pos, algo in enumerate(ranked, start=1):
        run = best_by_algo[algo]
        m = metrics[algo]
        if pos == 1:
            opening = f"{algo} xếp hạng 1 vì có điểm tổng hợp thấp nhất."
        else:
            gap = run["best_fitness"] - best_by_algo[ranked[0]]["best_fitness"]
            opening = f"{algo} xếp hạng {pos}, kém thuật toán đứng đầu khoảng {gap:.2f} điểm."

        reasons = []
        if algo == best_cost_algo:
            reasons.append(f"nó kiểm soát chi phí bơm tốt nhất, chỉ khoảng {m['cost']:.2f}")
        else:
            cost_gap = m["cost"] - metrics[best_cost_algo]["cost"]
            reasons.append(f"chi phí bơm cao hơn {best_cost_algo} khoảng {cost_gap:.2f}")

        if algo == best_water_algo:
            reasons.append(f"dùng ít nước nhất với {m['total']:.2f} m³")
        else:
            water_gap = m["total"] - metrics[best_water_algo]["total"]
            if water_gap > 0.01:
                reasons.append(f"dùng nhiều hơn {best_water_algo} khoảng {water_gap:.2f} m³")
            else:
                reasons.append("lượng nước dùng gần nhóm tốt nhất")

        issue = m["shortage"] + m["surplus"] + m["under_target"]
        if algo == best_clean_algo and issue <= 0.01:
            reasons.append("phương án gần như sạch ràng buộc, không thiếu và không dư đáng kể")
        elif issue > 0.01:
            reasons.append(
                f"vẫn còn {m['shortage']:.2f} m³ thiếu, {m['surplus']:.2f} m³ dư/vượt ngưỡng "
                f"và {m['under_target']:.2f} m³ thiếu mục tiêu"
            )
        else:
            reasons.append("ràng buộc nước đã được xử lý ổn")

        tradeoff = ""
        if algo == best_cost_algo and algo != best_water_algo:
            tradeoff = " Nói dễ hiểu: thuật toán này có thể không tiết kiệm nước nhất, nhưng biết đặt nước vào các thửa rẻ hơn nên chi phí đẹp."
        elif algo == best_water_algo and algo != best_cost_algo:
            tradeoff = " Nói dễ hiểu: thuật toán này tiết kiệm nước tốt, nhưng chưa chắc là rẻ nhất vì nước có thể nằm ở thửa bơm đắt."

        rows.append(
            {
                "Hạng": pos,
                "Thuật toán": algo,
                "Diễn giải": opening + " Cụ thể, " + "; ".join(reasons) + "." + tradeoff,
            }
        )
    return pd.DataFrame(rows)


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

st.header("2. Bảng so sánh chi tiết")
summary_df = build_summary_table(results)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("Tải CSV", csv, "summary.csv", "text/csv")

improvements = compute_improvement(results)
st.caption("% cải thiện so với GA: " +
           " | ".join(f"{k}: {v:+.1f}%" for k, v in improvements.items() if k != "GA"))

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

st.header("4. Phân tích phương án tốt nhất")
best_alloc = get_best_allocation(results)
st.markdown(f"**Thuật toán:** {best_alloc['algo_name']} | **Điểm tối ưu:** {best_alloc['fitness']:.4f}")
show_allocation_table(best_alloc["solution"], fields_data)
show_summary_metrics(best_alloc["solution"], fields_data, fields_data["W_total"])

st.header("5. Kết luận xếp hạng")
rank_text_df = build_rank_explanations(results, ranked, fields_data)
for _, row in rank_text_df.iterrows():
    if row["Hạng"] == 1:
        st.success(f"**Hạng {row['Hạng']} - {row['Thuật toán']}**  \n{row['Diễn giải']}")
    else:
        st.info(f"**Hạng {row['Hạng']} - {row['Thuật toán']}**  \n{row['Diễn giải']}")

if len(algo_names) == 1:
    st.caption("Hiện mới có một thuật toán được chạy, nên phần xếp hạng chỉ giải thích riêng thuật toán đó.")

render_comparison_ai_section(results, ranked)
