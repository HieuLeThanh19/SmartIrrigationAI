"""Trang 1 — Nhập và chỉnh sửa thông số bài toán."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from webapp.components.sidebar_nav import show_sidebar_nav
import webapp.session as session_state

session_state.init_session()
if hasattr(session_state, "sync_budget_state"):
    session_state.sync_budget_state()
show_sidebar_nav()
st.title("Nhập dữ liệu bài toán")

# ── Section 1: Thông số tổng quan ────────────────────────────────────────────
st.header("1. Thông số tổng quan")
col1, col2 = st.columns(2)
with col1:
    W_total = st.slider("Tổng nước W_total (m³/ngày)", 200, 500,
                        int(st.session_state["W_total"]), step=10)
with col2:
    n_runs = st.number_input("Số lần chạy mỗi thuật toán",
                             min_value=1, max_value=60,
                             value=min(60, max(1, int(st.session_state["n_runs"]))))

# Đồng bộ ngay khi người dùng thay đổi để chuyển trang bằng sidebar vẫn đúng dữ liệu mới.
st.session_state["W_total"] = float(W_total)
st.session_state["n_runs"] = int(n_runs)
st.session_state["fields_data"]["W_total"] = float(W_total)

# ── Section 2: Bảng thửa ruộng ───────────────────────────────────────────────
st.header("2. Chỉnh sửa dữ liệu thửa ruộng")
fields_data = st.session_state["fields_data"]
df_edit = pd.DataFrame({
    "Thửa":      fields_data["field_names"],
    "Cây trồng": fields_data["crop_names"],
    "Min (m³)":  fields_data["demand_min"].tolist(),
    "Max (m³)":  fields_data["demand_max"].tolist(),
    "Giá (đ/m³)": fields_data["prices"].tolist(),
})
edited = st.data_editor(df_edit, use_container_width=True, num_rows="fixed",
                        column_config={
                            "Min (m³)":   st.column_config.NumberColumn(min_value=1, max_value=200),
                            "Max (m³)":   st.column_config.NumberColumn(min_value=1, max_value=300),
                            "Giá (đ/m³)": st.column_config.NumberColumn(min_value=0.5, max_value=5.0, step=0.5),
                        })

# Validate
ok = all(edited["Min (m³)"][i] < edited["Max (m³)"][i] for i in range(len(edited)))
if not ok:
    st.error("demand_min phải nhỏ hơn demand_max!")

# ── Section 3: Chọn thuật toán ───────────────────────────────────────────────
st.header("3. Chọn thuật toán")
col1, col2, col3, col4 = st.columns(4)
sel = {
    "GA":     col1.checkbox("GA",           value="GA"     in st.session_state["selected_algos"]),
    "SA":     col2.checkbox("SA",            value="SA"     in st.session_state["selected_algos"]),
    "PSO":    col3.checkbox("PSO",           value="PSO"    in st.session_state["selected_algos"]),
    "Hybrid": col4.checkbox("Hybrid GA+SA", value="Hybrid" in st.session_state["selected_algos"]),
}
selected = [k for k, v in sel.items() if v]
if not selected:
    st.warning("Chọn ít nhất 1 thuật toán!")
else:
    st.session_state["selected_algos"] = selected

# ── Section 4: Xem trước ─────────────────────────────────────────────────────
st.header("4. Xem trước dữ liệu")
dmins = np.array(edited["Min (m³)"])
dmaxs = np.array(edited["Max (m³)"])
names = edited["Thửa"].tolist()
x     = np.arange(len(names))
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x - 0.2, dmins, 0.4, label="Min", color="#EF5350", alpha=0.8)
ax.bar(x + 0.2, dmaxs, 0.4, label="Max", color="#FFA726", alpha=0.8)
ax.axhline(W_total / len(names), linestyle="--", color="#42A5F5",
           label=f"Nước bình quân/thửa ({W_total/len(names):.0f} m³)")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha="right")
ax.set_ylabel("m³/ngày")
ax.set_title("Nhu cầu nước min/max từng thửa")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

total_min = dmins.sum()
total_max = dmaxs.sum()
budget_ok = total_min <= W_total
if not budget_ok:
    st.error(
        f"Tổng nhu cầu tối thiểu đang là {total_min:.0f} m³, lớn hơn ngân sách {W_total:.0f} m³. "
        "Hãy giảm Min hoặc tăng tổng nước trước khi chạy."
    )
c1, c2, c3 = st.columns(3)
c1.metric("Tổng nhu cầu min", f"{total_min:.0f} m³")
c2.metric("Tổng ngân sách nước", f"{W_total} m³")
c3.metric("Tổng nhu cầu max", f"{total_max:.0f} m³")

# ── Nút Tiếp theo ────────────────────────────────────────────────────────────
st.divider()
if st.button("Tiếp theo → Chạy thuật toán", type="primary",
             disabled=(not ok or not selected or not budget_ok), use_container_width=True):
    st.session_state["fields_data"] = {
        "demand_min":  np.array(edited["Min (m³)"], dtype=float),
        "demand_max":  np.array(edited["Max (m³)"], dtype=float),
        "prices":      np.array(edited["Giá (đ/m³)"], dtype=float),
        "W_total":     W_total,
        "field_names": edited["Thửa"].tolist(),
        "crop_names":  edited["Cây trồng"].tolist(),
    }
    st.session_state["W_total"]        = W_total
    st.session_state["n_runs"]         = n_runs
    st.session_state["selected_algos"] = selected
    st.switch_page("pages/02_run.py")
