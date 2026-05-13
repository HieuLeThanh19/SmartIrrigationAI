"""Trang 1 - Thiet lap bai toan dieu phoi nuoc."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core import config as cfg
from webapp.components.sidebar_nav import show_sidebar_nav
import webapp.session as session_state


DEMO_STEPS = 30

TEMP_TIERS = {
    "20-25°C - Lý tưởng": {
        "min_factor": 1.00,
        "max_factor": 1.00,
        "note": "Giữ nguyên nhu cầu chuẩn.",
    },
    "26-32°C - Nắng ấm": {
        "min_factor": 1.15,
        "max_factor": 1.10,
        "note": "Tăng nước tối thiểu để bù bốc hơi nhẹ.",
    },
    "33-38°C - Nắng gắt": {
        "min_factor": 1.30,
        "max_factor": 1.25,
        "note": "Tăng mạnh nhu cầu để chống stress nhiệt.",
    },
    ">38°C - Cực đoan": {
        "min_factor": 1.50,
        "max_factor": 1.40,
        "note": "Ưu tiên an toàn cây, làm mát vùng rễ.",
    },
}

SCENARIOS = {
    "Cân bằng mặc định": {
        "desc": "Nguồn nước vừa đủ, giá bơm chênh nhẹ, phù hợp để xem thuật toán cân bằng cơ bản.",
        "W_total": 260,
        "demand_min": [20, 18, 15, 16, 10, 10, 12, 12, 25, 22],
        "demand_max": [40, 38, 32, 30, 22, 20, 25, 24, 45, 40],
        "prices": [1.2, 1.1, 1.5, 1.4, 1.8, 1.7, 1.6, 1.5, 1.0, 1.1],
    },
    "Hạn nước cuối mùa": {
        "desc": "Tổng nước thấp, thuật toán phải ưu tiên thửa thiếu nghiêm trọng và tránh lãng phí.",
        "W_total": 235,
        "demand_min": [22, 20, 16, 17, 12, 11, 14, 13, 27, 24],
        "demand_max": [45, 42, 34, 33, 25, 23, 29, 27, 51, 46],
        "prices": [1.3, 1.2, 1.7, 1.6, 2.0, 1.9, 1.8, 1.7, 1.1, 1.2],
    },
    "Nắng gắt thiếu nước": {
        "desc": "Nhiều thửa cần nước cao hơn bình thường, tạo tình huống thiếu/dư rõ khi demo.",
        "W_total": 280,
        "demand_min": [24, 22, 18, 19, 13, 12, 15, 15, 30, 27],
        "demand_max": [48, 46, 36, 35, 27, 25, 31, 30, 55, 50],
        "prices": [1.4, 1.25, 1.65, 1.55, 2.1, 1.95, 1.75, 1.7, 1.05, 1.15],
    },
    "Chênh giá bơm mạnh": {
        "desc": "Một số thửa bơm rất rẻ, một số thửa rất đắt, giúp thấy thuật toán chuyển nước để giảm chi phí.",
        "W_total": 255,
        "demand_min": [18, 18, 14, 15, 10, 10, 12, 12, 24, 22],
        "demand_max": [44, 42, 34, 32, 25, 24, 27, 26, 48, 45],
        "prices": [0.8, 0.9, 2.2, 2.0, 2.8, 2.6, 1.9, 1.8, 0.7, 0.8],
    },
    "Cuối kênh thiếu áp": {
        "desc": "Các thửa cuối kênh cần biên an toàn lớn hơn, giá bơm cũng cao hơn vì phải đẩy nước xa.",
        "W_total": 270,
        "demand_min": [18, 18, 15, 16, 12, 12, 16, 17, 29, 28],
        "demand_max": [38, 37, 32, 31, 25, 25, 34, 35, 55, 53],
        "prices": [1.0, 1.05, 1.25, 1.3, 1.55, 1.6, 2.0, 2.1, 2.3, 2.4],
    },
    "Giờ cao điểm bơm": {
        "desc": "Nguồn nước không quá thiếu nhưng chi phí điện/bơm cao, thuật toán phải né thửa đắt khi có thể.",
        "W_total": 265,
        "demand_min": [20, 19, 15, 16, 11, 11, 13, 13, 26, 23],
        "demand_max": [42, 40, 33, 31, 24, 22, 27, 26, 48, 43],
        "prices": [1.6, 1.45, 2.4, 2.25, 3.0, 2.8, 2.3, 2.1, 1.3, 1.35],
    },
}


def apply_temperature(base_min, base_max, tier_name):
    tier = TEMP_TIERS[tier_name]
    adjusted_min = np.array(base_min, dtype=float) * tier["min_factor"]
    adjusted_max = np.array(base_max, dtype=float) * tier["max_factor"]
    adjusted_max = np.maximum(adjusted_max, adjusted_min + 1.0)
    return adjusted_min, adjusted_max


def apply_scenario(name, tier_name):
    scenario = SCENARIOS[name]
    current = st.session_state["fields_data"]
    dmins, dmaxs = apply_temperature(scenario["demand_min"], scenario["demand_max"], tier_name)
    min_total = float(np.sum(dmins))
    max_total = float(np.sum(dmaxs))
    proposed_w = float(scenario["W_total"])
    w_total = float(np.clip(proposed_w, np.ceil(min_total), np.floor(max_total)))
    st.session_state["fields_data"] = {
        "demand_min": dmins,
        "demand_max": dmaxs,
        "prices": np.array(scenario["prices"], dtype=float),
        "W_total": w_total,
        "field_names": current["field_names"],
        "crop_names": current["crop_names"],
        "temperature_mode": tier_name,
        "scenario_name": name,
    }
    st.session_state["W_total"] = w_total
    st.session_state["run_results"] = None
    st.session_state["is_ran"] = False


def pressure_message(pressure):
    if pressure < 0.18:
        return "Rất căng: gần mức tối thiểu, thuật toán phải xử lý thiếu nước rõ.", "warning"
    if pressure < 0.45:
        return "Căng vừa: đủ nước cơ bản nhưng vẫn cần ưu tiên thửa quan trọng/rẻ.", "success"
    if pressure < 0.82:
        return "Cân bằng: có nhiều cách điều phối, phù hợp để so sánh thuật toán.", "success"
    return "Dư tương đối: bài toán nghiêng về tránh vượt Max và giảm chi phí bơm.", "warning"


session_state.init_session()
if hasattr(session_state, "sync_budget_state"):
    session_state.sync_budget_state()
show_sidebar_nav()

st.markdown(
    """
<style>
.input-hero {
    border: 1px solid #D7E3DA;
    border-radius: 8px;
    padding: 14px 16px;
    background: #F6FBF7;
}
.input-hero h3 { margin: 0 0 4px 0; }
.soft-note {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 12px;
    background: #FFFFFF;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Thiết lập điều phối nước")
st.markdown(
    """
<div class="input-hero">
  <h3>Bài toán vận hành</h3>
  <p>Chọn điều kiện thực tế, nhiệt độ và ngân sách nước. Hệ thống sẽ biến chúng thành Min/Max, giá bơm và mức căng để thuật toán xử lý.</p>
</div>
""",
    unsafe_allow_html=True,
)

fields_data = st.session_state["fields_data"]
default_scenario = fields_data.get("scenario_name", "Cân bằng mặc định")
default_temp = fields_data.get("temperature_mode", "20-25°C - Lý tưởng")

st.header("1. Điều kiện vận hành")
scenario_col, temp_col, action_col = st.columns([1.35, 1.05, 0.55])
with scenario_col:
    scenario_name = st.selectbox(
        "Tình huống điều phối",
        list(SCENARIOS.keys()),
        index=list(SCENARIOS.keys()).index(default_scenario) if default_scenario in SCENARIOS else 0,
    )
    st.caption(SCENARIOS[scenario_name]["desc"])

with temp_col:
    temp_name = st.selectbox(
        "Nhiệt độ ngoài ruộng",
        list(TEMP_TIERS.keys()),
        index=list(TEMP_TIERS.keys()).index(default_temp) if default_temp in TEMP_TIERS else 0,
    )
    temp_cfg = TEMP_TIERS[temp_name]
    st.caption(
        f"Min x{temp_cfg['min_factor']:.2f}, Max x{temp_cfg['max_factor']:.2f}. {temp_cfg['note']}"
    )

with action_col:
    st.write("")
    st.write("")
    if st.button("Áp dụng", type="primary", use_container_width=True):
        apply_scenario(scenario_name, temp_name)
        st.rerun()

base_min = np.array(SCENARIOS[scenario_name]["demand_min"], dtype=float)
base_max = np.array(SCENARIOS[scenario_name]["demand_max"], dtype=float)
preview_min, preview_max = apply_temperature(base_min, base_max, temp_name)
delta_min = float(np.sum(preview_min) - np.sum(base_min))
delta_max = float(np.sum(preview_max) - np.sum(base_max))

t1, t2, t3, t4 = st.columns(4)
t1.metric("Tổng Min sau nhiệt độ", f"{np.sum(preview_min):.0f} m³", f"{delta_min:+.0f}")
t2.metric("Tổng Max sau nhiệt độ", f"{np.sum(preview_max):.0f} m³", f"{delta_max:+.0f}")
t3.metric("Hệ số Min", f"x{temp_cfg['min_factor']:.2f}")
t4.metric("Hệ số Max", f"x{temp_cfg['max_factor']:.2f}")

st.header("2. Bảng thửa ruộng")
df_edit = pd.DataFrame(
    {
        "Thửa": fields_data["field_names"],
        "Cây trồng": fields_data["crop_names"],
        "Min cần tưới (m³)": np.array(fields_data["demand_min"], dtype=float),
        "Max an toàn (m³)": np.array(fields_data["demand_max"], dtype=float),
        "Giá bơm/m³": np.array(fields_data["prices"], dtype=float),
    }
)

edited = st.data_editor(
    df_edit,
    use_container_width=True,
    num_rows="fixed",
    disabled=["Thửa", "Cây trồng"],
    column_config={
        "Min cần tưới (m³)": st.column_config.NumberColumn(min_value=1, max_value=300, step=1),
        "Max an toàn (m³)": st.column_config.NumberColumn(min_value=1, max_value=350, step=1),
        "Giá bơm/m³": st.column_config.NumberColumn(min_value=0.5, max_value=6.0, step=0.1),
    },
)

dmins = np.array(edited["Min cần tưới (m³)"], dtype=float)
dmaxs = np.array(edited["Max an toàn (m³)"], dtype=float)
prices = np.array(edited["Giá bơm/m³"], dtype=float)
names = edited["Thửa"].tolist()
crops = edited["Cây trồng"].tolist()

range_ok = bool(np.all(dmins < dmaxs))
if not range_ok:
    st.error("Mỗi thửa cần có Min nhỏ hơn Max để thuật toán có khoảng điều phối.")

total_min = float(np.sum(dmins))
total_max = float(np.sum(dmaxs))
current_w = float(st.session_state.get("W_total", cfg.W_TOTAL))
slider_min = int(np.ceil(total_min))
slider_max = int(np.floor(total_max))
slider_value = int(np.clip(current_w, slider_min, slider_max)) if slider_min <= slider_max else slider_min

st.header("3. Nguồn nước và thuật toán")
budget_col, algo_col = st.columns([1.15, 1])
with budget_col:
    if slider_min <= slider_max:
        W_total = st.slider(
            "Tổng nước có thể điều phối W_total (m³/ngày)",
            min_value=slider_min,
            max_value=slider_max,
            value=slider_value,
            step=5,
        )
    else:
        W_total = slider_value
        st.error("Dữ liệu Min/Max đang không hợp lệ nên chưa thể chọn nguồn nước.")

    pressure = (float(W_total) - total_min) / (total_max - total_min) if total_max > total_min else 0.0
    msg, msg_type = pressure_message(pressure)
    if msg_type == "warning":
        st.warning(msg)
    else:
        st.success(msg)

with algo_col:
    st.markdown("**Thuật toán demo**")
    a1, a2 = st.columns(2)
    with a1:
        ga = st.checkbox("GA", value="GA" in st.session_state["selected_algos"])
        pso = st.checkbox("PSO", value="PSO" in st.session_state["selected_algos"])
    with a2:
        sa = st.checkbox("SA", value="SA" in st.session_state["selected_algos"])
        hybrid = st.checkbox("Hybrid GA+SA", value="Hybrid" in st.session_state["selected_algos"])
    sel = {"GA": ga, "SA": sa, "PSO": pso, "Hybrid": hybrid}
    selected = [k for k, v in sel.items() if v]
    st.caption(f"Mỗi thuật toán chạy cố định {DEMO_STEPS} bước.")

if not selected:
    st.warning("Chọn ít nhất 1 thuật toán.")
else:
    st.session_state["selected_algos"] = selected

st.session_state["W_total"] = float(W_total)
st.session_state["n_runs"] = DEMO_STEPS
st.session_state["fields_data"] = {
    "demand_min": dmins,
    "demand_max": dmaxs,
    "prices": prices,
    "W_total": float(W_total),
    "field_names": names,
    "crop_names": crops,
    "temperature_mode": temp_name,
    "scenario_name": scenario_name,
}

st.header("4. Dashboard trước khi chạy")
x = np.arange(len(names))
fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.bar(x - 0.2, dmins, 0.35, label="Min cần tưới", color="#EF5350", alpha=0.85)
ax1.bar(x + 0.15, dmaxs - dmins, 0.35, bottom=dmins, label="Biên điều phối", color="#FFA726", alpha=0.75)
ax1.axhline(float(W_total) / len(names), linestyle="--", color="#2E7D32", label=f"Bình quân {float(W_total)/len(names):.1f} m³/thửa")
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=25, ha="right")
ax1.set_ylabel("Nước (m³/ngày)")
ax1.grid(axis="y", alpha=0.25)

ax2 = ax1.twinx()
ax2.plot(x, prices, marker="o", color="#1565C0", linewidth=2, label="Giá bơm/m³")
ax2.set_ylabel("Giá bơm/m³")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_title("Nhu cầu nước, biên điều phối và chi phí bơm từng thửa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

extra_water = float(W_total - total_min)
extra_capacity = float(total_max - total_min)
pressure = extra_water / extra_capacity if extra_capacity > 0 else 0.0
cheap_threshold = float(np.percentile(prices, 33))
expensive_threshold = float(np.percentile(prices, 67))
cheap_capacity = float(np.sum(np.maximum(0.0, dmaxs[prices <= cheap_threshold] - dmins[prices <= cheap_threshold])))
expensive_min = float(np.sum(dmins[prices >= expensive_threshold]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng Min", f"{total_min:.0f} m³")
c2.metric("W_total", f"{float(W_total):.0f} m³")
c3.metric("Tổng Max", f"{total_max:.0f} m³")
c4.metric("Mức căng", f"{pressure * 100:.0f}%")

d1, d2, d3 = st.columns(3)
with d1:
    st.markdown(
        f"<div class='soft-note'><b>Nước dư để điều phối</b><br>{extra_water:.1f} m³ trên {extra_capacity:.1f} m³ biên khả dụng.</div>",
        unsafe_allow_html=True,
    )
with d2:
    st.markdown(
        f"<div class='soft-note'><b>Biên thửa giá rẻ</b><br>{cheap_capacity:.1f} m³ có thể nhận thêm nếu còn hợp lệ.</div>",
        unsafe_allow_html=True,
    )
with d3:
    st.markdown(
        f"<div class='soft-note'><b>Min nhóm giá cao</b><br>{expensive_min:.1f} m³ là phần khó cắt vì vẫn phải đảm bảo cây.</div>",
        unsafe_allow_html=True,
    )

priority_df = pd.DataFrame(
    {
        "Thửa": names,
        "Cây trồng": crops,
        "Min": np.round(dmins, 1),
        "Max": np.round(dmaxs, 1),
        "Biên thêm": np.round(dmaxs - dmins, 1),
        "Giá bơm/m³": np.round(prices, 2),
        "Vai trò": np.where(
            prices <= cheap_threshold,
            "Ưu tiên nhận thêm",
            np.where(prices >= expensive_threshold, "Hạn chế nếu có thể", "Cân bằng"),
        ),
    }
).sort_values(["Vai trò", "Giá bơm/m³"])

st.dataframe(priority_df, use_container_width=True, hide_index=True)

st.divider()
can_continue = range_ok and bool(selected) and slider_min <= slider_max
if st.button("Tiếp theo -> Chạy thuật toán", type="primary", disabled=not can_continue, use_container_width=True):
    st.switch_page("pages/02_run.py")
