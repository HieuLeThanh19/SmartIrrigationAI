"""Trang 2 — Demo từng thuật toán riêng lẻ."""
import math
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core import config as cfg
from core.constraints import clip_to_bounds, repair_solution
from core.fitness import fitness as calculate_fitness
from core.fitness import fitness_breakdown
from webapp.ai_explainer import render_run_ai_section
from webapp.components.sidebar_nav import show_sidebar_nav
import webapp.session as session_state

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

session_state.init_session()
if hasattr(session_state, "sync_budget_state"):
    session_state.sync_budget_state()
show_sidebar_nav()

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.field-card { border-radius:12px; padding:8px; margin:4px; text-align:center; box-shadow:0 1px 6px rgba(0,0,0,.08); }
.ok { background:#E8F5E9; border:1px solid #4CAF50; }
.warn { background:#FFFDE7; border:1px solid #FFC107; }
.bad { background:#FFEBEE; border:1px solid #EF5350; }
.popup { border-radius:10px; padding:10px; background:#F3F6FF; border:1px solid #C5CAE9; }
.decision-box { border-radius:10px; padding:14px 16px; background:#FFFDF5; border:1px solid #F4C95D; line-height:1.55; }
.run-summary-card { border-radius:10px; padding:14px 16px; background:#F8FBFF; border:1px solid #B8CCE8; margin-bottom:10px; }
.run-summary-card b { color:#173A5E; }
</style>
""",
    unsafe_allow_html=True,
)

fields = st.session_state["fields_data"]
demand_min = np.array(fields["demand_min"], dtype=float)
demand_max = np.array(fields["demand_max"], dtype=float)
prices = np.array(fields["prices"], dtype=float)
field_names = fields["field_names"]
crop_names = fields["crop_names"]
W_total = float(fields.get("W_total", st.session_state["W_total"]))
st.session_state["W_total"] = W_total
n_fields = len(demand_min)
AUTO_DELAY_SECONDS = 3.0
DEFAULT_DEMO_STEPS = 30
DEMO_REPAIR_START_STEP = 4
DEMO_REPAIR_FULL_STEP = 14
ALGO_LABELS = {
    "GA": "GA - Genetic Algorithm",
    "SA": "SA - Simulated Annealing",
    "PSO": "PSO - Particle Swarm Optimization",
    "Hybrid": "Hybrid GA + SA",
}


def fit(x):
    return calculate_fitness(x, demand_min, demand_max, prices, W_total)


def rand_sol():
    x = np.random.uniform(demand_min, demand_max)
    return repair_solution(x, demand_min, demand_max, W_total)


def demo_loose_bounds():
    span = np.maximum(demand_max - demand_min, 1.0)
    return demand_min - 0.35 * span, demand_max + 0.35 * span


def rough_demo_sol():
    """Tạo nghiệm đầu cố tình có cả thiếu và dư để demo thuật toán đang sửa lỗi."""
    lo, hi = demo_loose_bounds()
    x = np.random.uniform(lo, hi)
    if n_fields >= 4:
        low_idx = np.random.choice(n_fields, 2, replace=False)
        remaining = [i for i in range(n_fields) if i not in set(low_idx)]
        high_idx = np.random.choice(remaining, 2, replace=False)
        span = demand_max - demand_min
        x[low_idx] = demand_min[low_idx] - np.random.uniform(0.12, 0.30, len(low_idx)) * span[low_idx]
        x[high_idx] = demand_max[high_idx] + np.random.uniform(0.12, 0.30, len(high_idx)) * span[high_idx]
    return np.clip(x, lo, hi)


def repair_ratio(step):
    if step <= DEMO_REPAIR_START_STEP:
        return 0.0
    return min(1.0, (step - DEMO_REPAIR_START_STEP) / (DEMO_REPAIR_FULL_STEP - DEMO_REPAIR_START_STEP))


def demo_candidate(x, step=None):
    """Những bước đầu giữ lỗi thiếu/dư; về sau tăng dần sửa ràng buộc."""
    if step is None:
        step = int(st.session_state.get("demo_step", 0))
    lo, hi = demo_loose_bounds()
    raw = np.clip(np.array(x, dtype=float), lo, hi)
    fixed = repair_solution(clip_to_bounds(raw, demand_min, demand_max), demand_min, demand_max, W_total)
    ratio = repair_ratio(step)
    return raw * (1.0 - ratio) + fixed * ratio


def clip_demo(x):
    return repair_solution(clip_to_bounds(x, demand_min, demand_max), demand_min, demand_max, W_total)


def final_repair(x):
    """Đảm bảo nghiệm cuối cùng đủ điều kiện trình bày kết quả tối ưu hợp lệ."""
    return repair_solution(clip_to_bounds(x, demand_min, demand_max), demand_min, demand_max, W_total)


def calc_metrics(sol):
    total = float(np.sum(sol))
    excess_field = float(np.sum(np.maximum(0.0, sol - demand_max)))
    over_budget = float(max(0.0, total - W_total))
    target = float(min(W_total, np.sum(demand_max)))
    under_target = float(max(0.0, target - total))
    return {
        "total": total,
        "target": target,
        "cost": float(np.dot(prices, sol)),
        "shortage": float(np.sum(np.maximum(0.0, demand_min - sol))),
        "excess_field": excess_field,
        "over_budget": over_budget,
        "under_target": under_target,
        "surplus": excess_field + over_budget,
    }


def calc_score_parts(sol):
    bd = fitness_breakdown(sol, demand_min, demand_max, prices, W_total)
    return {
        "cost": bd["cost_raw"],
        "shortage": bd["shortage"],
        "excess_field": bd["waste"],
        "over_budget": bd["over_budget"],
        "under_target": bd["under_target"],
        "cost_part": bd["cost"],
        "shortage_penalty": bd["shortage_penalty"],
        "surplus_penalty": bd["waste_penalty"],
        "underuse_penalty": bd["underuse_penalty"],
        "score": bd["total"],
    }


def build_field_explanation(prev_sol, cur_sol):
    rows = []
    for i, (prev, cur) in enumerate(zip(prev_sol, cur_sol)):
        mn, mx = float(demand_min[i]), float(demand_max[i])
        shortage = max(0.0, mn - float(cur))
        surplus = max(0.0, float(cur) - mx)
        if shortage > 0:
            status = "Thiếu"
            reason = f"Thiếu {shortage:.2f} m³ so với mức tối thiểu {mn:.0f}."
        elif surplus > 0:
            status = "Dư"
            reason = f"Dư {surplus:.2f} m³ so với mức tối đa {mx:.0f}; điểm bị phạt để lần sau giảm nước ở thửa này."
        else:
            status = "Hợp lệ"
            reason = f"Nằm trong khoảng cho phép {mn:.0f} đến {mx:.0f} m³."

        rows.append(
            {
                "Thửa": field_names[i],
                "Cây trồng": crop_names[i],
                "Trước bước": round(float(prev), 2),
                "Sau bước": round(float(cur), 2),
                "Thay đổi": round(float(cur - prev), 2),
                "Min": round(mn, 1),
                "Max": round(mx, 1),
                "Trạng thái": status,
                "Diễn giải": reason,
            }
        )
    return pd.DataFrame(rows)


def build_priority_dashboard(prev_sol, cur_sol):
    rows = []
    price_order = np.argsort(prices)
    priority_rank = {int(idx): rank + 1 for rank, idx in enumerate(price_order)}
    for i, (prev, cur) in enumerate(zip(prev_sol, cur_sol)):
        delta_water = float(cur - prev)
        delta_cost = float(delta_water * prices[i])
        if priority_rank[i] <= max(1, n_fields // 3):
            priority = "Ưu tiên cao"
        elif priority_rank[i] >= n_fields - max(1, n_fields // 3) + 1:
            priority = "Hạn chế"
        else:
            priority = "Trung bình"

        if delta_water > 0.01:
            action = "Tăng nước"
        elif delta_water < -0.01:
            action = "Giảm nước"
        else:
            action = "Giữ gần nguyên"

        rows.append(
            {
                "Thửa": field_names[i],
                "Cây trồng": crop_names[i],
                "Giá bơm/m³": round(float(prices[i]), 2),
                "Mức ưu tiên": priority,
                "Thứ hạng giá rẻ": priority_rank[i],
                "Nước thay đổi": round(delta_water, 2),
                "Chi phí thay đổi": round(delta_cost, 2),
                "Hành động": action,
            }
        )
    return pd.DataFrame(rows)


def render_cost_priority_dashboard():
    prev_sol = st.session_state.get("demo_prev_sol")
    cur_sol = st.session_state.get("demo_current")
    if prev_sol is None or cur_sol is None:
        st.info("Chạy ít nhất 1 bước để xem dashboard ưu tiên theo chi phí từng thửa.")
        return

    prev_sol = np.array(prev_sol, dtype=float)
    cur_sol = np.array(cur_sol, dtype=float)
    prev_total = float(np.sum(prev_sol))
    cur_total = float(np.sum(cur_sol))
    prev_cost = float(np.dot(prices, prev_sol))
    cur_cost = float(np.dot(prices, cur_sol))
    prev_avg = prev_cost / max(prev_total, 1e-9)
    cur_avg = cur_cost / max(cur_total, 1e-9)
    dashboard_df = build_priority_dashboard(prev_sol, cur_sol)

    cheap_cutoff = np.percentile(prices, 33)
    expensive_cutoff = np.percentile(prices, 67)
    added_to_cheap = float(dashboard_df.loc[(prices <= cheap_cutoff) & (dashboard_df["Nước thay đổi"] > 0), "Nước thay đổi"].sum())
    removed_from_expensive = float(-dashboard_df.loc[(prices >= expensive_cutoff) & (dashboard_df["Nước thay đổi"] < 0), "Nước thay đổi"].sum())

    st.subheader("Dashboard ưu tiên chi phí từng thửa")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Chi phí trung bình/m³", f"{cur_avg:.2f}", f"{cur_avg - prev_avg:+.2f}")
    k2.metric("Tổng chi phí bơm", f"{cur_cost:.2f}", f"{cur_cost - prev_cost:+.2f}")
    k3.metric("Nước thêm vào thửa rẻ", f"{added_to_cheap:.2f} m³")
    k4.metric("Nước giảm ở thửa đắt", f"{removed_from_expensive:.2f} m³")

    focus_df = dashboard_df.sort_values(
        by=["Nước thay đổi", "Chi phí thay đổi"],
        key=lambda col: col.abs() if col.name in ("Nước thay đổi", "Chi phí thay đổi") else col,
        ascending=False,
    ).head(6)
    st.dataframe(
        focus_df[
            [
                "Thửa",
                "Cây trồng",
                "Giá bơm/m³",
                "Mức ưu tiên",
                "Nước thay đổi",
                "Chi phí thay đổi",
                "Hành động",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    cheapest = dashboard_df.sort_values("Giá bơm/m³").head(3)
    expensive = dashboard_df.sort_values("Giá bơm/m³", ascending=False).head(3)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Nhóm giá bơm rẻ, thường được ưu tiên nhận thêm nếu còn thiếu nước")
        st.dataframe(
            cheapest[["Thửa", "Cây trồng", "Giá bơm/m³", "Nước thay đổi", "Hành động"]],
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.caption("Nhóm giá bơm cao, thường bị giảm trước nếu đang dư hoặc có thể chuyển nước")
        st.dataframe(
            expensive[["Thửa", "Cây trồng", "Giá bơm/m³", "Nước thay đổi", "Hành động"]],
            use_container_width=True,
            hide_index=True,
        )


def field_change_summary(prev_sol, cur_sol):
    df = build_priority_dashboard(prev_sol, cur_sol)
    inc = df[df["Nước thay đổi"] > 0.01].sort_values("Nước thay đổi", ascending=False)
    dec = df[df["Nước thay đổi"] < -0.01].sort_values("Nước thay đổi")
    cheap_cutoff = float(np.percentile(prices, 33))
    expensive_cutoff = float(np.percentile(prices, 67))
    cheap_inc = df[(df["Giá bơm/m³"] <= cheap_cutoff) & (df["Nước thay đổi"] > 0.01)]
    expensive_dec = df[(df["Giá bơm/m³"] >= expensive_cutoff) & (df["Nước thay đổi"] < -0.01)]
    return df, inc, dec, cheap_inc, expensive_dec


def format_field(row):
    return f"{row['Thửa']} ({row['Cây trồng']}, giá {row['Giá bơm/m³']:.2f}/m³)"


def build_decision_narrative(algo, step, info, prev_sol, cur_sol):
    if cur_sol is None:
        return "Chạy một bước để xem hướng ưu tiên điều phối nước."

    cur_sol = np.array(cur_sol, dtype=float)
    shortage = np.maximum(0.0, demand_min - cur_sol)
    field_surplus = np.maximum(0.0, cur_sol - demand_max)
    capacity = np.maximum(0.0, demand_max - cur_sol)
    total = float(np.sum(cur_sol))
    over_budget = max(0.0, total - W_total)
    under_target = max(0.0, min(W_total, float(np.sum(demand_max))) - total)
    metrics = calc_metrics(cur_sol)

    def fname(idx):
        return field_names[int(idx)]

    if float(np.sum(shortage)) > 0.01:
        add_idx = int(np.argmax(shortage))
        if float(np.sum(field_surplus)) > 0.01:
            cut_idx = int(np.argmax(field_surplus))
            return f"Dashboard thiếu {metrics['shortage']:.2f}; bù {fname(add_idx)}, giảm {fname(cut_idx)}."
        return f"Dashboard thiếu {metrics['shortage']:.2f}; bước tới bù {fname(add_idx)}."

    if float(np.sum(field_surplus)) > 0.01:
        cut_idx = int(np.argmax(field_surplus))
        return f"Dashboard dư {metrics['surplus']:.2f}; bước tới giảm {fname(cut_idx)}."

    if over_budget > 0.01:
        removable = np.maximum(0.0, cur_sol - demand_min)
        candidates = np.where(removable > 0.01)[0]
        cut_idx = max(candidates, key=lambda i: prices[i]) if len(candidates) else int(np.argmax(prices))
        return f"Dashboard vượt {over_budget:.2f}; bước tới giảm {fname(cut_idx)}."

    if under_target > 0.01:
        legal_receivers = np.where(capacity > 0.01)[0]
        if len(legal_receivers):
            add_idx = min(legal_receivers, key=lambda i: prices[i])
            return f"Dashboard thiếu mục tiêu {under_target:.2f}; ưu tiên {fname(add_idx)}."

    expensive = np.where((prices >= np.percentile(prices, 67)) & (cur_sol > demand_min + 0.01))[0]
    cheap = np.where((prices <= np.percentile(prices, 33)) & (capacity > 0.01))[0]
    if len(expensive) and len(cheap):
        cut_idx = max(expensive, key=lambda i: prices[i])
        add_idx = min(cheap, key=lambda i: prices[i])
        return f"Bước tiếp theo chuyển nước từ {fname(cut_idx)} sang {fname(add_idx)}."

    return "Bước tiếp theo giữ ổn định vì phương án đã hợp lệ."


def render_decision_narrative(algo, step):
    prev_sol = st.session_state.get("demo_prev_sol")
    cur_sol = st.session_state.get("demo_current")
    info = st.session_state.get("demo_step_info", {})
    text = build_decision_narrative(algo, step, info, prev_sol, cur_sol)
    st.markdown("**Mô tả quyết định của thuật toán**")
    st.markdown(f"<div class='decision-box'>{text}</div>", unsafe_allow_html=True)


def build_run_summary_text(algo_name, row):
    final_sol = np.array(row["best_solution"], dtype=float)
    first_sol = row.get("first_solution")
    if first_sol is None:
        first_sol = st.session_state.get("demo_first_sol")
    if first_sol is None:
        first_sol = final_sol.copy()
    first_sol = np.array(first_sol, dtype=float)

    start = calc_metrics(first_sol)
    end = calc_metrics(final_sol)
    water_saved = start["total"] - end["total"]
    cost_saved = start["cost"] - end["cost"]
    waste_saved = start["surplus"] - end["surplus"]
    shortage_fixed = start["shortage"] - end["shortage"]
    underuse_fixed = start["under_target"] - end["under_target"]
    avg_start = start["cost"] / max(start["total"], 1e-9)
    avg_end = end["cost"] / max(end["total"], 1e-9)
    avg_saved = avg_start - avg_end

    if water_saved > 0.01:
        water_text = f"giảm tổng lượng nước dùng {water_saved:.2f} m³"
    elif water_saved < -0.01:
        water_text = f"bơm thêm {abs(water_saved):.2f} m³ để bù thiếu và đạt mục tiêu tưới"
    else:
        water_text = "giữ tổng lượng nước gần như không đổi"

    if cost_saved > 0.01:
        cost_text = f"tiết kiệm {cost_saved:.2f} chi phí bơm"
    elif cost_saved < -0.01:
        cost_text = f"chấp nhận tăng {abs(cost_saved):.2f} chi phí bơm để xử lý thiếu nước hoặc ràng buộc"
    else:
        cost_text = "giữ chi phí bơm gần như không đổi"

    return (
        f"{algo_name} kết thúc với điểm {row['best_fitness']:.4f}. So với bước đầu, thuật toán {water_text}, "
        f"{cost_text}. Phần nước dư/vượt ngưỡng giảm {waste_saved:.2f} m³, thiếu nước giảm {shortage_fixed:.2f} m³, "
        f"và phần thiếu so với mục tiêu giảm {underuse_fixed:.2f} m³. Chi phí trung bình mỗi m³ thay đổi từ "
        f"{avg_start:.2f} xuống {avg_end:.2f}, tức {'tiết kiệm' if avg_saved >= 0 else 'tăng'} {abs(avg_saved):.2f}/m³."
    )


def render_finished_algorithms(saved):
    st.divider()
    st.subheader("Thuật toán đã chạy")
    best_algo = min(saved, key=lambda a: saved[a][0]["best_fitness"])
    for name, runs in saved.items():
        row = runs[0]
        is_best = name == best_algo
        badge = "Tốt nhất" if is_best else "Đã chạy"
        summary = build_run_summary_text(name, row)
        border = "#2E7D32" if is_best else "#B8CCE8"
        bg = "#F0FAF1" if is_best else "#F8FBFF"
        st.markdown(
            f"""
<div class='run-summary-card' style='border-color:{border}; background:{bg};'>
  <b>{name} - {badge}</b><br>
  {summary}<br>
  <span style='font-size:13px;color:#455A64;'>Thời gian chạy: {row.get('runtime', 0.0):.3f}s</span>
</div>
""",
            unsafe_allow_html=True,
        )


def algorithm_explanation(algo, step, info, prev_parts, cur_parts):
    delta_score = cur_parts["score"] - prev_parts["score"]
    direction = "giảm" if delta_score < 0 else "tăng hoặc giữ nguyên"
    common = f"Bước {step}: điểm phương án hiển thị {direction} {abs(delta_score):.3f} so với bước trước."

    if algo == "GA":
        detail = (
            "GA đang chạy đúng kiểu quần thể. Một bước demo tương ứng một thế hệ: giữ elite tốt nhất, chọn cha mẹ bằng "
            f"tournament k={cfg.GA_TOURNAMENT_K}, lai BLX-alpha để sinh nghiệm con, rồi đột biến Gaussian trên từng gen. "
            f"Thế hệ này tạo {info.get('offspring', 0)} cá thể con, lai {info.get('crossovers', 0)} lần và đột biến "
            f"{info.get('mutations', 0)} gen. Điểm tốt nhất thế hệ trước là {info.get('best_before', 0):.3f}, "
            f"sau thế hệ là {info.get('best_after', 0):.3f}."
        )
    elif algo == "SA":
        detail = (
            "SA không có quần thể. Nó giữ một nghiệm hiện tại, sinh nghiệm láng giềng bằng cách sửa ngẫu nhiên một thửa, "
            "rồi dùng luật Metropolis: tốt hơn thì nhận, xấu hơn vẫn có thể nhận theo exp(-delta/T). "
            f"Ở mức nhiệt T={info.get('temperature_before', 0):.2f}, bước này thử {info.get('trials', 0)} láng giềng: "
            f"nhận tốt hơn {info.get('accepted_better', 0)}, nhận xấu hơn {info.get('accepted_worse', 0)}, "
            f"từ chối {info.get('rejected', 0)}. Nhiệt sau bước còn {info.get('temperature_after', 0):.2f}."
        )
    elif algo == "PSO":
        detail = (
            "PSO đang chạy đúng kiểu bầy hạt. Mỗi hạt là một phương án phân bổ nước, có vị trí và vận tốc riêng. "
            "Một bước demo cập nhật vận tốc theo "
            f"v = {cfg.PSO_W:.2f}*v + {cfg.PSO_C1:.2f}*r1*(pbest - x) + {cfg.PSO_C2:.2f}*r2*(gbest - x). "
            f"Sau đó hạt di chuyển, sửa ràng buộc và cập nhật pbest/gbest. Bước này có {info.get('pbest_updates', 0)} hạt "
            f"cải thiện pbest; gbest {'được cải thiện' if info.get('gbest_improved') else 'chưa cải thiện'}; "
            f"độ lớn vận tốc trung bình là {info.get('mean_velocity', 0):.3f}."
        )
    else:
        phase = info.get("phase", "GA")
        boundary = int(info.get("phase_boundary", DEFAULT_DEMO_STEPS // 2))
        detail = (
            f"Hybrid dùng 2 pha rõ ràng: bước 1-{boundary} là GA để tìm vùng nghiệm tốt, sau đó bước "
            f"{boundary + 1}-{DEFAULT_DEMO_STEPS} là SA để tinh chỉnh từ nghiệm tốt nhất của GA. "
        )
        if phase == "GA":
            detail += (
                f"Hiện đang ở pha GA: tạo {info.get('offspring', 0)} cá thể con, lai {info.get('crossovers', 0)} lần, "
                f"đột biến {info.get('mutations', 0)} gen."
            )
        else:
            detail += (
                f"Hiện đang ở pha SA: thử {info.get('trials', 0)} láng giềng tại T={info.get('temperature_before', 0):.2f}, "
                f"nhận tốt hơn {info.get('accepted_better', 0)}, nhận xấu hơn {info.get('accepted_worse', 0)}, "
                f"từ chối {info.get('rejected', 0)}."
            )
    return common, detail


def build_algorithm_step_table(algo, info):
    if not info:
        return pd.DataFrame()
    if algo == "GA":
        rows = [
            ("Cơ chế", "Quần thể cá thể"),
            ("Kích thước quần thể", info.get("population_size")),
            ("Elite giữ lại", info.get("elite_n")),
            ("Cá thể con", info.get("offspring")),
            ("Số lần lai BLX-alpha", info.get("crossovers")),
            ("Số gen đột biến", info.get("mutations")),
            ("Fitness tốt nhất trước", round(info.get("best_before", 0.0), 3)),
            ("Fitness tốt nhất sau", round(info.get("best_after", 0.0), 3)),
            ("Fitness trung bình sau", round(info.get("avg_after", 0.0), 3)),
        ]
    elif algo == "SA":
        rows = [
            ("Cơ chế", "Một nghiệm hiện tại + nhiệt độ"),
            ("Nhiệt độ trước", round(info.get("temperature_before", 0.0), 3)),
            ("Nhiệt độ sau", round(info.get("temperature_after", 0.0), 3)),
            ("Số láng giềng đã thử", info.get("trials")),
            ("Nhận vì tốt hơn", info.get("accepted_better")),
            ("Nhận dù xấu hơn", info.get("accepted_worse")),
            ("Từ chối", info.get("rejected")),
            ("Fitness hiện tại trước", round(info.get("current_fit_before", 0.0), 3)),
            ("Fitness hiện tại sau", round(info.get("current_fit_after", 0.0), 3)),
        ]
    elif algo == "PSO":
        rows = [
            ("Cơ chế", "Bầy hạt có vị trí và vận tốc"),
            ("Số hạt", info.get("particles")),
            ("pbest được cập nhật", info.get("pbest_updates")),
            ("gbest cải thiện", "Có" if info.get("gbest_improved") else "Không"),
            ("Vận tốc trung bình", round(info.get("mean_velocity", 0.0), 3)),
            ("Fitness gbest trước", round(info.get("gbest_before", 0.0), 3)),
            ("Fitness gbest sau", round(info.get("gbest_after", 0.0), 3)),
        ]
    else:
        phase = info.get("phase", "GA")
        rows = [("Cơ chế", "GA tìm rộng, SA tinh chỉnh"), ("Pha hiện tại", phase)]
        rows.extend(build_algorithm_step_table("GA" if phase == "GA" else "SA", info).itertuples(index=False, name=None))
    return pd.DataFrame(rows, columns=["Thuộc tính", "Giá trị"])


def render_step_explanation(algo, step):
    prev_sol = st.session_state.get("demo_prev_sol")
    cur_sol = st.session_state.get("demo_current")
    if step <= 0 or prev_sol is None or cur_sol is None:
        st.info("Hãy bấm `Chạy 1 bước` hoặc `Chạy demo` để xem giải thích chi tiết cho từng bước.")
        return

    prev_sol = np.array(prev_sol, dtype=float)
    cur_sol = np.array(cur_sol, dtype=float)
    prev_metrics = calc_metrics(prev_sol)
    cur_metrics = calc_metrics(cur_sol)
    prev_parts = calc_score_parts(prev_sol)
    cur_parts = calc_score_parts(cur_sol)
    field_df = build_field_explanation(prev_sol, cur_sol)
    changed_df = field_df.reindex(field_df["Thay đổi"].abs().sort_values(ascending=False).index).head(5)

    info = st.session_state.get("demo_step_info", {})

    tab_overview, tab_formula, tab_fields = st.tabs(
        ["Cơ chế thuật toán", "Công thức điểm chung", "Chi tiết từng thửa"]
    )

    with tab_overview:
        summary, detail = algorithm_explanation(algo, step, info, prev_parts, cur_parts)
        st.markdown(f"**Bước đang xem:** {step}")
        st.write(summary)
        st.write(detail)
        table = build_algorithm_step_table(algo, info)
        if not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)

        if cur_metrics["shortage"] > 0:
            st.warning(
                f"Phương án sau bước này còn thiếu {cur_metrics['shortage']:.2f} m³. "
                "Thuật toán sẽ bị phạt mạnh vì thiếu nước ảnh hưởng trực tiếp đến yêu cầu tối thiểu của thửa ruộng."
            )
        elif cur_metrics["surplus"] > 0:
            st.warning(
                f"Phương án sau bước này còn dư {cur_metrics['surplus']:.2f} m³. "
                "Phần dư được tính từ nước vượt Max từng thửa và phần vượt tổng ngân sách nếu có."
            )
        else:
            st.success("Phương án sau bước này không thiếu và không dư so với các ngưỡng đang kiểm tra.")

        st.markdown("**Những thay đổi lớn nhất ở bước này**")
        st.dataframe(
            changed_df[["Thửa", "Cây trồng", "Trước bước", "Sau bước", "Thay đổi", "Trạng thái", "Diễn giải"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab_formula:
        st.write(
            "Công thức điểm được dùng chung cho cả GA, SA, PSO và Hybrid để so sánh công bằng trên cùng một bài toán. "
            "Khác biệt giữa các thuật toán không nằm ở công thức điểm, mà nằm ở cách mỗi thuật toán sinh nghiệm tiếp theo."
        )
        st.code(
            "Điểm = GAMMA_COST * chi_phí\n"
            "      + ALPHA_PENALTY * tổng(max(0, min_i - x_i)^2)\n"
            "      + BETA_WASTE * (tổng(max(0, x_i - max_i)) + max(0, tổng_nước - W_total))\n"
            "      + DELTA_UNDERUSE * max(0, mục_tiêu_nước - tổng_nước)^2",
            language=None,
        )
        formula_df = pd.DataFrame(
            [
                {
                    "Thành phần": "Chi phí bơm",
                    "Trước bước": round(prev_parts["cost_part"], 3),
                    "Sau bước": round(cur_parts["cost_part"], 3),
                    "Chênh lệch": round(cur_parts["cost_part"] - prev_parts["cost_part"], 3),
                    "Ý nghĩa": "Tổng nước từng thửa nhân với giá bơm.",
                },
                {
                    "Thành phần": "Phạt thiếu nước",
                    "Trước bước": round(prev_parts["shortage_penalty"], 3),
                    "Sau bước": round(cur_parts["shortage_penalty"], 3),
                    "Chênh lệch": round(cur_parts["shortage_penalty"] - prev_parts["shortage_penalty"], 3),
                    "Ý nghĩa": "Thiếu càng nhiều thì bị phạt bình phương, nên lỗi thiếu giảm sẽ kéo điểm xuống mạnh.",
                },
                {
                    "Thành phần": "Phạt dư nước",
                    "Trước bước": round(prev_parts["surplus_penalty"], 3),
                    "Sau bước": round(cur_parts["surplus_penalty"], 3),
                    "Chênh lệch": round(cur_parts["surplus_penalty"] - prev_parts["surplus_penalty"], 3),
                    "Ý nghĩa": "Dư vượt Max hoặc vượt ngân sách nước đều bị phạt để thuật toán học cách giảm lãng phí.",
                },
                {
                    "Thành phần": "Phạt chưa dùng đủ nước mục tiêu",
                    "Trước bước": round(prev_parts["underuse_penalty"], 3),
                    "Sau bước": round(cur_parts["underuse_penalty"], 3),
                    "Chênh lệch": round(cur_parts["underuse_penalty"] - prev_parts["underuse_penalty"], 3),
                    "Ý nghĩa": "Nếu tổng nước thấp hơn mục tiêu, nghiệm bị phạt để tránh tụt hết về Min.",
                },
                {
                    "Thành phần": "Tổng điểm",
                    "Trước bước": round(prev_parts["score"], 3),
                    "Sau bước": round(cur_parts["score"], 3),
                    "Chênh lệch": round(cur_parts["score"] - prev_parts["score"], 3),
                    "Ý nghĩa": "Tổng hợp ba phần trên; thấp hơn là tốt hơn.",
                },
            ]
        )
        st.dataframe(formula_df, use_container_width=True, hide_index=True)

        metric_df = pd.DataFrame(
            [
                {
                    "Chỉ số": "Tổng nước",
                    "Trước bước": round(prev_metrics["total"], 2),
                    "Sau bước": round(cur_metrics["total"], 2),
                    "Chênh lệch": round(cur_metrics["total"] - prev_metrics["total"], 2),
                },
                {
                    "Chỉ số": "Thiếu nước",
                    "Trước bước": round(prev_metrics["shortage"], 2),
                    "Sau bước": round(cur_metrics["shortage"], 2),
                    "Chênh lệch": round(cur_metrics["shortage"] - prev_metrics["shortage"], 2),
                },
                {
                    "Chỉ số": "Dư + vượt ngưỡng",
                    "Trước bước": round(prev_metrics["surplus"], 2),
                    "Sau bước": round(cur_metrics["surplus"], 2),
                    "Chênh lệch": round(cur_metrics["surplus"] - prev_metrics["surplus"], 2),
                },
                {
                    "Chỉ số": "Thiếu so với mục tiêu",
                    "Trước bước": round(prev_metrics["under_target"], 2),
                    "Sau bước": round(cur_metrics["under_target"], 2),
                    "Chênh lệch": round(cur_metrics["under_target"] - prev_metrics["under_target"], 2),
                },
                {
                    "Chỉ số": "Chi phí bơm",
                    "Trước bước": round(prev_metrics["cost"], 2),
                    "Sau bước": round(cur_metrics["cost"], 2),
                    "Chênh lệch": round(cur_metrics["cost"] - prev_metrics["cost"], 2),
                },
            ]
        )
        st.markdown("**Các chỉ số trước và sau bước này**")
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

    with tab_fields:
        st.write(
            "Bảng này đọc theo từng thửa ruộng. Nếu `Sau bước` nhỏ hơn Min thì thửa đó thiếu nước. "
            "Nếu lớn hơn Max thì thửa đó dư nước. Nếu nằm giữa Min và Max thì thửa đó hợp lệ."
        )
        st.dataframe(field_df, use_container_width=True, hide_index=True, height=360)


def ensure():
    defaults = {
        "selected_demo_algo": "GA",
        "prepared_algo": "GA",
        "demo_steps": DEFAULT_DEMO_STEPS,
        "demo_steps_version": 3,
        "demo_running": False,
        "demo_step": 0,
        "demo_internal": {},
        "demo_current": None,
        "demo_best": None,
        "demo_best_sol": None,
        "demo_history": [],
        "demo_log": [],
        "demo_runtime": 0.0,
        "demo_first_sol": None,
        "demo_prev_sol": None,
        "demo_step_info": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    demo_version_changed = (
        st.session_state.get("demo_steps_version") != 3
        or st.session_state.get("demo_steps") != DEFAULT_DEMO_STEPS
    )
    if demo_version_changed:
        st.session_state["demo_steps"] = DEFAULT_DEMO_STEPS
        st.session_state["demo_steps_version"] = 3
        reset_algo(st.session_state.get("prepared_algo", "GA"))


def sync_demo_step_limit():
    max_steps = int(st.session_state["demo_steps"])
    if st.session_state["demo_step"] <= max_steps:
        return

    reset_algo(st.session_state["prepared_algo"])
    st.info("Số bước mô phỏng đã giảm xuống thấp hơn bước hiện tại, nên demo được khởi động lại để dữ liệu không bị lẫn.")


def reset_algo(algo):
    st.session_state["prepared_algo"] = algo
    st.session_state["selected_demo_algo"] = algo
    st.session_state["demo_running"] = False
    st.session_state["demo_step"] = 0
    st.session_state["demo_current"] = rough_demo_sol()
    st.session_state["demo_best"] = None
    st.session_state["demo_best_sol"] = None
    st.session_state["demo_history"] = []
    st.session_state["demo_log"] = []
    st.session_state["demo_runtime"] = 0.0
    st.session_state["demo_first_sol"] = None
    st.session_state["demo_prev_sol"] = None
    st.session_state["demo_step_info"] = {}

    state = {"current": st.session_state["demo_current"].copy()}
    if algo in ("GA", "Hybrid"):
        pop = [rough_demo_sol() for _ in range(cfg.GA_POP_SIZE)]
        state["pop"] = pop
        state["pop_fits"] = [fit(x) for x in pop]
        state["phase_boundary"] = DEFAULT_DEMO_STEPS // 2
        if algo == "Hybrid":
            best_idx = int(np.argmin(state["pop_fits"]))
            state["hybrid_sa_current"] = pop[best_idx].copy()
            state["hybrid_sa_fit"] = float(state["pop_fits"][best_idx])
            state["hybrid_sa_T"] = float(cfg.HYBRID_SA_T_MAX)
    elif algo == "SA":
        state["T"] = float(cfg.SA_T_MAX)
    else:
        n = cfg.PSO_N_PARTICLES
        pos = np.array([rough_demo_sol() for _ in range(n)])
        vel = np.random.uniform(-cfg.PSO_V_MAX, cfg.PSO_V_MAX, (n, n_fields))
        pbest = pos.copy()
        pbest_fit = np.array([fit(x) for x in pbest])
        gbest = pbest[int(np.argmin(pbest_fit))].copy()
        state.update({"pos": pos, "vel": vel, "pbest": pbest, "pbest_fit": pbest_fit, "gbest": gbest})
    st.session_state["demo_internal"] = state


def step_once(algo):
    state = st.session_state["demo_internal"]
    cur = state["current"].copy()
    step = int(st.session_state["demo_step"])
    info = {"algo": algo, "step": step}

    if algo == "GA" or (algo == "Hybrid" and step <= state.get("phase_boundary", DEFAULT_DEMO_STEPS // 2)):
        pop = state["pop"]
        fits = state["pop_fits"]
        best_before = float(min(fits))
        avg_before = float(np.mean(fits))
        crossovers = 0
        mutations = 0

        def tournament():
            idx = np.random.choice(len(pop), cfg.GA_TOURNAMENT_K, replace=False)
            return pop[int(idx[np.argmin([fits[i] for i in idx])])].copy()

        elite_n = max(1, int(cfg.GA_ELITE_RATIO * cfg.GA_POP_SIZE))
        order = np.argsort(fits)
        new_pop = [demo_candidate(pop[int(i)], step) for i in order[:elite_n]]
        while len(new_pop) < cfg.GA_POP_SIZE:
            p1, p2 = tournament(), tournament()
            if np.random.rand() < cfg.GA_CROSSOVER_RATE:
                d = np.abs(p1 - p2)
                child = np.random.uniform(np.minimum(p1, p2) - cfg.GA_BLX_ALPHA * d, np.maximum(p1, p2) + cfg.GA_BLX_ALPHA * d)
                crossovers += 1
            else:
                child = p1.copy()
            for i in range(n_fields):
                if np.random.rand() < cfg.GA_MUTATION_RATE:
                    child[i] += np.random.normal(0, cfg.GA_MUTATION_SIGMA)
                    mutations += 1
            child = demo_candidate(child, step)
            new_pop.append(child)
        pop = new_pop
        fits = [fit(x) for x in pop]
        cur = pop[int(np.argmin(fits))].copy()
        cur_fit = float(min(fits))
        state["pop"], state["pop_fits"] = pop, fits
        if algo == "Hybrid":
            state["hybrid_sa_current"] = cur.copy()
            state["hybrid_sa_fit"] = cur_fit
        info.update(
            {
                "phase": "GA",
                "phase_boundary": state.get("phase_boundary", DEFAULT_DEMO_STEPS // 2),
                "population_size": cfg.GA_POP_SIZE,
                "elite_n": elite_n,
                "offspring": cfg.GA_POP_SIZE - elite_n,
                "crossovers": crossovers,
                "mutations": mutations,
                "best_before": best_before,
                "best_after": cur_fit,
                "avg_before": avg_before,
                "avg_after": float(np.mean(fits)),
            }
        )

    elif algo == "SA":
        T = state.get("T", cfg.SA_T_MAX)
        current_fit_before = float(fit(cur))
        accepted_better = 0
        accepted_worse = 0
        rejected = 0
        for _ in range(cfg.SA_ITER_PER_T):
            nb = cur.copy()
            nb[np.random.randint(n_fields)] += np.random.normal(0, cfg.SA_NEIGHBOR_SIGMA)
            nb = demo_candidate(nb, step)
            delta = fit(nb) - fit(cur)
            if delta < 0:
                cur = nb
                accepted_better += 1
            elif T > 0 and np.random.rand() < math.exp(-delta / max(T, 1e-9)):
                cur = nb
                accepted_worse += 1
            else:
                rejected += 1
        state["T"] = max(cfg.SA_T_MIN, T * cfg.SA_ALPHA)
        cur_fit = float(fit(cur))
        info.update(
            {
                "phase": "SA",
                "trials": cfg.SA_ITER_PER_T,
                "temperature_before": float(T),
                "temperature_after": float(state["T"]),
                "accepted_better": accepted_better,
                "accepted_worse": accepted_worse,
                "rejected": rejected,
                "current_fit_before": current_fit_before,
                "current_fit_after": cur_fit,
            }
        )

    elif algo == "PSO":
        pos, vel = state["pos"], state["vel"]
        pbest, pbest_fit, gbest = state["pbest"], state["pbest_fit"], state["gbest"]
        n = len(pos)
        if repair_ratio(step) >= 1.0:
            pbest = np.array([final_repair(p) for p in pbest])
            pbest_fit = np.array([fit(p) for p in pbest])
            gbest = pbest[int(np.argmin(pbest_fit))].copy()
        gbest_before = float(np.min(pbest_fit))
        pbest_updates = 0
        r1, r2 = np.random.rand(n, n_fields), np.random.rand(n, n_fields)
        vel = np.clip(cfg.PSO_W * vel + cfg.PSO_C1 * r1 * (pbest - pos) + cfg.PSO_C2 * r2 * (gbest - pos), -cfg.PSO_V_MAX, cfg.PSO_V_MAX)
        pos = pos + vel
        for i in range(n):
            pos[i] = demo_candidate(pos[i], step)
            fval = fit(pos[i])
            if fval < pbest_fit[i]:
                pbest[i], pbest_fit[i] = pos[i].copy(), fval
                pbest_updates += 1
        gbest = pbest[int(np.argmin(pbest_fit))].copy()
        cur, cur_fit = gbest.copy(), float(np.min(pbest_fit))
        state.update({"pos": pos, "vel": vel, "pbest": pbest, "pbest_fit": pbest_fit, "gbest": gbest})
        info.update(
            {
                "phase": "PSO",
                "particles": n,
                "pbest_updates": pbest_updates,
                "gbest_before": gbest_before,
                "gbest_after": cur_fit,
                "gbest_improved": cur_fit < gbest_before,
                "mean_velocity": float(np.mean(np.linalg.norm(vel, axis=1))),
            }
        )

    else:
        T = state.get("hybrid_sa_T", cfg.HYBRID_SA_T_MAX)
        cur = state.get("hybrid_sa_current", cur).copy()
        current_fit_before = float(state.get("hybrid_sa_fit", fit(cur)))
        accepted_better = 0
        accepted_worse = 0
        rejected = 0
        for _ in range(cfg.SA_ITER_PER_T):
            nb = cur.copy()
            nb[np.random.randint(n_fields)] += np.random.normal(0, cfg.SA_NEIGHBOR_SIGMA)
            nb = demo_candidate(nb, step)
            delta = fit(nb) - fit(cur)
            if delta < 0:
                cur = nb
                accepted_better += 1
            elif T > 0 and np.random.rand() < math.exp(-delta / max(T, 1e-9)):
                cur = nb
                accepted_worse += 1
            else:
                rejected += 1
        state["hybrid_sa_T"] = max(cfg.SA_T_MIN, T * cfg.SA_ALPHA)
        cur_fit = float(fit(cur))
        state["hybrid_sa_current"] = cur.copy()
        state["hybrid_sa_fit"] = cur_fit
        info.update(
            {
                "phase": "SA",
                "phase_boundary": state.get("phase_boundary", DEFAULT_DEMO_STEPS // 2),
                "trials": cfg.SA_ITER_PER_T,
                "temperature_before": float(T),
                "temperature_after": float(state["hybrid_sa_T"]),
                "accepted_better": accepted_better,
                "accepted_worse": accepted_worse,
                "rejected": rejected,
                "current_fit_before": current_fit_before,
                "current_fit_after": cur_fit,
            }
        )

    state["current"] = cur
    st.session_state["demo_step_info"] = info
    st.session_state["demo_internal"] = state
    return cur, cur_fit


def popup(algo, step, state):
    if algo == "GA":
        return f"Thế hệ {step}: Chọn lọc tự nhiên -> BLX crossover -> Đột biến -> Elite giữ lại."
    if algo == "SA":
        return f"Nhiệt độ T={state.get('T', cfg.SA_T_MAX):.2f}: chấp nhận nghiệm xấu với xác suất e^(-Δ/T)."
    if algo == "PSO":
        return f"Vòng {step}: cập nhật vận tốc theo pbest + gbest rồi di chuyển hạt."
    phase = "GA" if step <= state.get("phase_boundary", 0) else "SA"
    if phase == "GA":
        return f"Hybrid pha GA ({step}/{state.get('phase_boundary', 0)}): tìm rộng bằng quần thể."
    return f"Hybrid pha SA ({step}/{DEFAULT_DEMO_STEPS}): tinh chỉnh từ nghiệm tốt nhất của GA."


def render_grid(sol):
    cols = st.columns(5)
    for i in range(n_fields):
        v, mn, mx = float(sol[i]), float(demand_min[i]), float(demand_max[i])
        css, label = ("ok", "Đủ")
        if v < mn:
            css, label = ("bad", "Thiếu")
        elif v > mx:
            css, label = ("warn", "Dư")
        with cols[i % 5]:
            st.markdown(
                f"<div class='field-card {css}'><b>{field_names[i]}</b><br>{crop_names[i]}<br><small>Min {mn:.0f} | Max {mx:.0f}</small><br><b>{v:.2f}</b><br><small>{label}</small></div>",
                unsafe_allow_html=True,
            )


def apply_one_step(algo_name):
    t0 = time.perf_counter()
    prev_sol = st.session_state["demo_current"].copy()
    st.session_state["demo_step"] += 1
    step = st.session_state["demo_step"]
    sol, cur_fit = step_once(algo_name)
    step_info = st.session_state.get("demo_step_info", {})
    if st.session_state["demo_first_sol"] is None:
        st.session_state["demo_first_sol"] = sol.copy()
    if st.session_state["demo_best"] is None or cur_fit < st.session_state["demo_best"]:
        st.session_state["demo_best"] = cur_fit
        st.session_state["demo_best_sol"] = sol.copy()
    st.session_state["demo_current"] = sol.copy()
    st.session_state["demo_history"].append(float(st.session_state["demo_best"]))
    metrics = calc_metrics(sol)
    shortage = metrics["shortage"]
    waste = metrics["surplus"]
    st.session_state["demo_log"].append(
        {
            "Bước": step,
            "Điểm": round(cur_fit, 3),
            "Tốt nhất": round(float(st.session_state["demo_best"]), 3),
            "Thiếu": round(shortage, 2),
            "Dư": round(waste, 2),
            "Tổng": round(metrics["total"], 2),
            "Cơ chế": step_info.get("phase", algo_name),
        }
    )
    st.session_state["demo_prev_sol"] = prev_sol.copy()
    st.session_state["demo_runtime"] += time.perf_counter() - t0


def persist_finished_result(algo_name):
    if st.session_state.get("demo_best_sol") is None:
        return False
    best_sol = final_repair(st.session_state["demo_best_sol"])
    st.session_state["demo_best_sol"] = best_sol.copy()
    st.session_state["demo_best"] = float(fit(best_sol))
    saved = st.session_state.get("run_results", {}) or {}
    saved[algo_name] = [
        {
            "algo_name": algo_name,
            "run_id": 0,
            "best_fitness": float(st.session_state["demo_best"]),
            "best_solution": best_sol.tolist(),
            "first_solution": (
                st.session_state["demo_first_sol"].tolist()
                if st.session_state.get("demo_first_sol") is not None
                else best_sol.tolist()
            ),
            "history": st.session_state["demo_history"],
            "runtime": float(st.session_state["demo_runtime"]),
        }
    ]
    session_state.save_results(saved)
    return True


ensure()
if st.session_state["demo_current"] is None:
    reset_algo(st.session_state["prepared_algo"])

with st.sidebar:
    algo_choices = st.session_state.get("selected_algos") or ["GA", "SA", "PSO", "Hybrid"]
    if st.session_state["selected_demo_algo"] not in algo_choices:
        st.session_state["selected_demo_algo"] = algo_choices[0]
        reset_algo(algo_choices[0])
    choice = st.selectbox(
        "Chọn thuật toán",
        algo_choices,
        index=algo_choices.index(st.session_state["selected_demo_algo"]),
        format_func=lambda a: ALGO_LABELS.get(a, a),
    )
    if choice != st.session_state["prepared_algo"]:
        reset_algo(choice)
        st.rerun()
    st.caption("Tự đổi ngay khi chọn thuật toán.")
    st.session_state["demo_steps"] = DEFAULT_DEMO_STEPS
    st.caption(f"{DEFAULT_DEMO_STEPS} bước • {AUTO_DELAY_SECONDS:.0f}s/bước")
    sync_demo_step_limit()
    one_step = st.button("Chạy 1 bước", use_container_width=True)
    run_lbl = "Dừng" if st.session_state["demo_running"] else "Chạy demo"
    toggle_run = st.button(run_lbl, type="primary", use_container_width=True)
    if toggle_run:
        st.session_state["demo_running"] = not st.session_state["demo_running"]
        st.rerun()
    if st.button("Xem kết quả", use_container_width=True):
        if persist_finished_result(st.session_state["prepared_algo"]):
            st.switch_page("pages/03_result.py")
        else:
            st.warning("Bạn hãy chạy ít nhất 1 bước trước khi xem kết quả.")

algo = st.session_state["prepared_algo"]
st.title(f"Demo từng thuật toán: {ALGO_LABELS.get(algo, algo)}")

if one_step and st.session_state["demo_step"] < st.session_state["demo_steps"]:
    apply_one_step(algo)
    if st.session_state["demo_step"] >= st.session_state["demo_steps"]:
        persist_finished_result(algo)

if st.session_state["demo_running"] and st.session_state["demo_step"] < st.session_state["demo_steps"]:
    apply_one_step(algo)
    if st.session_state["demo_step"] >= st.session_state["demo_steps"]:
        st.session_state["demo_running"] = False
        persist_finished_result(algo)

if st.session_state["demo_step"] >= st.session_state["demo_steps"]:
    st.session_state["demo_running"] = False
    st.success("Đã chạy đủ số bước. Bạn có thể đổi thuật toán và chạy lại.")

display_step = min(st.session_state["demo_step"], st.session_state["demo_steps"])
progress_value = display_step / max(st.session_state["demo_steps"], 1)
best_label = "Chưa chạy" if st.session_state["demo_best"] is None else f"{st.session_state['demo_best']:.4f}"
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Bước", f"{display_step}/{st.session_state['demo_steps']}")
c2.metric("Điểm tối ưu tốt nhất", best_label)
c3.markdown(f"<div class='popup'>{popup(algo, display_step, st.session_state['demo_internal'])}</div>", unsafe_allow_html=True)
st.progress(progress_value)
if st.session_state["demo_running"]:
    st.info("Đang auto chạy từng bước... có thể bấm `Dừng` bất kỳ lúc nào.")

render_decision_narrative(algo, display_step)

current_metrics = calc_metrics(st.session_state["demo_current"])
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lượng nước đang sử dụng", f"{current_metrics['total']:.2f} m³")
m2.metric("Mục tiêu nước cần phân bổ", f"{current_metrics['target']:.2f} m³")
m3.metric("Thiếu nước", f"{current_metrics['shortage']:.2f} m³")
m4.metric("Dư/vượt ngưỡng", f"{current_metrics['surplus']:.2f} m³")

st.subheader("Lưới 10 thửa ruộng")
render_grid(st.session_state["demo_current"])

left, right = st.columns([1.4, 1])
with left:
    st.subheader("Hội tụ thời gian thực")
    fig, ax = plt.subplots(figsize=(6, 3))
    hist = st.session_state["demo_history"]
    if hist:
        ax.plot(hist, color="#1976D2", linewidth=2)
        ax.fill_between(range(len(hist)), hist, alpha=0.2, color="#90CAF9")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlabel("Bước")
    ax.set_ylabel("Điểm tối ưu tốt nhất")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.subheader("Nhật ký vòng lặp")
    if st.session_state["demo_log"]:
        log_df = pd.DataFrame(st.session_state["demo_log"]).rename(
            columns={"Fitness": "Điểm", "Best": "Tốt nhất"}
        )
        st.dataframe(
            log_df,
            use_container_width=True,
            hide_index=True,
            height=360,
        )

st.divider()
render_cost_priority_dashboard()

st.divider()
st.subheader("Giải thích bước hiện tại")
render_step_explanation(algo, display_step)

first_sol = st.session_state.get("demo_first_sol")
current_sol = st.session_state.get("demo_current")
if first_sol is not None and current_sol is not None:
    m_start = calc_metrics(first_sol)
    m_end = calc_metrics(current_sol)
    st.divider()
    st.subheader("So sánh bước 1 và hiện tại")
    compare_df = pd.DataFrame(
        [
            {
                "Chỉ số": "Tổng nước (m³)",
                "Bước 1": round(m_start["total"], 2),
                "Hiện tại": round(m_end["total"], 2),
                "Chênh lệch": round(m_end["total"] - m_start["total"], 2),
            },
            {
                "Chỉ số": "Chi phí bơm",
                "Bước 1": round(m_start["cost"], 2),
                "Hiện tại": round(m_end["cost"], 2),
                "Chênh lệch": round(m_end["cost"] - m_start["cost"], 2),
            },
            {
                "Chỉ số": "Thiếu nước",
                "Bước 1": round(m_start["shortage"], 2),
                "Hiện tại": round(m_end["shortage"], 2),
                "Chênh lệch": round(m_end["shortage"] - m_start["shortage"], 2),
            },
            {
                "Chỉ số": "Thiếu so với mục tiêu",
                "Bước 1": round(m_start["under_target"], 2),
                "Hiện tại": round(m_end["under_target"], 2),
                "Chênh lệch": round(m_end["under_target"] - m_start["under_target"], 2),
            },
            {
                "Chỉ số": "Dư + Vượt ngưỡng",
                "Bước 1": round(m_start["excess_field"] + m_start["over_budget"], 2),
                "Hiện tại": round(m_end["excess_field"] + m_end["over_budget"], 2),
                "Chênh lệch": round(
                    (m_end["excess_field"] + m_end["over_budget"])
                    - (m_start["excess_field"] + m_start["over_budget"]),
                    2,
                ),
            },
        ]
    )
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

saved = st.session_state.get("run_results", {}) or {}
if saved:
    render_finished_algorithms(saved)
    render_run_ai_section(saved, st.session_state.get("prepared_algo"))

st.divider()
if st.button("Chuyển qua trang kết quả", type="primary", use_container_width=True, key="bottom_result_page"):
    if persist_finished_result(st.session_state["prepared_algo"]):
        st.switch_page("pages/03_result.py")
    else:
        st.warning("Bạn hãy chạy ít nhất 1 bước trước khi xem kết quả.")

if st.session_state["demo_running"] and st.session_state["demo_step"] < st.session_state["demo_steps"]:
    time.sleep(AUTO_DELAY_SECONDS)
    st.rerun()
