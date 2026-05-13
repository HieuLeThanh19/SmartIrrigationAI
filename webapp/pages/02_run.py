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
from webapp.components.metric_card import show_algo_card
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
UPPER_FACTOR = 1.8


def fit(x):
    return calculate_fitness(x, demand_min, demand_max, prices, W_total)


def rand_sol():
    # Khởi tạo có chủ đích để demo nhìn thấy thiếu/thừa trước khi được tối ưu.
    low = np.maximum(0.0, demand_min - 0.8 * (demand_max - demand_min))
    high = demand_max + 0.8 * (demand_max - demand_min)
    x = np.random.uniform(low, high)
    return np.clip(x, 0.0, demand_max * UPPER_FACTOR)


def clip_demo(x):
    return np.clip(x, 0.0, demand_max * UPPER_FACTOR)


def final_repair(x):
    """Đảm bảo nghiệm cuối cùng đủ điều kiện trình bày kết quả tối ưu hợp lệ."""
    return repair_solution(clip_to_bounds(x, demand_min, demand_max), demand_min, demand_max, W_total)


def calc_metrics(sol):
    total = float(np.sum(sol))
    excess_field = float(np.sum(np.maximum(0.0, sol - demand_max)))
    over_budget = float(max(0.0, total - W_total))
    return {
        "total": total,
        "cost": float(np.dot(prices, sol)),
        "shortage": float(np.sum(np.maximum(0.0, demand_min - sol))),
        "excess_field": excess_field,
        "over_budget": over_budget,
        "surplus": excess_field + over_budget,
    }


def calc_score_parts(sol):
    bd = fitness_breakdown(sol, demand_min, demand_max, prices, W_total)
    return {
        "cost": bd["cost_raw"],
        "shortage": bd["shortage"],
        "excess_field": bd["waste"],
        "over_budget": bd["over_budget"],
        "cost_part": bd["cost"],
        "shortage_penalty": bd["shortage_penalty"],
        "surplus_penalty": bd["waste_penalty"],
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


def algorithm_explanation(algo, step, state, prev_parts, cur_parts):
    delta_score = cur_parts["score"] - prev_parts["score"]
    direction = "tốt hơn" if delta_score < 0 else "chưa tốt hơn"
    common = (
        f"Bước {step} tạo một phương án phân bổ mới rồi tính lại điểm tối ưu. "
        f"Điểm thay đổi {delta_score:+.3f}, tức là phương án hiện tại {direction} so với ngay trước bước này. "
        "Trong bài toán này điểm càng thấp càng tốt, vì điểm gồm chi phí bơm cộng với phạt thiếu nước và phạt dư nước."
    )

    if algo == "GA":
        detail = (
            "GA xem mỗi phương án phân bổ nước như một cá thể. Ở bước này, chương trình chọn các cá thể có điểm thấp hơn "
            f"bằng tournament selection với k={cfg.GA_TOURNAMENT_K}, lai BLX-alpha để sinh con mới, sau đó đột biến nhẹ từng thửa "
            f"với xác suất {cfg.GA_MUTATION_RATE:.2f}. Công thức lai dùng khoảng giữa hai cha mẹ: "
            "con_i được lấy ngẫu nhiên quanh [min(cha_i, mẹ_i), max(cha_i, mẹ_i)] có nới thêm alpha. "
            "Nếu phương án con làm giảm thiếu/dư hoặc giảm chi phí thì điểm giảm và có cơ hội trở thành tốt nhất."
        )
    elif algo == "SA":
        t = float(state.get("T", cfg.SA_T_MAX))
        detail = (
            "SA sửa từng bước bằng cách chọn ngẫu nhiên một thửa rồi cộng/trừ một lượng nhỏ. Nếu nghiệm mới tốt hơn thì nhận ngay. "
            "Nếu nghiệm mới xấu hơn, thuật toán vẫn có thể nhận với xác suất exp(-mức xấu đi / T) để tránh kẹt ở phương án cục bộ. "
            f"Nhiệt độ hiện tại sau bước này khoảng T={t:.2f}; T càng thấp thì thuật toán càng ít chấp nhận phương án xấu."
        )
    elif algo == "PSO":
        detail = (
            "PSO xem mỗi phương án là một hạt đang bay trong không gian phân bổ nước. Ở bước này, mỗi hạt cập nhật vận tốc theo "
            f"v = {cfg.PSO_W:.2f}*v + {cfg.PSO_C1:.2f}*r1*(pbest - x) + {cfg.PSO_C2:.2f}*r2*(gbest - x). "
            "pbest là phương án tốt nhất từng hạt từng gặp, còn gbest là phương án tốt nhất toàn đàn. "
            "Vì vậy nếu một thửa đang dư, các hạt có xu hướng bị kéo về những phương án trước đó có ít dư hơn và điểm thấp hơn."
        )
    else:
        boundary = int(state.get("phase_boundary", 0))
        phase = "GA" if step <= boundary else "SA"
        detail = (
            f"Hybrid đang ở giai đoạn {phase}. Trước ranh giới bước {boundary}, thuật toán dùng GA để tìm vùng nghiệm tốt trên diện rộng. "
            "Sau đó chuyển sang SA để tinh chỉnh từng thửa nhỏ hơn. Cách này giúp vừa khám phá nhiều phương án, vừa sửa chi tiết các lỗi "
            "như thiếu nước, dư nước hoặc chi phí bơm còn cao."
        )
    return common, detail


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

    tab_overview, tab_formula, tab_fields = st.tabs(
        ["Diễn giải dễ hiểu", "Công thức tính điểm", "Chi tiết từng thửa"]
    )

    with tab_overview:
        summary, detail = algorithm_explanation(algo, step, st.session_state["demo_internal"], prev_parts, cur_parts)
        st.markdown(f"**Bước đang xem:** {step}")
        st.write(summary)
        st.write(detail)

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
            "Điểm tối ưu được tính theo nguyên tắc: điểm càng thấp thì phương án càng tốt. "
            "Chi phí bơm làm điểm tăng nhẹ, còn thiếu nước và dư nước làm điểm tăng mạnh hơn vì đó là lỗi cần tránh."
        )
        st.code(
            "Điểm = GAMMA_COST * chi_phí\n"
            "      + ALPHA_PENALTY * tổng(max(0, min_i - x_i)^2)\n"
            "      + BETA_WASTE * (tổng(max(0, x_i - max_i)) + max(0, tổng_nước - W_total))",
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
        "demo_steps": 60,
        "demo_delay_ms": 220,
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sync_demo_step_limit():
    max_steps = int(st.session_state["demo_steps"])
    if st.session_state["demo_step"] <= max_steps:
        return

    reset_algo(st.session_state["prepared_algo"])
    st.info("Số bước mô phỏng đã giảm xuống thấp hơn bước hiện tại, nên demo được khởi động lại để dữ liệu không bị lẫn.")


def reset_algo(algo):
    st.session_state["prepared_algo"] = algo
    st.session_state["demo_running"] = False
    st.session_state["demo_step"] = 0
    st.session_state["demo_current"] = rand_sol()
    st.session_state["demo_best"] = None
    st.session_state["demo_best_sol"] = None
    st.session_state["demo_history"] = []
    st.session_state["demo_log"] = []
    st.session_state["demo_runtime"] = 0.0
    st.session_state["demo_first_sol"] = None
    st.session_state["demo_prev_sol"] = None

    state = {"current": st.session_state["demo_current"].copy()}
    if algo in ("GA", "Hybrid"):
        pop = [rand_sol() for _ in range(cfg.GA_POP_SIZE)]
        state["pop"] = pop
        state["pop_fits"] = [fit(x) for x in pop]
        state["phase_boundary"] = max(1, st.session_state["demo_steps"] // 2)
    elif algo == "SA":
        state["T"] = float(cfg.SA_T_MAX)
    else:
        n = cfg.PSO_N_PARTICLES
        pos = np.array([rand_sol() for _ in range(n)])
        vel = np.random.uniform(-cfg.PSO_V_MAX, cfg.PSO_V_MAX, (n, n_fields))
        pbest = pos.copy()
        pbest_fit = np.array([fit(x) for x in pbest])
        gbest = pbest[int(np.argmin(pbest_fit))].copy()
        state.update({"pos": pos, "vel": vel, "pbest": pbest, "pbest_fit": pbest_fit, "gbest": gbest})
    st.session_state["demo_internal"] = state


def step_once(algo):
    state = st.session_state["demo_internal"]
    cur = state["current"].copy()

    if algo in ("GA", "Hybrid"):
        pop = state["pop"]
        fits = state["pop_fits"]

        def tournament():
            idx = np.random.choice(len(pop), cfg.GA_TOURNAMENT_K, replace=False)
            return pop[int(idx[np.argmin([fits[i] for i in idx])])].copy()

        elite_n = max(1, int(cfg.GA_ELITE_RATIO * cfg.GA_POP_SIZE))
        order = np.argsort(fits)
        new_pop = [pop[int(i)].copy() for i in order[:elite_n]]
        while len(new_pop) < cfg.GA_POP_SIZE:
            p1, p2 = tournament(), tournament()
            if np.random.rand() < cfg.GA_CROSSOVER_RATE:
                d = np.abs(p1 - p2)
                child = np.random.uniform(np.minimum(p1, p2) - cfg.GA_BLX_ALPHA * d, np.maximum(p1, p2) + cfg.GA_BLX_ALPHA * d)
            else:
                child = p1.copy()
            for i in range(n_fields):
                if np.random.rand() < cfg.GA_MUTATION_RATE:
                    child[i] += np.random.normal(0, cfg.GA_MUTATION_SIGMA)
            child = clip_demo(child)
            new_pop.append(child)
        pop = new_pop
        fits = [fit(x) for x in pop]
        cur = pop[int(np.argmin(fits))].copy()
        cur_fit = float(min(fits))
        if algo == "Hybrid" and st.session_state["demo_step"] > state["phase_boundary"]:
            nb = cur.copy()
            idx = np.random.randint(n_fields)
            nb[idx] += np.random.normal(0, cfg.SA_NEIGHBOR_SIGMA)
            nb = clip_demo(nb)
            nb_fit = fit(nb)
            if nb_fit < cur_fit:
                cur, cur_fit = nb, float(nb_fit)
        state["pop"], state["pop_fits"] = pop, fits

    elif algo == "SA":
        T = state.get("T", cfg.SA_T_MAX)
        for _ in range(cfg.SA_ITER_PER_T):
            nb = cur.copy()
            nb[np.random.randint(n_fields)] += np.random.normal(0, cfg.SA_NEIGHBOR_SIGMA)
            nb = clip_demo(nb)
            delta = fit(nb) - fit(cur)
            if delta < 0 or (T > 0 and np.random.rand() < math.exp(-delta / max(T, 1e-9))):
                cur = nb
        state["T"] = max(cfg.SA_T_MIN, T * cfg.SA_ALPHA)
        cur_fit = float(fit(cur))

    else:
        pos, vel = state["pos"], state["vel"]
        pbest, pbest_fit, gbest = state["pbest"], state["pbest_fit"], state["gbest"]
        n = len(pos)
        r1, r2 = np.random.rand(n, n_fields), np.random.rand(n, n_fields)
        vel = np.clip(cfg.PSO_W * vel + cfg.PSO_C1 * r1 * (pbest - pos) + cfg.PSO_C2 * r2 * (gbest - pos), -cfg.PSO_V_MAX, cfg.PSO_V_MAX)
        pos = pos + vel
        for i in range(n):
            pos[i] = clip_demo(pos[i])
            fval = fit(pos[i])
            if fval < pbest_fit[i]:
                pbest[i], pbest_fit[i] = pos[i].copy(), fval
        gbest = pbest[int(np.argmin(pbest_fit))].copy()
        cur, cur_fit = gbest.copy(), float(np.min(pbest_fit))
        state.update({"pos": pos, "vel": vel, "pbest": pbest, "pbest_fit": pbest_fit, "gbest": gbest})

    state["current"] = cur
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
    return f"Hybrid giai đoạn {phase}: GA tìm vùng tốt, SA tinh chỉnh cục bộ."


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
        }
    )
    st.session_state["demo_prev_sol"] = prev_sol.copy()
    st.session_state["demo_runtime"] += time.perf_counter() - t0


def persist_finished_result(algo_name):
    if st.session_state.get("demo_best_sol") is None:
        return False
    best_sol = final_repair(st.session_state["demo_best_sol"])
    st.session_state["demo_best_sol"] = best_sol.copy()
    st.session_state["demo_current"] = best_sol.copy()
    st.session_state["demo_best"] = float(fit(best_sol))
    saved = st.session_state.get("run_results", {}) or {}
    saved[algo_name] = [
        {
            "algo_name": algo_name,
            "run_id": 0,
            "best_fitness": float(st.session_state["demo_best"]),
            "best_solution": best_sol.tolist(),
            "history": st.session_state["demo_history"],
            "runtime": float(st.session_state["demo_runtime"]),
        }
    ]
    session_state.save_results(saved)
    return True


ensure()
if st.session_state["demo_current"] is None:
    reset_algo(st.session_state["prepared_algo"])

st.title("Demo thuật toán")
with st.sidebar:
    choice = st.selectbox("Chọn thuật toán", ["GA", "SA", "PSO", "Hybrid"], index=["GA", "SA", "PSO", "Hybrid"].index(st.session_state["selected_demo_algo"]))
    st.session_state["selected_demo_algo"] = choice
    st.caption("Auto demo chạy cố định: **3 giây / bước**")
    st.session_state["demo_steps"] = st.slider("Số bước mô phỏng", 20, 200, int(st.session_state["demo_steps"]), step=10)
    sync_demo_step_limit()
    st.divider()
    if st.button("Đổi thuật toán", use_container_width=True):
        reset_algo(choice)
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

c1, c2, c3 = st.columns([1, 1, 2])
display_step = min(st.session_state["demo_step"], st.session_state["demo_steps"])
progress_value = display_step / max(st.session_state["demo_steps"], 1)
c1.metric("Bước", f"{display_step}/{st.session_state['demo_steps']}")
c2.metric("Điểm tối ưu tốt nhất", f"{(st.session_state['demo_best'] or 0.0):.4f}")
c3.markdown(f"<div class='popup'>{popup(algo, display_step, st.session_state['demo_internal'])}</div>", unsafe_allow_html=True)
st.progress(progress_value)
if st.session_state["demo_running"]:
    st.info("Đang auto chạy từng bước... có thể bấm `Dừng` bất kỳ lúc nào.")

current_metrics = calc_metrics(st.session_state["demo_current"])
m1, m2 = st.columns(2)
m1.metric("Lượng nước đang sử dụng", f"{current_metrics['total']:.2f} m³")
m2.metric("Tổng ngân sách nước (W_total)", f"{W_total:.2f} m³")

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
    st.divider()
    st.subheader("Thuật toán đã chạy")
    best_algo = min(saved, key=lambda a: saved[a][0]["best_fitness"])
    cols = st.columns(len(saved))
    for col, (name, runs) in zip(cols, saved.items()):
        with col:
            row = runs[0]
            show_algo_card(name, row["best_fitness"], row["best_fitness"], 0.0, row.get("runtime", 0.0), name == best_algo)

# Auto mode phải giống 100% nút "Chạy 1 bước":
# mỗi lần rerun chỉ gọi apply_one_step đúng 1 lần rồi chờ 3s mới rerun tiếp.
if st.session_state["demo_running"] and st.session_state["demo_step"] < st.session_state["demo_steps"]:
    time.sleep(AUTO_DELAY_SECONDS)
    st.rerun()
