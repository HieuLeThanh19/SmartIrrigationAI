"""Trang demo chạy đồng thời 4 thuật toán."""
import html
import math
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core import config as cfg
from core.constraints import clip_to_bounds, repair_solution
from core.fitness import fitness as calculate_fitness
from webapp.components.sidebar_nav import show_sidebar_nav
import webapp.session as session_state

matplotlib.use("Agg")

session_state.init_session()
if hasattr(session_state, "sync_budget_state"):
    session_state.sync_budget_state()
show_sidebar_nav()

st.markdown(
    """
<style>
.field-grid { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:7px; margin-top:8px; }
.field-card { border-radius:8px; padding:7px 5px; text-align:center; box-shadow:0 1px 5px rgba(0,0,0,.10); font-size:.78rem; line-height:1.25; min-height:86px; }
.field-card b { font-size:.82rem; }
.field-card small { font-size:.68rem; }
.ok { background:#E8F5E9; border:2px solid #2E7D32; }
.warn { background:#FFF8E1; border:2px solid #F9A825; }
.bad { background:#FFEBEE; border:2px solid #C62828; }
.algo-title { font-weight:700; color:#173A5E; margin:4px 0 8px; }
@media (max-width: 900px) { .field-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
</style>
""",
    unsafe_allow_html=True,
)

BASE_FIELDS = st.session_state["fields_data"]
ALGORITHMS = ["GA", "SA", "PSO", "Hybrid"]
ALGO_LABELS = {
    "GA": "GA - Genetic Algorithm",
    "SA": "SA - Simulated Annealing",
    "PSO": "PSO - Particle Swarm Optimization",
    "Hybrid": "Hybrid GA + SA",
}
DEFAULT_STEPS = 30
MAX_STEP_SECONDS = 2.0
AUTO_DELAY_SECONDS = 0.2
DEMO_REPAIR_START_STEP = 4
DEMO_REPAIR_FULL_STEP = 14


def build_algo_fields(algo_index):
    rng = np.random.default_rng(2026 + algo_index)
    demand_min = np.array(BASE_FIELDS["demand_min"], dtype=float)
    demand_max = np.array(BASE_FIELDS["demand_max"], dtype=float)
    prices = np.array(BASE_FIELDS["prices"], dtype=float)
    varied_min = np.maximum(1.0, demand_min * rng.uniform(0.92, 1.08, len(demand_min)))
    varied_max = np.maximum(varied_min + 4.0, demand_max * rng.uniform(0.93, 1.10, len(demand_max)))
    varied_total = float(BASE_FIELDS.get("W_total", st.session_state["W_total"])) * float(rng.uniform(0.94, 1.06))
    return {
        "demand_min": varied_min,
        "demand_max": varied_max,
        "prices": np.maximum(0.1, prices * rng.uniform(0.88, 1.16, len(prices))),
        "W_total": varied_total,
        "field_names": list(BASE_FIELDS["field_names"]),
        "crop_names": list(BASE_FIELDS["crop_names"]),
    }


def fit(sol, fields):
    return calculate_fitness(sol, fields["demand_min"], fields["demand_max"], fields["prices"], fields["W_total"])


def repair_ratio(step):
    if step <= DEMO_REPAIR_START_STEP:
        return 0.0
    return min(1.0, (step - DEMO_REPAIR_START_STEP) / (DEMO_REPAIR_FULL_STEP - DEMO_REPAIR_START_STEP))


def demo_loose_bounds(fields):
    span = np.maximum(fields["demand_max"] - fields["demand_min"], 1.0)
    return fields["demand_min"] - 0.35 * span, fields["demand_max"] + 0.35 * span


def rough_demo_sol(fields):
    lo, hi = demo_loose_bounds(fields)
    n_fields = len(fields["demand_min"])
    sol = np.random.uniform(lo, hi)
    if n_fields >= 4:
        low_idx = np.random.choice(n_fields, 2, replace=False)
        remaining = [i for i in range(n_fields) if i not in set(low_idx)]
        high_idx = np.random.choice(remaining, 2, replace=False)
        span = fields["demand_max"] - fields["demand_min"]
        sol[low_idx] = fields["demand_min"][low_idx] - np.random.uniform(0.12, 0.30, len(low_idx)) * span[low_idx]
        sol[high_idx] = fields["demand_max"][high_idx] + np.random.uniform(0.12, 0.30, len(high_idx)) * span[high_idx]
    return np.clip(sol, lo, hi)


def demo_candidate(sol, fields, step):
    lo, hi = demo_loose_bounds(fields)
    raw = np.clip(np.array(sol, dtype=float), lo, hi)
    fixed = repair_solution(
        clip_to_bounds(raw, fields["demand_min"], fields["demand_max"]),
        fields["demand_min"],
        fields["demand_max"],
        fields["W_total"],
    )
    ratio = repair_ratio(step)
    return raw * (1.0 - ratio) + fixed * ratio


def final_repair(sol, fields):
    return repair_solution(
        clip_to_bounds(sol, fields["demand_min"], fields["demand_max"]),
        fields["demand_min"],
        fields["demand_max"],
        fields["W_total"],
    )


def calc_metrics(sol, fields):
    demand_min = fields["demand_min"]
    demand_max = fields["demand_max"]
    total = float(np.sum(sol))
    over_budget = float(max(0.0, total - fields["W_total"]))
    target = float(min(fields["W_total"], np.sum(demand_max)))
    return {
        "total": total,
        "target": target,
        "cost": float(np.dot(fields["prices"], sol)),
        "shortage": float(np.sum(np.maximum(0.0, demand_min - sol))),
        "surplus": float(np.sum(np.maximum(0.0, sol - demand_max))) + over_budget,
        "under_target": float(max(0.0, target - total)),
    }


def make_algo_state(algo, fields):
    n_fields = len(fields["demand_min"])
    current = rough_demo_sol(fields)
    state = {"current": current.copy()}
    if algo in ("GA", "Hybrid"):
        pop = [rough_demo_sol(fields) for _ in range(cfg.GA_POP_SIZE)]
        fits = [fit(x, fields) for x in pop]
        state.update({"pop": pop, "pop_fits": fits, "phase_boundary": DEFAULT_STEPS // 2})
        if algo == "Hybrid":
            best_idx = int(np.argmin(fits))
            state.update(
                {
                    "hybrid_sa_current": pop[best_idx].copy(),
                    "hybrid_sa_fit": float(fits[best_idx]),
                    "hybrid_sa_T": float(cfg.HYBRID_SA_T_MAX),
                }
            )
    elif algo == "SA":
        state["T"] = float(cfg.SA_T_MAX)
    else:
        n_particles = cfg.PSO_N_PARTICLES
        pos = np.array([rough_demo_sol(fields) for _ in range(n_particles)])
        vel = np.random.uniform(-cfg.PSO_V_MAX, cfg.PSO_V_MAX, (n_particles, n_fields))
        pbest = pos.copy()
        pbest_fit = np.array([fit(x, fields) for x in pbest])
        state.update(
            {
                "pos": pos,
                "vel": vel,
                "pbest": pbest,
                "pbest_fit": pbest_fit,
                "gbest": pbest[int(np.argmin(pbest_fit))].copy(),
            }
        )
    return {
        "fields": fields,
        "step": 0,
        "internal": state,
        "current": current.copy(),
        "prev": None,
        "best": None,
        "best_sol": None,
        "first_sol": None,
        "runtime": 0.0,
        "history": [],
        "log": [],
        "timeline": [],
        "slow_warning": False,
    }


def reset_all():
    st.session_state["all_demo_steps"] = DEFAULT_STEPS
    st.session_state["all_demo_running"] = False
    st.session_state["all_demo_algos"] = {
        algo: make_algo_state(algo, build_algo_fields(i)) for i, algo in enumerate(ALGORITHMS)
    }


def ensure_all_demo():
    if (
        "all_demo_algos" not in st.session_state
        or set(st.session_state.get("all_demo_algos", {}).keys()) != set(ALGORITHMS)
        or st.session_state.get("all_demo_steps") != DEFAULT_STEPS
    ):
        reset_all()


def step_algorithm(algo, ctx):
    fields = ctx["fields"]
    state = ctx["internal"]
    cur = state["current"].copy()
    step = int(ctx["step"])
    n_fields = len(fields["demand_min"])
    phase = algo

    if algo == "GA" or (algo == "Hybrid" and step <= state.get("phase_boundary", DEFAULT_STEPS // 2)):
        pop = state["pop"]
        fits = state["pop_fits"]
        elite_n = max(1, int(cfg.GA_ELITE_RATIO * cfg.GA_POP_SIZE))
        order = np.argsort(fits)
        new_pop = [demo_candidate(pop[int(i)], fields, step) for i in order[:elite_n]]

        def tournament():
            idx = np.random.choice(len(pop), cfg.GA_TOURNAMENT_K, replace=False)
            return pop[int(idx[np.argmin([fits[i] for i in idx])])].copy()

        while len(new_pop) < cfg.GA_POP_SIZE:
            p1, p2 = tournament(), tournament()
            if np.random.rand() < cfg.GA_CROSSOVER_RATE:
                dist = np.abs(p1 - p2)
                child = np.random.uniform(
                    np.minimum(p1, p2) - cfg.GA_BLX_ALPHA * dist,
                    np.maximum(p1, p2) + cfg.GA_BLX_ALPHA * dist,
                )
            else:
                child = p1.copy()
            for i in range(n_fields):
                if np.random.rand() < cfg.GA_MUTATION_RATE:
                    child[i] += np.random.normal(0, cfg.GA_MUTATION_SIGMA)
            new_pop.append(demo_candidate(child, fields, step))
        fits = [fit(x, fields) for x in new_pop]
        cur = new_pop[int(np.argmin(fits))].copy()
        cur_fit = float(min(fits))
        state["pop"], state["pop_fits"] = new_pop, fits
        if algo == "Hybrid":
            state["hybrid_sa_current"] = cur.copy()
            state["hybrid_sa_fit"] = cur_fit
        phase = "GA"

    elif algo == "SA":
        temp = state.get("T", cfg.SA_T_MAX)
        for _ in range(cfg.SA_ITER_PER_T):
            nb = cur.copy()
            nb[np.random.randint(n_fields)] += np.random.normal(0, cfg.SA_NEIGHBOR_SIGMA)
            nb = demo_candidate(nb, fields, step)
            delta = fit(nb, fields) - fit(cur, fields)
            if delta < 0 or (temp > 0 and np.random.rand() < math.exp(-delta / max(temp, 1e-9))):
                cur = nb
        state["T"] = max(cfg.SA_T_MIN, temp * cfg.SA_ALPHA)
        cur_fit = float(fit(cur, fields))
        phase = "SA"

    elif algo == "PSO":
        pos, vel = state["pos"], state["vel"]
        pbest, pbest_fit, gbest = state["pbest"], state["pbest_fit"], state["gbest"]
        n_particles = len(pos)
        if repair_ratio(step) >= 1.0:
            pbest = np.array([final_repair(p, fields) for p in pbest])
            pbest_fit = np.array([fit(p, fields) for p in pbest])
            gbest = pbest[int(np.argmin(pbest_fit))].copy()
        r1, r2 = np.random.rand(n_particles, n_fields), np.random.rand(n_particles, n_fields)
        vel = np.clip(
            cfg.PSO_W * vel + cfg.PSO_C1 * r1 * (pbest - pos) + cfg.PSO_C2 * r2 * (gbest - pos),
            -cfg.PSO_V_MAX,
            cfg.PSO_V_MAX,
        )
        pos = pos + vel
        for i in range(n_particles):
            pos[i] = demo_candidate(pos[i], fields, step)
            fval = fit(pos[i], fields)
            if fval < pbest_fit[i]:
                pbest[i], pbest_fit[i] = pos[i].copy(), fval
        gbest = pbest[int(np.argmin(pbest_fit))].copy()
        cur, cur_fit = gbest.copy(), float(np.min(pbest_fit))
        state.update({"pos": pos, "vel": vel, "pbest": pbest, "pbest_fit": pbest_fit, "gbest": gbest})
        phase = "PSO"

    else:
        temp = state.get("hybrid_sa_T", cfg.HYBRID_SA_T_MAX)
        cur = state.get("hybrid_sa_current", cur).copy()
        for _ in range(cfg.SA_ITER_PER_T):
            nb = cur.copy()
            nb[np.random.randint(n_fields)] += np.random.normal(0, cfg.SA_NEIGHBOR_SIGMA)
            nb = demo_candidate(nb, fields, step)
            delta = fit(nb, fields) - fit(cur, fields)
            if delta < 0 or (temp > 0 and np.random.rand() < math.exp(-delta / max(temp, 1e-9))):
                cur = nb
        state["hybrid_sa_T"] = max(cfg.SA_T_MIN, temp * cfg.SA_ALPHA)
        state["hybrid_sa_current"] = cur.copy()
        cur_fit = float(fit(cur, fields))
        state["hybrid_sa_fit"] = cur_fit
        phase = "SA"

    state["current"] = cur.copy()
    ctx["internal"] = state
    return cur, cur_fit, phase


def apply_all_step():
    for algo, ctx in st.session_state["all_demo_algos"].items():
        if ctx["step"] >= DEFAULT_STEPS:
            continue
        t0 = time.perf_counter()
        prev = ctx["current"].copy()
        ctx["step"] += 1
        sol, cur_fit, phase = step_algorithm(algo, ctx)
        elapsed = time.perf_counter() - t0
        if elapsed > MAX_STEP_SECONDS:
            ctx["slow_warning"] = True
            st.session_state["all_demo_running"] = False
        if ctx["first_sol"] is None:
            ctx["first_sol"] = sol.copy()
        if ctx["best"] is None or cur_fit < ctx["best"]:
            ctx["best"] = cur_fit
            ctx["best_sol"] = sol.copy()
        metrics = calc_metrics(sol, ctx["fields"])
        ctx["prev"] = prev.copy()
        ctx["current"] = sol.copy()
        ctx["runtime"] += elapsed
        ctx["history"].append(float(ctx["best"]))
        ctx["timeline"].append(
            {
                "Bước": ctx["step"],
                "Nước": metrics["total"],
                "Chi phí": metrics["cost"],
                "Thời gian": ctx["runtime"],
                "Tối ưu": float(ctx["best"]),
            }
        )
        ctx["log"].append(
            {
                "Bước": ctx["step"],
                "Lượng nước": round(metrics["total"], 2),
                "Điểm": round(cur_fit, 3),
                "Tốt nhất": round(float(ctx["best"]), 3),
                "Thiếu": round(metrics["shortage"], 2),
                "Dư": round(metrics["surplus"], 2),
                "Chi phí": round(metrics["cost"], 2),
                "Thời gian": round(ctx["runtime"], 4),
                "Cơ chế": phase,
            }
        )
    if all(ctx["step"] >= DEFAULT_STEPS for ctx in st.session_state["all_demo_algos"].values()):
        st.session_state["all_demo_running"] = False
        persist_all_results()


def persist_all_results():
    results = st.session_state.get("run_results", {}) or {}
    for algo, ctx in st.session_state["all_demo_algos"].items():
        if ctx["best_sol"] is None:
            continue
        best_sol = final_repair(ctx["best_sol"], ctx["fields"])
        results[algo] = [
            {
                "algo_name": algo,
                "run_id": 0,
                "best_fitness": float(fit(best_sol, ctx["fields"])),
                "best_solution": best_sol.tolist(),
                "first_solution": ctx["first_sol"].tolist() if ctx["first_sol"] is not None else best_sol.tolist(),
                "history": list(ctx["history"]),
                "runtime": float(ctx["runtime"]),
            }
        ]
    if results:
        session_state.save_results(results)


def render_grid(sol, fields):
    cards = []
    for i in range(len(fields["demand_min"])):
        val = float(sol[i])
        mn = float(fields["demand_min"][i])
        mx = float(fields["demand_max"][i])
        css, label = "ok", "Đủ"
        if val < mn:
            css, label = "bad", "Thiếu"
        elif val > mx:
            css, label = "warn", "Dư"
        cards.append(
            "<div class='field-card {css}'><b>{field}</b><br>{crop}"
            "<br><small>Min {mn:.0f} | Max {mx:.0f}</small><br><b>{val:.2f}</b><br><small>{label}</small></div>".format(
                css=css,
                field=html.escape(str(fields["field_names"][i])),
                crop=html.escape(str(fields["crop_names"][i])),
                mn=mn,
                mx=mx,
                val=val,
                label=label,
            )
        )
    st.markdown(f"<div class='field-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_algo_dashboard(algo, ctx):
    with st.container(border=True):
        metrics = calc_metrics(ctx["current"], ctx["fields"])
        best_text = "Chưa chạy" if ctx["best"] is None else f"{ctx['best']:.4f}"
        st.markdown(f"<div class='algo-title'>{ALGO_LABELS[algo]}</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bước", f"{ctx['step']}/{DEFAULT_STEPS}")
        c2.metric("Điểm tốt nhất", best_text)
        c3.metric("Nước", f"{metrics['total']:.1f}")
        c4.metric("Thiếu/Dư", f"{metrics['shortage']:.1f}/{metrics['surplus']:.1f}")
        render_grid(ctx["current"], ctx["fields"])
        if ctx.get("slow_warning"):
            st.warning(f"{algo} có bước chạy vượt {MAX_STEP_SECONDS:.0f}s nên auto đã dừng để tránh treo app.")


def timeline_frame(metric_name):
    rows = []
    for algo, ctx in st.session_state["all_demo_algos"].items():
        for point in ctx["timeline"]:
            rows.append({"Bước": point["Bước"], "Giá trị": point[metric_name], "Thuật toán": algo})
    return pd.DataFrame(rows)


def render_compare_chart(metric_name, title, ylabel):
    df = timeline_frame(metric_name)
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for algo in ALGORITHMS:
        part = df[df["Thuật toán"] == algo] if not df.empty else pd.DataFrame()
        if not part.empty:
            ax.plot(part["Bước"], part["Giá trị"], marker="o", linewidth=2, markersize=3, label=algo)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Bước")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


ensure_all_demo()

with st.sidebar:
    st.caption(f"{DEFAULT_STEPS} bước • auto không dùng delay 3s")
    one_step = st.button("Chạy 1 bước", use_container_width=True)
    run_label = "Dừng auto" if st.session_state.get("all_demo_running") else "Chạy auto"
    if st.button(run_label, type="primary", use_container_width=True):
        st.session_state["all_demo_running"] = not st.session_state.get("all_demo_running", False)
        st.rerun()
    if st.button("Chạy lại từ đầu", use_container_width=True):
        reset_all()
        st.rerun()
    if st.button("Lưu và xem kết quả", use_container_width=True):
        persist_all_results()
        st.switch_page("pages/03_result.py")

st.title("Demo tất cả thuật toán")
st.caption("4 thuật toán chạy song song; mỗi thuật toán có 10 thửa ruộng riêng và bảng nhật ký riêng.")

if one_step:
    apply_all_step()

if st.session_state.get("all_demo_running"):
    apply_all_step()

all_ctx = st.session_state["all_demo_algos"]
max_step = max(ctx["step"] for ctx in all_ctx.values())
st.progress(max_step / DEFAULT_STEPS)
if st.session_state.get("all_demo_running"):
    st.info("Đang chạy auto từng bước cho cả 4 thuật toán.")

row1 = st.columns(2)
for idx, algo in enumerate(ALGORITHMS[:2]):
    with row1[idx]:
        render_algo_dashboard(algo, all_ctx[algo])
row2 = st.columns(2)
for idx, algo in enumerate(ALGORITHMS[2:]):
    with row2[idx]:
        render_algo_dashboard(algo, all_ctx[algo])

st.divider()
st.subheader("Nhật ký chạy của 4 thuật toán")
for pair in (ALGORITHMS[:2], ALGORITHMS[2:]):
    log_cols = st.columns(2)
    for col, algo in zip(log_cols, pair):
        with col:
            with st.container(border=True):
                st.markdown(f"**{algo}**")
                log_df = pd.DataFrame(all_ctx[algo]["log"])
                if log_df.empty:
                    st.info("Chưa có bước chạy.")
                else:
                    st.dataframe(log_df, use_container_width=True, hide_index=True, height=360)

st.divider()
st.subheader("Biểu đồ so sánh cập nhật liên tục")
chart_row1 = st.columns(2)
with chart_row1[0]:
    render_compare_chart("Nước", "Lượng nước", "m³")
with chart_row1[1]:
    render_compare_chart("Chi phí", "Chi phí", "Chi phí bơm")
chart_row2 = st.columns(2)
with chart_row2[0]:
    render_compare_chart("Thời gian", "Thời gian chạy tích lũy", "Giây")
with chart_row2[1]:
    render_compare_chart("Tối ưu", "Điểm tối ưu", "Fitness thấp hơn là tốt hơn")

st.divider()
if st.button("Chuyển qua trang kết quả", type="primary", use_container_width=True, key="bottom_result_page"):
    persist_all_results()
    if st.session_state.get("run_results"):
        st.switch_page("pages/03_result.py")
    else:
        st.warning("Bạn hãy chạy ít nhất 1 bước trước khi xem kết quả.")

if st.session_state.get("all_demo_running") and max_step < DEFAULT_STEPS:
    time.sleep(AUTO_DELAY_SECONDS)
    st.rerun()
