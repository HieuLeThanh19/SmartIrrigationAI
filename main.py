"""
Terminal demo for SmartIrrigationAI.

Run:
    python main.py

The Streamlit app is still the main UI, but this runner now behaves like a
small interactive console demo instead of a silent batch script.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.comparator import build_summary_table, compute_improvement, rank_algorithms
from analysis.runner import ExperimentRunner, build_fields_data
from analysis.statistics import full_statistical_report
from core import config as cfg
from core.constraints import clip_to_bounds, repair_solution
from core.fitness import fitness as calculate_fitness
from core.fitness import fitness_breakdown
from visualization.dashboard import plot_full_dashboard


ALGOS = ["GA", "SA", "PSO", "Hybrid"]
ALGO_LABELS = {
    "GA": "GA - Genetic Algorithm",
    "SA": "SA - Simulated Annealing",
    "PSO": "PSO - Particle Swarm Optimization",
    "Hybrid": "Hybrid GA + SA",
}
DEFAULT_DEMO_STEPS = 30
DEMO_REPAIR_START_STEP = 4
DEMO_REPAIR_FULL_STEP = 14


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause(message="Nhấn Enter để tiếp tục..."):
    input(f"\n{message}")


def ask_choice(prompt, choices, default=None):
    valid = {str(i + 1): value for i, value in enumerate(choices)}
    valid.update({value.lower(): value for value in choices})
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        key = raw.lower()
        if key in valid:
            return valid[key]
        print("Lựa chọn chưa đúng. Hãy nhập số trong menu hoặc tên thuật toán.")


def format_bar(value, width=24):
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def print_header(title):
    clear_screen()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


def print_fields_overview(fields):
    print("Dữ liệu đang dùng giống web Streamlit:")
    print(f"  Ngân sách nước: {fields['W_total']:.2f} m3")
    print("  10 thửa ruộng, mỗi thửa có min/max và giá bơm riêng.\n")
    print("  Thửa       Cây trồng       Min     Max     Giá/m3")
    print("  " + "-" * 58)
    for name, crop, mn, mx, price in zip(
        fields["field_names"],
        fields["crop_names"],
        fields["demand_min"],
        fields["demand_max"],
        fields["prices"],
    ):
        print(f"  {name:<10} {crop:<14} {mn:>6.1f} {mx:>7.1f} {price:>8.2f}")


class TerminalDemo:
    def __init__(self, algo_name, fields_data):
        self.algo_name = algo_name
        self.fields = fields_data
        self.demand_min = np.array(fields_data["demand_min"], dtype=float)
        self.demand_max = np.array(fields_data["demand_max"], dtype=float)
        self.prices = np.array(fields_data["prices"], dtype=float)
        self.W_total = float(fields_data["W_total"])
        self.field_names = fields_data["field_names"]
        self.crop_names = fields_data["crop_names"]
        self.n_fields = len(self.demand_min)
        self.step = 0
        self.history = []
        self.log = []
        self.runtime = 0.0
        self.first_solution = None
        self.best_solution = None
        self.best_fitness = None
        self.prev_solution = None
        self.current = self.rough_demo_solution()
        self.info = {"phase": algo_name}
        self.state = {}
        self.reset_internal_state()

    def fit(self, x):
        return calculate_fitness(x, self.demand_min, self.demand_max, self.prices, self.W_total)

    def demo_loose_bounds(self):
        span = np.maximum(self.demand_max - self.demand_min, 1.0)
        return self.demand_min - 0.35 * span, self.demand_max + 0.35 * span

    def rough_demo_solution(self):
        lo, hi = self.demo_loose_bounds()
        x = np.random.uniform(lo, hi)
        if self.n_fields >= 4:
            low_idx = np.random.choice(self.n_fields, 2, replace=False)
            remaining = [i for i in range(self.n_fields) if i not in set(low_idx)]
            high_idx = np.random.choice(remaining, 2, replace=False)
            span = self.demand_max - self.demand_min
            x[low_idx] = self.demand_min[low_idx] - np.random.uniform(0.12, 0.30, len(low_idx)) * span[low_idx]
            x[high_idx] = self.demand_max[high_idx] + np.random.uniform(0.12, 0.30, len(high_idx)) * span[high_idx]
        return np.clip(x, lo, hi)

    def repair_ratio(self):
        if self.step <= DEMO_REPAIR_START_STEP:
            return 0.0
        return min(1.0, (self.step - DEMO_REPAIR_START_STEP) / (DEMO_REPAIR_FULL_STEP - DEMO_REPAIR_START_STEP))

    def demo_candidate(self, x):
        lo, hi = self.demo_loose_bounds()
        raw = np.clip(np.array(x, dtype=float), lo, hi)
        fixed = repair_solution(
            clip_to_bounds(raw, self.demand_min, self.demand_max),
            self.demand_min,
            self.demand_max,
            self.W_total,
        )
        ratio = self.repair_ratio()
        return raw * (1.0 - ratio) + fixed * ratio

    def final_repair(self, x):
        return repair_solution(
            clip_to_bounds(np.array(x, dtype=float), self.demand_min, self.demand_max),
            self.demand_min,
            self.demand_max,
            self.W_total,
        )

    def metrics(self, sol):
        total = float(np.sum(sol))
        excess_field = float(np.sum(np.maximum(0.0, sol - self.demand_max)))
        over_budget = float(max(0.0, total - self.W_total))
        target = float(min(self.W_total, np.sum(self.demand_max)))
        under_target = float(max(0.0, target - total))
        return {
            "total": total,
            "target": target,
            "cost": float(np.dot(self.prices, sol)),
            "shortage": float(np.sum(np.maximum(0.0, self.demand_min - sol))),
            "surplus": excess_field + over_budget,
            "under_target": under_target,
        }

    def reset_internal_state(self):
        self.state = {"current": self.current.copy()}
        if self.algo_name in ("GA", "Hybrid"):
            pop = [self.rough_demo_solution() for _ in range(cfg.GA_POP_SIZE)]
            fits = [self.fit(x) for x in pop]
            self.state.update({"pop": pop, "pop_fits": fits, "phase_boundary": DEFAULT_DEMO_STEPS // 2})
            if self.algo_name == "Hybrid":
                best_idx = int(np.argmin(fits))
                self.state["hybrid_sa_current"] = pop[best_idx].copy()
                self.state["hybrid_sa_fit"] = float(fits[best_idx])
                self.state["hybrid_sa_T"] = float(cfg.HYBRID_SA_T_MAX)
        elif self.algo_name == "SA":
            self.state["T"] = float(cfg.SA_T_MAX)
        else:
            n = cfg.PSO_N_PARTICLES
            pos = np.array([self.rough_demo_solution() for _ in range(n)])
            vel = np.random.uniform(-cfg.PSO_V_MAX, cfg.PSO_V_MAX, (n, self.n_fields))
            pbest = pos.copy()
            pbest_fit = np.array([self.fit(x) for x in pbest])
            gbest = pbest[int(np.argmin(pbest_fit))].copy()
            self.state.update({"pos": pos, "vel": vel, "pbest": pbest, "pbest_fit": pbest_fit, "gbest": gbest})

    def step_once(self):
        started = time.perf_counter()
        self.prev_solution = self.current.copy()
        self.step += 1
        cur = self.current.copy()
        info = {"phase": self.algo_name}

        if self.algo_name == "GA" or (
            self.algo_name == "Hybrid" and self.step <= self.state.get("phase_boundary", DEFAULT_DEMO_STEPS // 2)
        ):
            cur, fit_value, info = self._step_ga()
        elif self.algo_name == "SA":
            cur, fit_value, info = self._step_sa()
        elif self.algo_name == "PSO":
            cur, fit_value, info = self._step_pso()
        else:
            cur, fit_value, info = self._step_hybrid_sa()

        self.current = cur.copy()
        self.info = info
        if self.first_solution is None:
            self.first_solution = cur.copy()
        if self.best_fitness is None or fit_value < self.best_fitness:
            self.best_fitness = float(fit_value)
            self.best_solution = cur.copy()
        self.history.append(float(self.best_fitness))
        m = self.metrics(cur)
        self.log.append(
            {
                "step": self.step,
                "score": fit_value,
                "best": self.best_fitness,
                "total": m["total"],
                "shortage": m["shortage"],
                "surplus": m["surplus"],
                "phase": info.get("phase", self.algo_name),
            }
        )
        self.runtime += time.perf_counter() - started

    def _step_ga(self):
        pop = self.state["pop"]
        fits = self.state["pop_fits"]
        best_before = float(min(fits))
        avg_before = float(np.mean(fits))
        crossovers = 0
        mutations = 0

        def tournament():
            idx = np.random.choice(len(pop), cfg.GA_TOURNAMENT_K, replace=False)
            return pop[int(idx[np.argmin([fits[i] for i in idx])])].copy()

        elite_n = max(1, int(cfg.GA_ELITE_RATIO * cfg.GA_POP_SIZE))
        order = np.argsort(fits)
        new_pop = [self.demo_candidate(pop[int(i)]) for i in order[:elite_n]]
        while len(new_pop) < cfg.GA_POP_SIZE:
            p1, p2 = tournament(), tournament()
            if np.random.rand() < cfg.GA_CROSSOVER_RATE:
                d = np.abs(p1 - p2)
                child = np.random.uniform(
                    np.minimum(p1, p2) - cfg.GA_BLX_ALPHA * d,
                    np.maximum(p1, p2) + cfg.GA_BLX_ALPHA * d,
                )
                crossovers += 1
            else:
                child = p1.copy()
            for i in range(self.n_fields):
                if np.random.rand() < cfg.GA_MUTATION_RATE:
                    child[i] += np.random.normal(0, cfg.GA_MUTATION_SIGMA)
                    mutations += 1
            new_pop.append(self.demo_candidate(child))

        fits = [self.fit(x) for x in new_pop]
        cur = new_pop[int(np.argmin(fits))].copy()
        cur_fit = float(min(fits))
        self.state["pop"], self.state["pop_fits"] = new_pop, fits
        if self.algo_name == "Hybrid":
            self.state["hybrid_sa_current"] = cur.copy()
            self.state["hybrid_sa_fit"] = cur_fit
        return cur, cur_fit, {
            "phase": "GA",
            "best_before": best_before,
            "best_after": cur_fit,
            "avg_before": avg_before,
            "avg_after": float(np.mean(fits)),
            "elite_n": elite_n,
            "crossovers": crossovers,
            "mutations": mutations,
        }

    def _step_sa(self):
        T = float(self.state.get("T", cfg.SA_T_MAX))
        return self._sa_loop(cur=self.current.copy(), T=T, state_key="T", phase="SA")

    def _step_hybrid_sa(self):
        T = float(self.state.get("hybrid_sa_T", cfg.HYBRID_SA_T_MAX))
        cur = self.state.get("hybrid_sa_current", self.current).copy()
        return self._sa_loop(cur=cur, T=T, state_key="hybrid_sa_T", phase="SA")

    def _sa_loop(self, cur, T, state_key, phase):
        current_fit_before = float(self.fit(cur))
        accepted_better = 0
        accepted_worse = 0
        rejected = 0
        for _ in range(cfg.SA_ITER_PER_T):
            nb = cur.copy()
            nb[np.random.randint(self.n_fields)] += np.random.normal(0, cfg.SA_NEIGHBOR_SIGMA)
            nb = self.demo_candidate(nb)
            delta = self.fit(nb) - self.fit(cur)
            if delta < 0:
                cur = nb
                accepted_better += 1
            elif T > 0 and np.random.rand() < math.exp(-delta / max(T, 1e-9)):
                cur = nb
                accepted_worse += 1
            else:
                rejected += 1
        self.state[state_key] = max(cfg.SA_T_MIN, T * cfg.SA_ALPHA)
        cur_fit = float(self.fit(cur))
        if state_key == "hybrid_sa_T":
            self.state["hybrid_sa_current"] = cur.copy()
            self.state["hybrid_sa_fit"] = cur_fit
        return cur, cur_fit, {
            "phase": phase,
            "temperature_before": T,
            "temperature_after": float(self.state[state_key]),
            "accepted_better": accepted_better,
            "accepted_worse": accepted_worse,
            "rejected": rejected,
            "current_fit_before": current_fit_before,
            "current_fit_after": cur_fit,
        }

    def _step_pso(self):
        pos, vel = self.state["pos"], self.state["vel"]
        pbest, pbest_fit, gbest = self.state["pbest"], self.state["pbest_fit"], self.state["gbest"]
        if self.repair_ratio() >= 1.0:
            pbest = np.array([self.final_repair(p) for p in pbest])
            pbest_fit = np.array([self.fit(p) for p in pbest])
            gbest = pbest[int(np.argmin(pbest_fit))].copy()

        gbest_before = float(np.min(pbest_fit))
        pbest_updates = 0
        r1 = np.random.rand(len(pos), self.n_fields)
        r2 = np.random.rand(len(pos), self.n_fields)
        vel = np.clip(
            cfg.PSO_W * vel + cfg.PSO_C1 * r1 * (pbest - pos) + cfg.PSO_C2 * r2 * (gbest - pos),
            -cfg.PSO_V_MAX,
            cfg.PSO_V_MAX,
        )
        pos = pos + vel
        for i in range(len(pos)):
            pos[i] = self.demo_candidate(pos[i])
            fval = self.fit(pos[i])
            if fval < pbest_fit[i]:
                pbest[i], pbest_fit[i] = pos[i].copy(), fval
                pbest_updates += 1
        gbest = pbest[int(np.argmin(pbest_fit))].copy()
        cur_fit = float(np.min(pbest_fit))
        self.state.update({"pos": pos, "vel": vel, "pbest": pbest, "pbest_fit": pbest_fit, "gbest": gbest})
        return gbest.copy(), cur_fit, {
            "phase": "PSO",
            "gbest_before": gbest_before,
            "gbest_after": cur_fit,
            "pbest_updates": pbest_updates,
            "mean_velocity": float(np.mean(np.linalg.norm(vel, axis=1))),
        }

    def decision_text(self):
        sol = self.current
        shortage = np.maximum(0.0, self.demand_min - sol)
        surplus = np.maximum(0.0, sol - self.demand_max)
        capacity = np.maximum(0.0, self.demand_max - sol)
        total = float(np.sum(sol))
        over_budget = max(0.0, total - self.W_total)
        target = min(self.W_total, float(np.sum(self.demand_max)))
        under_target = max(0.0, target - total)

        if float(np.sum(shortage)) > 0.01:
            idx = int(np.argmax(shortage))
            return f"Đang thiếu nước nhiều nhất ở {self.field_names[idx]}; bước sau ưu tiên bù thêm."
        if float(np.sum(surplus)) > 0.01:
            idx = int(np.argmax(surplus))
            return f"Đang dư nước ở {self.field_names[idx]}; thuật toán sẽ giảm để tránh phạt lãng phí."
        if over_budget > 0.01:
            candidates = np.where((sol - self.demand_min) > 0.01)[0]
            idx = max(candidates, key=lambda i: self.prices[i]) if len(candidates) else int(np.argmax(self.prices))
            return f"Đang vượt ngân sách {over_budget:.2f} m3; giảm trước ở {self.field_names[idx]} vì chi phí cao."
        if under_target > 0.01:
            receivers = np.where(capacity > 0.01)[0]
            if len(receivers):
                idx = min(receivers, key=lambda i: self.prices[i])
                return f"Còn thiếu mục tiêu {under_target:.2f} m3; ưu tiên thêm vào {self.field_names[idx]} có giá bơm thấp."
        return "Phương án đã hợp lệ; bước sau chủ yếu tối ưu chi phí và giữ cân bằng nước."

    def mechanism_text(self):
        if self.step == 0:
            return "Sẵn sàng chạy. Nhấn Enter để xem thuật toán cập nhật phương án đầu tiên."
        info = self.info
        phase = info.get("phase", self.algo_name)
        if phase == "GA":
            return (
                f"GA: giữ {info.get('elite_n', 0)} cá thể tốt, lai {info.get('crossovers', 0)} lần, "
                f"đột biến {info.get('mutations', 0)} gene."
            )
        if self.algo_name == "PSO":
            return (
                f"PSO: {info.get('pbest_updates', 0)} hạt cải thiện pbest, "
                f"vận tốc TB {info.get('mean_velocity', 0.0):.2f}."
            )
        return (
            f"SA: T {info.get('temperature_before', 0.0):.2f} -> {info.get('temperature_after', 0.0):.2f}, "
            f"nhận tốt hơn {info.get('accepted_better', 0)}, nhận xấu hơn {info.get('accepted_worse', 0)}, "
            f"từ chối {info.get('rejected', 0)}."
        )

    def render(self):
        print_header(f"Demo thuật toán: {ALGO_LABELS.get(self.algo_name, self.algo_name)}")
        progress = self.step / DEFAULT_DEMO_STEPS
        best_label = "Chưa có" if self.best_fitness is None else f"{self.best_fitness:.4f}"
        print(f"Bước: {self.step:02d}/{DEFAULT_DEMO_STEPS}  [{format_bar(progress)}]  Best: {best_label}")
        print(f"Cơ chế: {self.mechanism_text()}")
        print(f"Quyết định: {self.decision_text()}\n")

        m = self.metrics(self.current)
        parts = fitness_breakdown(self.current, self.demand_min, self.demand_max, self.prices, self.W_total)
        print(
            f"Tổng nước {m['total']:.2f}/{m['target']:.2f} m3 | "
            f"Thiếu {m['shortage']:.2f} | Dư/vượt {m['surplus']:.2f} | "
            f"Chi phí {m['cost']:.2f} | Điểm {parts['total']:.2f}\n"
        )
        print("Thửa       Cây trồng       Nước    Min    Max    Giá   Trạng thái")
        print("-" * 76)
        for name, crop, water, mn, mx, price in zip(
            self.field_names,
            self.crop_names,
            self.current,
            self.demand_min,
            self.demand_max,
            self.prices,
        ):
            if water < mn:
                status = "Thiếu"
            elif water > mx:
                status = "Dư"
            else:
                status = "OK"
            print(f"{name:<10} {crop:<14} {water:>7.2f} {mn:>6.1f} {mx:>6.1f} {price:>6.2f}   {status}")

        if self.log:
            print("\nNhật ký 5 bước gần nhất:")
            print("  Bước   Pha      Điểm       Best       Tổng     Thiếu    Dư")
            for row in self.log[-5:]:
                print(
                    f"  {row['step']:>3}    {row['phase']:<6} "
                    f"{row['score']:>9.2f} {row['best']:>9.2f} "
                    f"{row['total']:>8.2f} {row['shortage']:>8.2f} {row['surplus']:>7.2f}"
                )

    def final_result(self):
        if self.best_solution is None:
            self.best_solution = self.final_repair(self.current)
            self.best_fitness = float(self.fit(self.best_solution))
        best = self.final_repair(self.best_solution)
        fit_value = float(self.fit(best))
        return {
            "algo_name": self.algo_name,
            "run_id": 0,
            "best_fitness": fit_value,
            "best_solution": best.tolist(),
            "first_solution": (self.first_solution.tolist() if self.first_solution is not None else best.tolist()),
            "history": self.history,
            "runtime": round(self.runtime, 4),
        }


def run_interactive_demo(fields_data):
    print_header("Chọn thuật toán demo")
    for idx, algo in enumerate(ALGOS, start=1):
        print(f"{idx}. {ALGO_LABELS[algo]}")
    algo = ask_choice("\nNhập lựa chọn [1-4, mặc định 1]: ", ALGOS, default="GA")
    demo = TerminalDemo(algo, fields_data)

    while True:
        demo.render()
        if demo.step >= DEFAULT_DEMO_STEPS:
            print("\nĐã chạy đủ 30 bước giống demo Streamlit.")
            print("Kết quả đã sẵn sàng để xem ở màn hình tổng kết.")
            pause()
            return {algo: [demo.final_result()]}

        print("\nLệnh: Enter = chạy 1 bước | a = auto đến 30 bước | r = chạy lại | q = về menu")
        cmd = input("> ").strip().lower()
        if cmd == "q":
            return None
        if cmd == "r":
            demo = TerminalDemo(algo, fields_data)
            continue
        if cmd == "a":
            while demo.step < DEFAULT_DEMO_STEPS:
                demo.step_once()
                demo.render()
                time.sleep(0.35)
            pause("\nAuto xong. Nhấn Enter để xem tổng kết...")
            return {algo: [demo.final_result()]}
        demo.step_once()


def run_compare_all(fields_data):
    print_header("So sánh đầy đủ GA, SA, PSO, Hybrid")
    print(f"Chạy {len(ALGOS)} thuật toán x {cfg.N_RUNS} lần theo cấu hình trong core/config.py.\n")

    def on_progress(algo_name, run_num, total_runs, done, total):
        pct = done / total
        print(
            f"\r[{format_bar(pct)}] {pct * 100:5.1f}%  "
            f"{algo_name} lần {run_num}/{total_runs}",
            end="",
            flush=True,
        )

    runner = ExperimentRunner(ALGOS, fields_data, config=cfg, n_runs=cfg.N_RUNS)
    results = runner.run_all(callback=on_progress)
    print("\n")
    render_results(results, fields_data, save_dashboard=True)
    pause()


def render_results(results, fields_data, save_dashboard=False):
    print("\nBẢNG SO SÁNH\n")
    summary = build_summary_table(results)
    try:
        from tabulate import tabulate

        print(tabulate(summary, headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        print(summary.to_string(index=False))

    print("\nXẾP HẠNG:", " > ".join(rank_algorithms(results)))

    imp = compute_improvement(results)
    if imp:
        print("\n% CẢI THIỆN SO VỚI GA:")
        for name, value in imp.items():
            print(f"  {name}: {value:+.1f}%")

    best_algo = None
    best_run = None
    for algo, runs in results.items():
        for run in runs:
            if best_run is None or run["best_fitness"] < best_run["best_fitness"]:
                best_algo, best_run = algo, run
    if best_run:
        sol = np.array(best_run["best_solution"], dtype=float)
        print(f"\nPHÂN BỔ TỐT NHẤT: {best_algo} | điểm {best_run['best_fitness']:.4f}")
        for name, crop, water in zip(fields_data["field_names"], fields_data["crop_names"], sol):
            print(f"  {name:<10} {crop:<14} {water:>7.2f} m3")

    if len(results) > 1:
        print("\nTHỐNG KÊ:\n")
        print(full_statistical_report(results))

    if save_dashboard:
        out_path = "dashboard.png"
        plot_full_dashboard(results, fields_data, save_path=out_path)
        print(f"\nDashboard đã lưu: {out_path}")


def main():
    setup_console()
    fields_data = build_fields_data(W_total=cfg.W_TOTAL)

    while True:
        print_header("SmartIrrigationAI - Terminal Demo")
        print("Terminal này bám theo demo Streamlit: chọn thuật toán, chạy từng bước, xem chỉ số và quyết định ngay trên màn hình.\n")
        print("1. Demo tương tác 30 bước")
        print("2. So sánh đầy đủ và xuất dashboard.png")
        print("3. Xem dữ liệu 10 thửa ruộng")
        print("4. Mở gợi ý chạy Streamlit")
        print("0. Thoát")
        choice = input("\nChọn chức năng [1]: ").strip() or "1"

        if choice == "0":
            print("Tạm biệt. Khi cần UI đầy đủ: python -m streamlit run webapp/app.py")
            return
        if choice == "1":
            result = run_interactive_demo(fields_data)
            if result:
                print_header("Tổng kết demo terminal")
                render_results(result, fields_data, save_dashboard=False)
                pause()
        elif choice == "2":
            run_compare_all(fields_data)
        elif choice == "3":
            print_header("Dữ liệu 10 thửa ruộng")
            print_fields_overview(fields_data)
            pause()
        elif choice == "4":
            print_header("Chạy giao diện Streamlit")
            print("Lệnh chính:")
            print("  python -m streamlit run webapp/app.py")
            print("\nTrên web app bạn có sidebar tiếng Việt, nút Chạy 1 bước, Chạy demo/Dừng và trang kết quả.")
            pause()
        else:
            print("Mục này chưa có trong menu.")
            pause()


if __name__ == "__main__":
    main()
