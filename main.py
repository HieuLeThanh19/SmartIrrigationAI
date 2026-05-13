"""
main.py — Chạy so sánh tất cả thuật toán từ terminal.
Dùng: python main.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import data as data_module
from core import config as cfg
from analysis.runner import ExperimentRunner
from analysis.comparator import build_summary_table, rank_algorithms, compute_improvement
from analysis.statistics import full_statistical_report
from visualization.dashboard import plot_full_dashboard

ALGOS = ["GA", "SA", "PSO", "Hybrid"]


def main():
    print("=" * 60)
    print("  SmartIrrigationAI — Terminal Runner")
    print("=" * 60)

    fields_data = {
        "demand_min":  data_module.get_demand_min(),
        "demand_max":  data_module.get_demand_max(),
        "prices":      data_module.get_prices(),
        "W_total":     cfg.W_TOTAL,
        "field_names": data_module.get_field_names(),
        "crop_names":  data_module.get_crop_names(),
    }

    def on_progress(algo_name, run_num, total_runs, done, total):
        pct = int(done / total * 100)
        bar = "#" * (pct // 5) + "." * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {algo_name} lần {run_num}/{total_runs}", end="", flush=True)

    print(f"\nChạy {len(ALGOS)} thuật toán × {cfg.N_RUNS} lần ...\n")
    runner  = ExperimentRunner(ALGOS, fields_data, config=cfg, n_runs=cfg.N_RUNS)
    results = runner.run_all(callback=on_progress)
    print("\n")

    # Bảng kết quả
    print("\nBẢNG SO SÁNH\n")
    summary = build_summary_table(results)
    try:
        from tabulate import tabulate
        print(tabulate(summary, headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        print(summary.to_string(index=False))

    print("\nXẾP HẠNG:", " > ".join(rank_algorithms(results)))

    imp = compute_improvement(results)
    print("\n% CẢI THIỆN SO VỚI GA:")
    for k, v in imp.items():
        print(f"   {k}: {v:+.1f}%")

    print("\nTHỐNG KÊ:\n")
    print(full_statistical_report(results))

    # Lưu dashboard PNG
    out_path = "dashboard.png"
    plot_full_dashboard(results, fields_data, save_path=out_path)
    print(f"\nBảng kết quả đã lưu: {out_path}")
    print("Web UI: streamlit run webapp/app.py")


if __name__ == "__main__":
    main()
