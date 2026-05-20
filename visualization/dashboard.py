import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from visualization.convergence import plot_convergence
from visualization.allocation  import plot_water_allocation
from visualization.boxplot     import plot_fitness_boxplot


def plot_full_dashboard(results: dict, fields_data: dict, save_path: str = None):
    algo_names = list(results.keys())
    demand_min = fields_data["demand_min"]
    demand_max = fields_data["demand_max"]
    field_names = fields_data.get("field_names", [f"T{i+1}" for i in range(len(demand_min))])

    histories = {}
    for algo, runs in results.items():
        best_run = min(runs, key=lambda r: r["best_fitness"])
        histories[algo] = best_run["history"]

    best_fitness = float("inf")
    best_sol     = None
    for algo, runs in results.items():
        for r in runs:
            if r["best_fitness"] < best_fitness:
                best_fitness = r["best_fitness"]
                best_sol     = np.array(r["best_solution"])

    all_fitnesses = {algo: [r["best_fitness"] for r in runs]
                     for algo, runs in results.items()}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("SmartIrrigationAI — Bảng kết quả", fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(2, 2, 1)
    for algo in algo_names:
        ax1.plot(histories[algo], label=algo, linewidth=1.5)
    ax1.set_title("Đường hội tụ")
    ax1.set_xlabel("Vòng lặp")
    ax1.set_ylabel("Điểm tối ưu")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(len(demand_min))
    ax2.bar(x - 0.25, demand_min, 0.25, label="Min", color="#EF5350", alpha=0.8)
    ax2.bar(x,         best_sol,  0.25, label="Thực tế", color="#42A5F5", alpha=0.9)
    ax2.bar(x + 0.25, demand_max, 0.25, label="Max", color="#FFA726", alpha=0.8)
    ax2.set_title("Phân bổ nước tốt nhất")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T{i+1}" for i in range(len(demand_min))], fontsize=7)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    data   = [all_fitnesses[a] for a in algo_names]
    bp = ax3.boxplot(data, patch_artist=True)
    ax3.set_xticklabels(algo_names)
    ax3.set_title("Phân phối điểm tối ưu")
    ax3.set_ylabel("Điểm tối ưu")
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    runtimes = [np.mean([r["runtime"] for r in results[a]]) for a in algo_names]
    bars = ax4.bar(algo_names, runtimes, color=["#2196F3","#FF9800","#9C27B0","#F44336"][:len(algo_names)])
    ax4.set_title("Thời gian chạy trung bình")
    ax4.set_ylabel("Giây (s)")
    for bar, val in zip(bars, runtimes):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}s",
                 ha="center", fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
