# visualization/convergence.py — Đường hội tụ fitness
import matplotlib.pyplot as plt
import numpy as np


ALGO_COLORS = {
    "GA":     "#E53935",
    "SA":     "#1E88E5",
    "PSO":    "#43A047",
    "Hybrid": "#8E24AA",
}


def plot_convergence(histories: dict, algo_names: list,
                     phase_boundary: int = None, save_path: str = None):
    """Vẽ đường hội tụ fitness theo iteration."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for algo in algo_names:
        h = histories.get(algo, [])
        if not h:
            continue
        color = ALGO_COLORS.get(algo, "gray")
        ax.plot(h, label=algo, color=color, linewidth=2.2, alpha=0.9)

    if phase_boundary is not None:
        ax.axvline(phase_boundary, linestyle="--", color="#FF7043",
                   linewidth=1.5, label="GA→SA (Hybrid)")

    ax.set_xlabel("Vòng lặp / Thế hệ", fontsize=11)
    ax.set_ylabel("Điểm tối ưu (thấp = tốt)", fontsize=11)
    ax.set_title("Đường hội tụ điểm tối ưu", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence_with_std(histories_all_runs: dict, algo_names: list):
    """Vẽ đường trung bình ± std qua N_RUNS lần chạy."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for algo in algo_names:
        runs = histories_all_runs.get(algo, [])
        if not runs:
            continue
        min_len = min(len(h) for h in runs)
        arr = np.array([h[:min_len] for h in runs])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        color = ALGO_COLORS.get(algo, "gray")
        x = np.arange(min_len)
        ax.plot(x, mean, label=algo, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Vòng lặp"); ax.set_ylabel("Điểm tối ưu (trung bình ± độ lệch chuẩn)")
    ax.set_title("Đường hội tụ trung bình qua nhiều lần chạy", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
