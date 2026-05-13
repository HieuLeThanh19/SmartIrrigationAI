# visualization/allocation.py — Biểu đồ phân bổ nước
import matplotlib.pyplot as plt
import numpy as np


def plot_water_allocation(solution: np.ndarray, demand_min: np.ndarray,
                          demand_max: np.ndarray, field_names: list,
                          save_path: str = None):
    """Bar chart: mỗi thửa ruộng 1 nhóm 3 cột (min / actual / max)."""
    n = len(field_names)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_min = ax.bar(x - width,   demand_min, width, label="Min",         color="#EF5350", alpha=0.8)
    bars_act = ax.bar(x,           solution,   width, label="Phân bổ AI",  color="#42A5F5", alpha=0.9)
    bars_max = ax.bar(x + width,   demand_max, width, label="Max",         color="#FFA726", alpha=0.8)

    # Số liệu trên đầu cột thực tế
    for bar in bars_act:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(field_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Lượng nước (m³/ngày)", fontsize=11)
    ax.set_title("Phân bổ nước thực tế so với nhu cầu min/max", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_allocation_comparison(solutions_dict: dict, demand_min: np.ndarray,
                               demand_max: np.ndarray, field_names: list):
    """So sánh phân bổ của nhiều thuật toán cùng lúc."""
    n_algos = len(solutions_dict)
    n = len(field_names)
    x = np.arange(n)
    width = 0.7 / max(n_algos, 1)
    colors = ["#42A5F5", "#EF5350", "#43A047", "#8E24AA", "#FF7043"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, (algo, sol) in enumerate(solutions_dict.items()):
        offset = (idx - n_algos / 2 + 0.5) * width
        ax.bar(x + offset, sol, width, label=algo, color=colors[idx % len(colors)], alpha=0.8)

    # Min/Max reference lines
    ax.step(np.append(x - 0.4, x[-1] + 0.4), np.append(demand_min, demand_min[-1]),
            where="mid", color="red", linestyle="--", linewidth=1, label="Min")
    ax.step(np.append(x - 0.4, x[-1] + 0.4), np.append(demand_max, demand_max[-1]),
            where="mid", color="orange", linestyle="--", linewidth=1, label="Max")

    ax.set_xticks(x); ax.set_xticklabels(field_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("m³/ngày"); ax.set_title("So sánh phân bổ các thuật toán", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig
