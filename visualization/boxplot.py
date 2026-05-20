import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_boxplot(all_fitnesses: dict, algo_names: list, save_path: str = None):
    """Boxplot phân phối fitness qua N_RUNS lần chạy."""
    data   = [all_fitnesses.get(a, [0]) for a in algo_names]
    colors = ["#E53935", "#1E88E5", "#43A047", "#8E24AA"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=algo_names, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2.5})

    for patch, color in zip(bp["boxes"], colors[:len(algo_names)]):
        patch.set_facecolor(color); patch.set_alpha(0.75)

    for i, d in enumerate(data):
        if d:
            ax.plot(i + 1, np.mean(d), marker="^", color="yellow",
                    markersize=9, zorder=5, label="Trung bình" if i == 0 else "")

    ax.set_ylabel("Điểm tối ưu", fontsize=11)
    ax.set_title("Phân phối điểm tối ưu qua các lần chạy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
