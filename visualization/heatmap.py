# visualization/heatmap.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_param_heatmap(data_2d: np.ndarray, param1_values, param2_values,
                       param1_name: str, param2_name: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data_2d, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(param2_values)))
    ax.set_yticks(range(len(param1_values)))
    ax.set_xticklabels(param2_values)
    ax.set_yticklabels(param1_values)
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Điểm tối ưu trung bình")
    for i in range(len(param1_values)):
        for j in range(len(param2_values)):
            ax.text(j, i, f"{data_2d[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.tight_layout()
    return fig


def plot_ga_heatmap(tuner_results: dict):
    grid   = tuner_results.get("grid", {})
    pop    = sorted({eval(k)["GA_POP_SIZE"]    for k in grid})
    mut    = sorted({eval(k)["GA_MUTATION_RATE"] for k in grid})
    data   = np.zeros((len(pop), len(mut)))
    for i, p in enumerate(pop):
        for j, m in enumerate(mut):
            key = str({"GA_POP_SIZE": p, "GA_MUTATION_RATE": m})
            data[i, j] = grid.get(key, 0)
    return plot_param_heatmap(data, pop, mut, "Kích thước quần thể", "Tỉ lệ đột biến",
                              "GA: kích thước quần thể và tỉ lệ đột biến")


def plot_sa_heatmap(tuner_results: dict):
    grid  = tuner_results.get("grid", {})
    alpha = sorted({eval(k)["SA_ALPHA"] for k in grid})
    tmax  = sorted({eval(k)["SA_T_MAX"] for k in grid})
    data  = np.zeros((len(alpha), len(tmax)))
    for i, a in enumerate(alpha):
        for j, t in enumerate(tmax):
            key = str({"SA_ALPHA": a, "SA_T_MAX": t})
            data[i, j] = grid.get(key, 0)
    return plot_param_heatmap(data, alpha, tmax, "Hệ số làm nguội", "Nhiệt độ khởi đầu",
                              "SA: hệ số làm nguội và nhiệt độ khởi đầu")
