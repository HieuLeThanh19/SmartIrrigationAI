# analysis/statistics.py — Kiểm định thống kê
import numpy as np
from scipy import stats


def t_test_compare(results_a: list, results_b: list, alpha: float = 0.05) -> dict:
    """Welch t-test giữa 2 thuật toán."""
    if not results_a or not results_b:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False, "better_algo": "equal"}
    t_stat, p_value = stats.ttest_ind(results_a, results_b, equal_var=False, nan_policy="omit")
    significant = bool(p_value < alpha)
    mean_a, mean_b = np.mean(results_a), np.mean(results_b)
    better = "equal" if abs(mean_a - mean_b) < 1e-12 else ("A" if mean_a < mean_b else "B")
    return {
        "t_stat":      round(float(t_stat), 4),
        "p_value":     round(float(p_value), 6),
        "significant": significant,
        "better_algo": better,
    }


def wilcoxon_test(results_a: list, results_b: list, alpha: float = 0.05) -> dict:
    """Kiểm định Wilcoxon (không cần phân phối chuẩn)."""
    if not results_a or not results_b or len(results_a) != len(results_b):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    try:
        stat, p_value = stats.wilcoxon(results_a, results_b)
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    return {
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p_value), 6),
        "significant": bool(p_value < alpha),
    }


def compute_effect_size(results_a: list, results_b: list) -> float:
    """Tính Cohen's d."""
    if len(results_a) < 2 or len(results_b) < 2:
        return 0.0
    mean_a, mean_b = np.mean(results_a), np.mean(results_b)
    std_pool = np.sqrt((np.std(results_a, ddof=1)**2 + np.std(results_b, ddof=1)**2) / 2)
    if std_pool == 0:
        return 0.0
    return round(float(abs(mean_a - mean_b) / std_pool), 4)


def full_statistical_report(all_results: dict) -> str:
    """Báo cáo thống kê đầy đủ so sánh tất cả cặp thuật toán."""
    algos = list(all_results.keys())
    lines = ["=" * 60, "BÁO CÁO THỐNG KÊ SO SÁNH THUẬT TOÁN", "=" * 60, ""]

    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            fa = [r["best_fitness"] for r in all_results[a]]
            fb = [r["best_fitness"] for r in all_results[b]]
            tt = t_test_compare(fa, fb)
            d  = compute_effect_size(fa, fb)
            wil = wilcoxon_test(fa, fb)
            sig = "Có ý nghĩa thống kê" if tt["significant"] else "Không có ý nghĩa"
            effect = ("nhỏ" if d < 0.2 else "vừa" if d < 0.8 else "lớn")
            better_name = a if tt["better_algo"] == "A" else b if tt["better_algo"] == "B" else "Tương đương"
            lines += [
                f"[{a}] vs [{b}]",
                f"  t-statistic : {tt['t_stat']}",
                f"  p-value     : {tt['p_value']}",
                f"  Wilcoxon p  : {wil['p_value']}",
                f"  Kết luận    : {sig}",
                f"  Cohen's d   : {d} ({effect})",
                f"  Tốt hơn     : {better_name}",
                "",
            ]

    # Xếp hạng cuối cùng
    means = {a: float(np.mean([r["best_fitness"] for r in runs]))
             for a, runs in all_results.items()}
    ranked = sorted(means, key=means.get)
    lines += ["=" * 60, "XẾP HẠNG (điểm trung bình thấp = tốt):"]
    for rank, algo in enumerate(ranked, 1):
        lines.append(f"  {rank}. {algo}: trung bình = {means[algo]:.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)
