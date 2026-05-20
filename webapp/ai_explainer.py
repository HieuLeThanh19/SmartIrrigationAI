"""AI-style explanations for irrigation optimization results.

This module is intentionally offline: it explains and answers from the real
numbers already produced by the app, so the demo works without an API key.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import streamlit as st

from core.fitness import fitness_breakdown


@dataclass
class AlgoInsight:
    name: str
    fitness: float
    runtime: float
    total_water: float
    cost: float
    avg_cost: float
    shortage: float
    surplus: float
    under_target: float
    shortage_penalty: float
    surplus_penalty: float
    underuse_penalty: float
    cost_score: float
    score: float
    start_total_water: float | None = None
    start_cost: float | None = None
    start_shortage: float | None = None
    start_surplus: float | None = None
    start_under_target: float | None = None

    @property
    def issue_total(self) -> float:
        return self.shortage + self.surplus + self.under_target

    @property
    def water_saved(self) -> float | None:
        if self.start_total_water is None:
            return None
        return self.start_total_water - self.total_water

    @property
    def cost_saved(self) -> float | None:
        if self.start_cost is None:
            return None
        return self.start_cost - self.cost


def _style_once() -> None:
    return None


def _field_arrays(fields_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return (
        np.array(fields_data["demand_min"], dtype=float),
        np.array(fields_data["demand_max"], dtype=float),
        np.array(fields_data["prices"], dtype=float),
        float(fields_data["W_total"]),
    )


def _raw_metrics(solution, fields_data: dict) -> dict:
    sol = np.array(solution, dtype=float)
    dmin, dmax, prices, w_total = _field_arrays(fields_data)
    total = float(np.sum(sol))
    cost = float(np.dot(prices, sol))
    target = float(min(w_total, np.sum(dmax)))
    return {
        "total_water": total,
        "cost": cost,
        "avg_cost": cost / max(total, 1e-9),
        "shortage": float(np.sum(np.maximum(0.0, dmin - sol))),
        "surplus": float(np.sum(np.maximum(0.0, sol - dmax)) + max(0.0, total - w_total)),
        "under_target": float(max(0.0, target - total)),
    }


def _insight_from_run(algo: str, run: dict, fields_data: dict) -> AlgoInsight:
    dmin, dmax, prices, w_total = _field_arrays(fields_data)
    sol = np.array(run["best_solution"], dtype=float)
    raw = _raw_metrics(sol, fields_data)
    parts = fitness_breakdown(sol, dmin, dmax, prices, w_total)
    start = _raw_metrics(run["first_solution"], fields_data) if run.get("first_solution") is not None else {}
    return AlgoInsight(
        name=algo,
        fitness=float(run["best_fitness"]),
        runtime=float(run.get("runtime", 0.0)),
        total_water=raw["total_water"],
        cost=raw["cost"],
        avg_cost=raw["avg_cost"],
        shortage=raw["shortage"],
        surplus=raw["surplus"],
        under_target=raw["under_target"],
        shortage_penalty=float(parts["shortage_penalty"]),
        surplus_penalty=float(parts["waste_penalty"]),
        underuse_penalty=float(parts["underuse_penalty"]),
        cost_score=float(parts["cost"]),
        score=float(parts["total"]),
        start_total_water=start.get("total_water"),
        start_cost=start.get("cost"),
        start_shortage=start.get("shortage"),
        start_surplus=start.get("surplus"),
        start_under_target=start.get("under_target"),
    )


def _best_runs(results: dict, ranked: list[str] | None = None) -> list[AlgoInsight]:
    fields_data = st.session_state["fields_data"]
    names = ranked or list(results.keys())
    insights = []
    for algo in names:
        if algo in results and results[algo]:
            best_run = min(results[algo], key=lambda r: r["best_fitness"])
            insights.append(_insight_from_run(algo, best_run, fields_data))
    return insights


def _fmt_delta(value: float | None, unit: str = "") -> str:
    if value is None:
        return "chưa có dữ liệu bước đầu"
    if abs(value) < 0.01:
        return f"gần như không đổi{unit}"
    verb = "giảm" if value > 0 else "tăng"
    return f"{verb} {abs(value):.2f}{unit}"


def _dominant_penalty(insight: AlgoInsight) -> str:
    parts = {
        "thiếu nước": insight.shortage_penalty,
        "dư/vượt ngưỡng": insight.surplus_penalty,
        "chưa dùng đủ mục tiêu": insight.underuse_penalty,
        "chi phí bơm": insight.cost_score,
    }
    return max(parts, key=parts.get)


def _run_explanation(insight: AlgoInsight) -> str:
    dominant = _dominant_penalty(insight)
    water_sentence = _fmt_delta(insight.water_saved, " m3")
    cost_sentence = _fmt_delta(insight.cost_saved, "")
    return f"""
#### Giải thích AI cho {insight.name}

> **Lưu ý:** Điểm fitness càng thấp càng tốt. Điểm không chỉ nhìn vào lượng nước giảm, mà cộng cả thiếu nước,
> nước dư/vượt ngưỡng, phần chưa đạt mục tiêu phân bổ và chi phí bơm.

**1. Thuật toán đã làm được gì?**  
{insight.name} kết thúc với điểm **{insight.fitness:.4f}** sau **{insight.runtime:.3f}s**.
So với bước đầu, tổng nước **{water_sentence}**, chi phí bơm **{cost_sentence}**.
Phương án cuối dùng **{insight.total_water:.2f} m3**, chi phí bơm **{insight.cost:.2f}**,
trung bình **{insight.avg_cost:.2f}/m3**.

**2. Vì sao điểm lại như vậy?**  
Phần ảnh hưởng lớn nhất hiện là **{dominant}**. Nếu một thuật toán giảm nước rất mạnh nhưng làm
một số thửa bị thiếu hoặc chưa dùng đủ mục tiêu nước, điểm có thể vẫn cao. Ngược lại, thuật toán
dùng nhiều nước hơn một chút nhưng đặt đúng thửa và ít vi phạm ràng buộc thường được điểm thấp hơn.

**3. Các lỗi còn lại trong phương án**  
Thiếu nước: **{insight.shortage:.2f} m3**. Dư/vượt ngưỡng: **{insight.surplus:.2f} m3**.
Thiếu so với mục tiêu phân bổ: **{insight.under_target:.2f} m3**.

**4. Nên hiểu kết quả này thế nào?**  
Nếu mục tiêu là cân bằng giữa tiết kiệm nước và an toàn cho cây, hãy ưu tiên điểm fitness thấp.
Nếu mục tiêu riêng là cắt tổng nước, có thể nhìn thêm chỉ số tổng nước, nhưng cần kiểm tra xem
việc cắt đó có làm tăng thiếu nước hoặc bỏ phí mục tiêu tưới hay không.
"""


def _comparison_explanation(insights: list[AlgoInsight]) -> str:
    ordered = sorted(insights, key=lambda x: x.fitness)
    best = ordered[0]
    least_water = min(ordered, key=lambda x: x.total_water)
    cheapest = min(ordered, key=lambda x: x.cost)
    cleanest = min(ordered, key=lambda x: x.issue_total)
    rows = []
    for pos, item in enumerate(ordered, start=1):
        saved = _fmt_delta(item.water_saved, " m3")
        rows.append(
            f"**Hạng {pos}. {item.name}**: điểm {item.fitness:.4f}, dùng {item.total_water:.2f} m3, "
            f"chi phí {item.cost:.2f}, nước so với bước đầu {saved}, còn thiếu {item.shortage:.2f} m3, "
            f"dư/vượt {item.surplus:.2f} m3."
        )

    paradox = ""
    if least_water.name != best.name:
        paradox = (
            f"> **Vì sao {least_water.name} dùng ít nước hơn nhưng không đứng đầu?**  \n> "
            f"{least_water.name} dùng ít nước nhất ({least_water.total_water:.2f} m3), nhưng điểm fitness còn tính cả "
            f"thiếu nước, dư/vượt ngưỡng, thiếu mục tiêu và chi phí. {best.name} thắng vì tổng các khoản phạt của nó "
            f"nhẹ hơn, nên điểm cuối thấp hơn."
        )
    else:
        paradox = (
            f"> {best.name} vừa có điểm tốt nhất vừa là phương án dùng ít nước nhất trong nhóm đang chạy. "
            "Trường hợp này khá dễ đọc: tiết kiệm nước không làm tăng phạt ràng buộc quá nhiều."
        )

    return f"""
#### Giải thích AI so sánh tổng quan

> **Cách đọc:** fitness càng thấp càng tốt. Một thuật toán giảm nhiều nước chưa chắc thắng nếu nó tạo thiếu nước,
> dư/vượt ngưỡng hoặc bỏ lỡ mục tiêu phân bổ nước.

{paradox}

**Thuật toán tốt nhất:** {best.name} với điểm **{best.fitness:.4f}**.  
**Dùng ít nước nhất:** {least_water.name} với **{least_water.total_water:.2f} m3**.  
**Chi phí thấp nhất:** {cheapest.name} với **{cheapest.cost:.2f}**.  
**Ít lỗi ràng buộc nhất:** {cleanest.name} với tổng lỗi **{cleanest.issue_total:.2f} m3**.

**Xếp hạng dễ hiểu**

{chr(10).join(rows)}

**Kết luận thực tế**  
Chọn thuật toán đứng đầu khi cần phương án cân bằng. Chọn thuật toán dùng ít nước nhất khi ưu tiên tiết kiệm nước,
nhưng phải chấp nhận kiểm tra thêm thiếu nước. Chọn thuật toán chi phí thấp nhất khi tiền bơm quan trọng hơn việc
đạt sát mục tiêu nước.
"""


def _answer_from_context(question: str, insights: list[AlgoInsight]) -> str:
    q = question.lower()
    ordered = sorted(insights, key=lambda x: x.fitness)
    best = ordered[0]
    least_water = min(ordered, key=lambda x: x.total_water)
    cheapest = min(ordered, key=lambda x: x.cost)
    cleanest = min(ordered, key=lambda x: x.issue_total)

    if any(word in q for word in ["fitness", "điểm", "diem", "score"]):
        return (
            f"Fitness là điểm tổng hợp và càng thấp càng tốt. {best.name} đang tốt nhất với {best.fitness:.4f} "
            f"vì nó cân bằng các khoản phạt tốt hơn: thiếu {best.shortage:.2f} m3, dư/vượt {best.surplus:.2f} m3, "
            f"thiếu mục tiêu {best.under_target:.2f} m3 và chi phí {best.cost:.2f}. Vì vậy đừng chỉ nhìn một chỉ số nước."
        )

    if any(word in q for word in ["nước", "nuoc", "giảm", "giam", "ít", "it"]):
        if least_water.name == best.name:
            return (
                f"{least_water.name} hiện vừa dùng ít nước nhất ({least_water.total_water:.2f} m3) vừa có điểm tốt nhất. "
                "Điều đó nghĩa là việc giảm nước không làm tăng lỗi thiếu nước hoặc lỗi mục tiêu quá mạnh."
            )
        return (
            f"{least_water.name} dùng ít nước nhất ({least_water.total_water:.2f} m3), nhưng {best.name} có điểm tốt hơn "
            f"vì fitness còn phạt thiếu nước, dư/vượt ngưỡng, thiếu mục tiêu và chi phí. Nói dễ hiểu: giảm nước là tốt, "
            "nhưng giảm quá tay hoặc giảm sai thửa sẽ làm điểm xấu đi."
        )

    if any(word in q for word in ["chi phí", "chiphi", "cost", "bơm", "bom", "rẻ", "re"]):
        return (
            f"Nếu ưu tiên tiền bơm, hãy nhìn {cheapest.name}: chi phí thấp nhất là {cheapest.cost:.2f}, "
            f"trung bình {cheapest.avg_cost:.2f}/m3. Tuy nhiên vẫn nên so với điểm fitness tổng, vì rẻ hơn "
            "không luôn đồng nghĩa với ít thiếu nước hơn."
        )

    if any(word in q for word in ["thiếu", "thieu", "dư", "du", "vượt", "vuot", "ràng buộc", "rang buoc"]):
        return (
            f"Ít lỗi ràng buộc nhất hiện là {cleanest.name}: thiếu {cleanest.shortage:.2f} m3, "
            f"dư/vượt {cleanest.surplus:.2f} m3, thiếu mục tiêu {cleanest.under_target:.2f} m3. "
            "Các lỗi này quan trọng vì cây thiếu nước hoặc nước vượt ngưỡng đều bị phạt trong fitness."
        )

    return (
        f"Tóm tắt nhanh: {best.name} đang có điểm tốt nhất ({best.fitness:.4f}), "
        f"{least_water.name} dùng ít nước nhất ({least_water.total_water:.2f} m3), "
        f"và {cheapest.name} có chi phí thấp nhất ({cheapest.cost:.2f}). "
        "Nếu bạn hỏi cụ thể về điểm, nước, chi phí hoặc thiếu/dư nước, tôi sẽ giải thích sâu theo đúng số liệu đó."
    )


def _render_chat_history(history: list[dict]) -> None:
    if not history:
        return
    for msg in history:
        if msg["role"] == "user":
            st.markdown(f"**Bạn hỏi:** {msg['content']}")
        else:
            st.markdown(f"**AI giải thích:** {msg['content']}")


def _chat_box(key: str, insights: list[AlgoInsight], placeholder: str) -> None:
    history_key = f"ai_chat_history_{key}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    st.markdown("**Hỏi thêm nếu chưa hiểu**")
    _render_chat_history(st.session_state[history_key])

    with st.form(key=f"ai_chat_form_{key}", clear_on_submit=True):
        user_question = st.text_input(placeholder, key=f"ai_chat_text_{key}")
        submitted = st.form_submit_button("Gửi câu hỏi")

    if submitted and user_question.strip():
        st.session_state[history_key].append({"role": "user", "content": user_question.strip()})
        answer = _answer_from_context(user_question.strip(), insights)
        st.session_state[history_key].append({"role": "assistant", "content": answer})
        st.rerun()


def _toggle_explanation(key: str, label: str) -> bool:
    open_key = f"ai_explain_open_{key}"
    if open_key not in st.session_state:
        st.session_state[open_key] = False

    button_label = "Ẩn giải thích" if st.session_state[open_key] else label
    if st.button(button_label, key=f"ai_explain_btn_{key}"):
        st.session_state[open_key] = not st.session_state[open_key]
        st.rerun()
    return bool(st.session_state[open_key])


def render_run_ai_section(results: dict, current_algo: str | None = None) -> None:
    if not results:
        return
    _style_once()
    algo = current_algo if current_algo in results else min(results, key=lambda a: results[a][0]["best_fitness"])
    insights = _best_runs(results, [algo])
    if not insights:
        return

    st.divider()
    st.header("AI giải thích kết quả")
    st.caption("Bấm mở khi bạn muốn xem giải thích chi tiết từ số liệu thuật toán vừa chạy.")
    section_key = f"run_{algo}_{_stable_hash(insights[0])}"
    if _toggle_explanation(section_key, "Xem giải thích kết quả vừa chạy"):
        st.markdown(_run_explanation(insights[0]))
        _chat_box(
            key=section_key,
            insights=insights,
            placeholder="Ví dụ: Tại sao thuật toán này giảm nước nhiều nhưng điểm chưa tốt?",
        )


def render_comparison_ai_section(results: dict, ranked: list[str]) -> None:
    if not results:
        return
    _style_once()
    insights = _best_runs(results, ranked)
    if not insights:
        return

    st.divider()
    st.header("6. AI giải thích và hỏi đáp")
    st.caption("Bấm mở khi bạn muốn xem giải thích so sánh; nếu vẫn chưa rõ thì hỏi thêm bên dưới.")
    section_key = f"compare_{_stable_hash(*insights)}"
    if _toggle_explanation(section_key, "Xem giải thích so sánh các thuật toán"):
        st.markdown(_comparison_explanation(insights))
        _chat_box(
            key=section_key,
            insights=insights,
            placeholder="Ví dụ: Tại sao SA giảm nhiều nước hơn GA nhưng điểm lại cao hơn?",
        )


def _stable_hash(*items: AlgoInsight) -> str:
    raw = "|".join(
        f"{i.name}:{i.fitness:.4f}:{i.total_water:.2f}:{i.cost:.2f}:{i.shortage:.2f}:{i.surplus:.2f}"
        for i in items
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
