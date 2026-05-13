# webapp/components/result_table.py — Bảng kết quả phân bổ nước
import streamlit as st
import numpy as np
import pandas as pd


def show_allocation_table(solution: np.ndarray, fields_data: dict):
    """Hiển thị bảng phân bổ nước chi tiết với màu trạng thái."""
    demand_min  = np.array(fields_data["demand_min"])
    demand_max  = np.array(fields_data["demand_max"])
    field_names = fields_data.get("field_names", [f"Thửa {i+1}" for i in range(len(solution))])
    crop_names  = fields_data.get("crop_names",  ["—"] * len(solution))

    rows = []
    for i, w in enumerate(solution):
        mn, mx = demand_min[i], demand_max[i]
        if w < mn:
            status = "Thiếu"
        elif w > mx:
            status = "Lãng phí"
        else:
            status = "Tối ưu"
        rows.append({
            "Thửa":       field_names[i],
            "Cây trồng":  crop_names[i],
            "Min (m³)":   round(mn, 1),
            "Phân bổ AI": round(float(w), 2),
            "Max (m³)":   round(mx, 1),
            "Trạng thái": status,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={
                     "Phân bổ AI": st.column_config.ProgressColumn(
                         "Phân bổ AI (m³)",
                         format="%.2f",
                         min_value=0,
                         max_value=float(demand_max.max()),
                     ),
                 })


def show_summary_metrics(solution: np.ndarray, fields_data: dict, W_total: float):
    """Hiển thị các metric tổng hợp."""
    demand_min = np.array(fields_data["demand_min"])
    prices     = np.array(fields_data["prices"])
    total_used = float(np.sum(solution))
    saved      = W_total - total_used
    cost       = float(np.dot(prices, solution))
    shortage   = float(np.sum(np.maximum(0, demand_min - solution)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng nước dùng", f"{total_used:.1f} m³")
    c2.metric("Chi phí bơm",    f"{cost:.1f} đ")
    c3.metric("Tiết kiệm",       f"{saved:.1f} m³")
    c4.metric("Thiếu hụt",       f"{shortage:.1f} m³")
