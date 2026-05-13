# core/data.py — Dữ liệu 10 thửa ruộng cho bài toán tưới tiêu

import numpy as np

# ── Tên thửa ruộng và loại cây trồng ──────────────────
FIELD_NAMES = [
    "Thửa 1", "Thửa 2", "Thửa 3", "Thửa 4", "Thửa 5",
    "Thửa 6", "Thửa 7", "Thửa 8", "Thửa 9", "Thửa 10",
]

CROP_NAMES = [
    "Lúa",    "Lúa",    "Ngô",    "Ngô",    "Rau cải",
    "Rau cải","Đậu",    "Đậu",    "Mía",    "Mía",
]

# ── Nhu cầu nước tối thiểu (m³/ngày) ─────────────────
DEMAND_MIN = np.array([20, 18, 15, 16, 10, 10, 12, 12, 25, 22], dtype=float)

# ── Nhu cầu nước tối đa (m³/ngày) ────────────────────
DEMAND_MAX = np.array([40, 38, 32, 30, 22, 20, 25, 24, 45, 40], dtype=float)

# ── Giá bơm nước (đ/m³) ──────────────────────────────
PRICES = np.array([1.2, 1.1, 1.5, 1.4, 1.8, 1.7, 1.6, 1.5, 1.0, 1.1], dtype=float)


# ── Hàm truy xuất ────────────────────────────────────
def get_demand_min() -> np.ndarray:
    return DEMAND_MIN.copy()

def get_demand_max() -> np.ndarray:
    return DEMAND_MAX.copy()

def get_prices() -> np.ndarray:
    return PRICES.copy()

def get_field_names() -> list:
    return FIELD_NAMES.copy()

def get_crop_names() -> list:
    return CROP_NAMES.copy()
