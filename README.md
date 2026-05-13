# SmartIrrigationAI

SmartIrrigationAI là hệ thống tối ưu phân bổ nước tưới tiêu cho 10 thửa ruộng bằng 4 thuật toán:
Genetic Algorithm (GA), Simulated Annealing (SA), Particle Swarm Optimization (PSO), và Hybrid GA+SA.

Ứng dụng có 2 cách chạy:

- Web app Streamlit để nhập dữ liệu, chạy demo từng bước và xem kết quả.
- Terminal runner để chạy thực nghiệm đầy đủ và xuất biểu đồ tổng hợp.

## Cài Đặt

Khuyến nghị dùng Python 3.11+.

```bash
pip install -r requirements.txt
```

## Chạy Ứng Dụng

Chạy giao diện web (chính):

```bash
python -m streamlit run webapp/app.py
```

Chạy thực nghiệm từ terminal:

```bash
python main.py
```

Khi chạy terminal, chương trình so sánh GA, SA, PSO và Hybrid theo cấu hình trong `core/config.py`, sau đó tạo file `dashboard.png`.

## Giao Diện Web

Web app gồm:

- `webapp/app.py`: trang chủ.
- `webapp/pages/01_input.py`: chọn kịch bản, chỉnh dữ liệu thửa ruộng, ngân sách nước và thuật toán cần demo.
- `webapp/pages/02_run.py`: demo 30 bước cho từng thuật toán, có giải thích riêng cho GA, SA, PSO và Hybrid.
- `webapp/pages/03_result.py`: bảng so sánh, biểu đồ hội tụ, phân bổ nước và thống kê.

Streamlit navigation mặc định đã được tắt trong `.streamlit/config.toml`; app dùng menu tiếng Việt riêng ở sidebar.

## Cấu Trúc Project

| Thư mục | Vai trò |
|---|---|
| `core/` | Dữ liệu mặc định, cấu hình, hàm fitness và ràng buộc |
| `algorithms/` | GA, SA, PSO, Hybrid và các operator dùng chung |
| `analysis/` | Runner thực nghiệm, so sánh, thống kê và grid search tham số |
| `visualization/` | Biểu đồ hội tụ, phân bổ nước, boxplot, heatmap và dashboard |
| `webapp/` | Streamlit multipage app |
| `tests/` | Kiểm thử fitness, ràng buộc và operator |
| `Tài liệu/` | Tài liệu thiết kế và mô tả bài toán |

## Kiểm Thử

```bash
python -m pytest tests/ -v
```

## Ghi Chú Cấu Hình

- Tổng lượng nước mặc định: `core/config.py` -> `W_TOTAL` = 260.
- Số bước/lần chạy mặc định: `core/config.py` -> `N_RUNS` = 30; web demo cũng cố định 30 bước.
- Các trọng số fitness nằm trong `ALPHA_PENALTY`, `BETA_WASTE`, `DELTA_UNDERUSE`, `GAMMA_COST`.

Môn: Trí Tuệ Nhân Tạo.
