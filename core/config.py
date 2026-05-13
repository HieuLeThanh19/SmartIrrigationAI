# core/config.py — Tập trung TOÀN BỘ tham số project

# ── Dữ liệu bài toán ──────────────────────────────────
N_FIELDS = 10
W_TOTAL  = 260

# ── Tham số GA ────────────────────────────────────────
GA_POP_SIZE        = 100
GA_MAX_GEN         = 300
GA_CROSSOVER_RATE  = 0.8
GA_MUTATION_RATE   = 0.1
GA_MUTATION_SIGMA  = 3.0
GA_TOURNAMENT_K    = 3
GA_ELITE_RATIO     = 0.05
GA_BLX_ALPHA       = 0.5

# ── Tham số SA ────────────────────────────────────────
SA_T_MAX         = 200.0
SA_T_MIN         = 0.1
SA_ALPHA         = 0.97
SA_ITER_PER_T    = 100
SA_NEIGHBOR_SIGMA = 3.0

# ── Tham số PSO ───────────────────────────────────────
PSO_N_PARTICLES = 50
PSO_MAX_ITER    = 300
PSO_W           = 0.7
PSO_C1          = 1.5
PSO_C2          = 1.5
PSO_V_MAX       = 5.0

# ── Tham số Hybrid GA+SA ──────────────────────────────
HYBRID_GA_GEN   = 200
HYBRID_SA_T_MAX = 50.0

# ── Hàm fitness — trọng số ───────────────────────────
ALPHA_PENALTY = 10
BETA_WASTE    = 3
GAMMA_COST    = 1
DELTA_UNDERUSE = 8

# ── Thực nghiệm ───────────────────────────────────────
N_RUNS = 30
