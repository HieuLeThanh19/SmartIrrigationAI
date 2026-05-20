import streamlit as st
from core import data as data_module
from core import config as cfg


def init_session():
    defaults = {
        "fields_data":     None,
        "run_results":     None,
        "selected_algos":  ["GA", "SA", "PSO", "Hybrid"],
        "W_total":         cfg.W_TOTAL,
        "n_runs":          cfg.N_RUNS,
        "is_ran":          False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.get("n_runs_default_version") != 3:
        st.session_state["n_runs"] = cfg.N_RUNS
        st.session_state["n_runs_default_version"] = 3

    if st.session_state["fields_data"] is None:
        st.session_state["fields_data"] = {
            "demand_min":  data_module.get_demand_min(),
            "demand_max":  data_module.get_demand_max(),
            "prices":      data_module.get_prices(),
            "W_total":     cfg.W_TOTAL,
            "field_names": data_module.get_field_names(),
            "crop_names":  data_module.get_crop_names(),
        }
        st.session_state["fields_data_default_version"] = 2
    elif st.session_state.get("fields_data_default_version") != 2:
        st.session_state["fields_data"]["W_total"] = cfg.W_TOTAL
        st.session_state["fields_data_default_version"] = 2
    sync_budget_state()


def sync_budget_state():
    """Đồng bộ W_total toàn cục theo fields_data để các trang luôn nhất quán."""
    fields_data = st.session_state.get("fields_data")
    if fields_data and "W_total" in fields_data:
        st.session_state["W_total"] = float(fields_data["W_total"])


def save_results(results):
    st.session_state["run_results"] = results
    st.session_state["is_ran"]      = True


def get_results():
    return st.session_state.get("run_results", None)


def clear_results():
    st.session_state["run_results"] = None
    st.session_state["is_ran"]      = False
