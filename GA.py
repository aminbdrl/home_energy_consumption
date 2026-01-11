import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Page Setup
# ==========================================================
st.set_page_config(page_title="Smart Home Energy Scheduling", layout="wide")
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

# ==========================================================
# Session State (FIXES CRASH)
# ==========================================================
if "best_solution" not in st.session_state:
    st.session_state.best_solution = None

# ==========================================================
# 1. Load & Preprocess Dataset
# ==========================================================
@st.cache_data
def load_data():
    data = pd.read_csv("project_benchmark_data.csv")

    data["Shiftable"] = data["Is_Shiftable"].astype(int)
    data["Avg_kWh"] = data["Avg_Power_kW"]
    data["Duration"] = data["Duration_Hours"]
    data["Preferred_Time"] = data["Preferred_Start_Hour"]

    # ±3 hour flexibility for shiftable appliances
    WINDOW = 3
    data["Start_Window"] = data.apply(
        lambda r: max(0, r["Preferred_Time"] - WINDOW) if r["Shiftable"] else r["Preferred_Time"],
        axis=1
    )
    data["End_Window"] = data.apply(
        lambda r: min(23, r["Preferred_Time"] + WINDOW) if r["Shiftable"] else r["Preferred_Time"],
        axis=1
    )

    return data

df = load_data()

st.subheader("Appliance Dataset")
st.dataframe(df[[
    "Appliance", "Avg_kWh", "Preferred_Time",
    "Duration", "Shiftable"
]])

shiftable = df[df["Shiftable"] == 1].reset_index(drop=True)
non_shiftable = df[df["Shiftable"] == 0].reset_index(drop=True)

# ==========================================================
# 2. Malaysia Time-of-Use Tariff
# ==========================================================
TARIFF_PEAK = 0.50      # 8 AM – 10 PM
TARIFF_OFFPEAK = 0.30  # 10 PM – 8 AM

def get_tariff(hour):
    return TARIFF_PEAK if 8 <= hour < 22 else TARIFF_OFFPEAK

# ==========================================================
# 3. Fixed Cost (Non-shiftable)
# ==========================================================
def calculate_fixed_cost():
    cost = 0
    for _, row in non_shiftable.iterrows():
        for h in range(row["Preferred_Time"], row["Preferred_Time"] + row["Duration"]):
            cost += row["Avg_kWh"] * get_tariff(h % 24)
    return cost

fixed_cost = calculate_fixed_cost()

# ==========================================================
# 4. GA Parameters
# ==========================================================
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 40)
GENERATIONS = st.sidebar.slider("Generations", 50, 400, 200)
CROS
