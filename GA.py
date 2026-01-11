import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# ==========================================================
# Page Setup
# ==========================================================
st.set_page_config(page_title="Smart Home Energy Scheduling", layout="wide")
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

# ==========================================================
# Session State
# ==========================================================
if "best_solution" not in st.session_state:
    st.session_state.best_solution = None

# ==========================================================
# Load Dataset
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("project_benchmark_data.csv")

    df["Shiftable"] = df["Is_Shiftable"].astype(int)
    df["Avg_kWh"] = df["Avg_Power_kW"]
    df["Duration"] = df["Duration_Hours"]
    df["Preferred_Time"] = df["Preferred_Start_Hour"]

    WINDOW = 3
    df["Start_Window"] = df.apply(
        lambda r: max(0, r["Preferred_Time"] - WINDOW) if r["Shiftable"] else r["Preferred_Time"], axis=1
    )
    df["End_Window"] = df.apply(
        lambda r: min(23, r["Preferred_Time"] + WINDOW) if r["Shiftable"] else r["Preferred_Time"], axis=1
    )

    return df

df = load_data()
shiftable = df[df.Shiftable == 1].reset_index(drop=True)
non_shiftable = df[df.Shiftable == 0].reset_index(drop=True)

# ==========================================================
# Tariff (Malaysia TOU)
# ==========================================================
PEAK = 0.50
OFFPEAK = 0.30

def tariff(hour):
    return PEAK if 8 <= hour < 22 else OFFPEAK

MAX_POWER = 5.0

# ==========================================================
# Fixed Cost
# ==========================================================
def fixed_cost():
    cost = 0
    for _, r in non_shiftable.iterrows():
        for h in range(r.Preferred_Time, r.Preferred_Time + r.Duration):
            cost += r.Avg_kWh * tariff(h % 24)
    return cost

BASE_FIXED_COST = fixed_cost()

# ==========================================================
# GA Parameters
# ==========================================================
st.sidebar.header("GA Parameters")

POP_SIZE = st.sidebar.slider("Population", 20, 100, 50)
GENS = st.sidebar.slider("Generations", 100, 500, 250)
MUT_RATE = st.sidebar.slider("Mutation Rate", 0.05, 0.5, 0.2)
ALPHA = st.sidebar.slider("Discomfort Weight", 0.0, 5.0, 0.3)

# ==========================================================
# GA Functions
# ==========================================================
def create_individual():
    return [
        random.randint(r.Start_Window, r.End_Window)
        for _, r in shiftable.iterrows()
    ]

def evaluate(ind):
    power = [0]*24
    cost = 0
    discomfort = 0
    penalty = 0

    # Non-shiftable
    for _, r in non_shiftable.iterrows():
        for h in range(r.Preferred_Time, r.Preferred_Time + r.Duration):
            power[h % 24] += r.Avg_kWh

    # Shiftable
    for i, start in enumerate(ind):
        r = shiftable.iloc[i]
        discomfort += abs(start - r.Preferred_Time)

        for h in range(start, start + r.Duration):
            hour = h % 24
            power[hour] += r.Avg_kWh
            cost += r.Avg_kWh * tariff(hour)

    # Power constraint
    for p in power:
        if p > MAX_POWER:
            penalty += (p - MAX_POWER) * 1000  # VERY strong

    return cost + ALPHA * discomfort + penalty

def tournament(pop):
    return min(random.sample(pop, 3), key=evaluate)

def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUT_RATE:
            r = shiftable.iloc[i]
            ind[i] = random.randint(r.Start_Window, r.End_Window)
    return ind

# ==========================================================
# Run Optimization
# ==========================================================
if st.button("Run Optimization"):
    population = [create_individual() for _ in range(POP_SIZE)]
    best = min(population, key=evaluate)

    bar = st.progress(0)
    for g in range(GENS):
        new_pop = [best]  # elitism

        while len(new_pop) < POP_SIZE:
            p = tournament(population)
            child = mutate(p.copy())
            new_pop.append(child)

        population = new_pop
        best = min(population, key=evaluate)
        bar.progress((g + 1) / GENS)

    st.session_state.best_solution = best

# ==========================================================
# Results
# ==========================================================
if st.session_state.best_solution:
    st.subheader("Optimized Schedule")

    rows = []
    for _, r in non_shiftable.iterrows():
        rows.append([r.Appliance, "Non-Shiftable", r.Preferred_Time, r.Preferred_Time])

    for i, start in enumerate(st.session_state.best_solution):
        r = shiftable.iloc[i]
        rows.append([r.Appliance, "Shiftable", r.Preferred_Time, start])

    st.dataframe(pd.DataFrame(
        rows, columns=["Appliance", "Type", "Preferred", "Scheduled"]
    ))

    # Cost comparison
    baseline = [r.Preferred_Time for _, r in shiftable.iterrows()]

    def total_cost(sol):
        c = BASE_FIXED_COST
        for i, s in enumerate(sol):
            r = shiftable.iloc[i]
            for h in range(s, s + r.Duration):
                c += r.Avg_kWh * tariff(h % 24)
        return c

    b_cost = total_cost(baseline)
    o_cost = total_cost(st.session_state.best_solution)

    st.metric("Baseline Cost (RM)", f"{b_cost:.2f}")
    st.metric("Optimized Cost (RM)", f"{o_cost:.2f}")
    st.metric("Savings (RM)", f"{b_cost - o_cost:.2f}")

    # Power Chart
    power = [0]*24
    for _, r in non_shiftable.iterrows():
        for h in range(r.Preferred_Time, r.Preferred_Time + r.Duration):
            power[h % 24] += r.Avg_kWh

    for i, s in enumerate(st.session_state.best_solution):
        r = shiftable.iloc[i]
        for h in range(s, s + r.Duration):
            power[h % 24] += r.Avg_kWh

    fig, ax = plt.subplots()
    ax.plot(range(24), power, marker='o')
    ax.axhline(MAX_POWER, linestyle="--")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Optimized Power Consumption")
    st.pyplot(fig)
