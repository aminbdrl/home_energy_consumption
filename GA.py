import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="Smart Home Energy Scheduling", layout="wide")
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

# ----------------------------------------------------------
# 1. Load & Preprocess Dataset
# ----------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("project_benchmark_data.csv")

    data["Shiftable"] = data["Is_Shiftable"].astype(int)
    data["Avg_kWh"] = data["Avg_Power_kW"]
    data["Duration"] = data["Duration_Hours"]
    data["Preferred_Time"] = data["Preferred_Start_Hour"]

    # Realistic flexibility: ±3 hours
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

# ----------------------------------------------------------
# 2. Malaysia Time-of-Use Tariff
# ----------------------------------------------------------
TARIFF_PEAK = 0.50
TARIFF_OFFPEAK = 0.30

def get_tariff(hour):
    return TARIFF_PEAK if 8 <= hour < 22 else TARIFF_OFFPEAK

# ----------------------------------------------------------
# 3. Fixed Cost (Non-shiftable)
# ----------------------------------------------------------
def calculate_fixed_cost():
    cost = 0
    for _, row in non_shiftable.iterrows():
        for h in range(row["Preferred_Time"], row["Preferred_Time"] + row["Duration"]):
            cost += row["Avg_kWh"] * get_tariff(h % 24)
    return cost

fixed_cost = calculate_fixed_cost()

# ----------------------------------------------------------
# 4. GA Parameters
# ----------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 40)
GENERATIONS = st.sidebar.slider("Generations", 50, 400, 200)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.1, 0.95, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.15)
ALPHA = st.sidebar.slider("Discomfort Weight (α)", 0.0, 5.0, 0.5)

MAX_POWER = 5.0  # kW

# ----------------------------------------------------------
# 5. GA Core Functions
# ----------------------------------------------------------
def create_individual():
    return [
        random.randint(row["Start_Window"], row["End_Window"])
        for _, row in shiftable.iterrows()
    ]

def fitness(individual):
    shiftable_cost = 0
    discomfort = 0
    penalty = 0
    hourly_power = [0.0] * 24

    # Non-shiftable load
    for _, row in non_shiftable.iterrows():
        for h in range(row["Preferred_Time"], row["Preferred_Time"] + row["Duration"]):
            hourly_power[h % 24] += row["Avg_kWh"]

    # Shiftable appliances
    for i, start in enumerate(individual):
        row = shiftable.iloc[i]

        discomfort += abs(start - row["Preferred_Time"])

        for h in range(start, start + row["Duration"]):
            hour = h % 24
            hourly_power[hour] += row["Avg_kWh"]
            shiftable_cost += row["Avg_kWh"] * get_tariff(hour)

    # Peak power constraint
    for p in hourly_power:
        if p > MAX_POWER:
            penalty += (p - MAX_POWER) * 300

    return shiftable_cost + ALPHA * discomfort + penalty

def calculate_total_cost(solution):
    cost = fixed_cost
    for i, start in enumerate(solution):
        row = shiftable.iloc[i]
        for h in range(start, start + row["Duration"]):
            cost += row["Avg_kWh"] * get_tariff(h % 24)
    return cost

def selection(pop):
    return min(random.sample(pop, 3), key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE and len(p1) > 1:
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1, p2

def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            row = shiftable.iloc[i]
            ind[i] = random.randint(row["Start_Window"], row["End_Window"])
    return ind

# ----------------------------------------------------------
# 6. Run Optimization
# ----------------------------------------------------------
if st.button("Run Optimization"):

    population = [create_individual() for _ in range(POP_SIZE)]
    best_history = []
    progress = st.progress(0)

    for g in range(GENERATIONS):
        new_pop = []
        for _ in range(POP_SIZE // 2):
            p1, p2 = selection(population), selection(population)
            c1, c2 = crossover(p1, p2)
            new_pop.extend([mutate(c1), mutate(c2)])
        population = new_pop
        best_history.append(fitness(min(population, key=fitness)))
        progress.progress((g + 1) / GENERATIONS)

    best_solution = min(population, key=fitness)

    # ------------------------------------------------------
    # 7. Results Table
    # ------------------------------------------------------
    st.subheader("Optimized Appliance Schedule")

    rows = []

    for _, r in non_shiftable.iterrows():
        rows.append({
            "Appliance": r["Appliance"],
            "Type": "Non-Shiftable",
            "Preferred": r["Preferred_Time"],
            "Scheduled": r["Preferred_Time"],
            "Duration": r["Duration"],
            "Power (kW)": r["Avg_kWh"]
        })

    for i, start in enumerate(best_solution):
        r = shiftable.iloc[i]
        rows.append({
            "Appliance": r["Appliance"],
            "Type": "Shiftable",
            "Preferred": r["Preferred_Time"],
            "Scheduled": start,
            "Duration": r["Duration"],
            "Power (kW)": r["Avg_kWh"]
        })

    st.dataframe(pd.DataFrame(rows))

    # ------------------------------------------------------
    # 8. Cost Metrics
    # ------------------------------------------------------
    baseline_solution = [r["Preferred_Time"] for _, r in shiftable.iterrows()]

    baseline_cost = calculate_total_cost(baseline_solution)
    optimized_cost = calculate_total_cost(best_solution)

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Cost (RM)", f"{baseline_cost:.2f}")
    c2.metric("Optimized Cost (RM)", f"{optimized_cost:.2f}")
    c3.metric("Savings (RM)", f"{baseline_cost - optimized_cost:.2f}")

# ------------------------------------------------------
# 9. Hourly Power Consumption Plot (Constraint Validation)
# ------------------------------------------------------
st.subheader("Hourly Power Consumption (kW)")

# Calculate hourly power for optimized schedule
hourly_power = [0.0] * 24

# Non-shiftable appliances
for _, row in non_shiftable.iterrows():
    for h in range(row["Preferred_Time"], row["Preferred_Time"] + row["Duration"]):
        hourly_power[h % 24] += row["Avg_kWh"]

# Shiftable appliances (optimized)
for i, start in enumerate(best_solution):
    row = shiftable.iloc[i]
    for h in range(start, start + row["Duration"]):
        hourly_power[h % 24] += row["Avg_kWh"]

# Plot
fig, ax = plt.subplots()
ax.plot(range(24), hourly_power, marker='o', label="Total Power (kW)")
ax.axhline(MAX_POWER, linestyle="--", label="5.0 kW Limit")

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power Consumption (kW)")
ax.set_xticks(range(24))
ax.legend()
ax.grid(True)

st.pyplot(fig)

