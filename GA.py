# ==========================================================
# Smart Home Energy Scheduling using Genetic Algorithm
# ==========================================================

import streamlit as st
import pandas as pd
import random

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

@st.cache_data
def load_data():
    return pd.read_csv("project_benchmark_data.csv")

df = load_data()
st.subheader("Appliance Dataset")
st.dataframe(df)

shiftable = df[df["Shiftable"] == 1].reset_index(drop=True)
non_shiftable = df[df["Shiftable"] == 0].reset_index(drop=True)

# ----------------------------------------------------------
# 2. Malaysia Peak / Off-Peak Tariff
# ----------------------------------------------------------
TARIFF_PEAK = 0.50
TARIFF_OFFPEAK = 0.30

def get_tariff(hour):
    return TARIFF_PEAK if 8 <= hour < 22 else TARIFF_OFFPEAK

# ----------------------------------------------------------
# 3. Fixed Cost
# ----------------------------------------------------------
fixed_cost = 0
for _, row in non_shiftable.iterrows():
    tariff = get_tariff(row["Start_Window"])
    fixed_cost += row["Avg_kWh"] * row["Duration"] * tariff

# ----------------------------------------------------------
# 4. Preferred Time
# ----------------------------------------------------------
shiftable["Preferred_Time"] = shiftable.apply(
    lambda r: random.randint(r["Start_Window"], r["End_Window"]),
    axis=1
)

# ----------------------------------------------------------
# 5. GA Parameters
# ----------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 40)
GENERATIONS = st.sidebar.slider("Generations", 50, 300, 150)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.1, 0.95, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.15)
ALPHA = st.sidebar.slider("Discomfort Weight (α)", 0.1, 2.0, 0.5)

MAX_POWER = 5.0  # kW

# ----------------------------------------------------------
# 6. GA Functions
# ----------------------------------------------------------
def create_individual():
    return [random.randint(row["Start_Window"], row["End_Window"]) for _, row in shiftable.iterrows()]

def fitness(individual):
    shiftable_cost = 0
    discomfort = 0
    penalty = 0
    hourly_power = [0.0]*24

    for _, row in non_shiftable.iterrows():
        for h in range(row["Start_Window"], min(row["Start_Window"]+row["Duration"], 24)):
            hourly_power[h] += row["Avg_kWh"]

    for i, start in enumerate(individual):
        row = shiftable.iloc[i]
        if not (row["Start_Window"] <= start <= row["End_Window"]):
            penalty += 200

        tariff = get_tariff(start)
        shiftable_cost += row["Avg_kWh"]*row["Duration"]*tariff
        discomfort += abs(start - row["Preferred_Time"])

        for h in range(start, min(start+row["Duration"],24)):
            hourly_power[h] += row["Avg_kWh"]

    for power in hourly_power:
        if power > MAX_POWER:
            penalty += (power - MAX_POWER)*300

    return shiftable_cost + ALPHA*discomfort + penalty

def selection(pop):
    return min(random.sample(pop, 3), key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(p1)-1)
        return p1[:point]+p2[point:], p2[:point]+p1[point:]
    return p1, p2

def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            row = shiftable.iloc[i]
            ind[i] = random.randint(row["Start_Window"], row["End_Window"])
    return ind

# ----------------------------------------------------------
# 7. Run Optimization
# ----------------------------------------------------------
if st.button("Run Optimization"):

    # Initialize population
    population = [create_individual() for _ in range(POP_SIZE)]
    best_history = []

    # GA iterations
    for _ in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE//2):
            p1 = selection(population)
            p2 = selection(population)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            new_population.append(mutate(c2))
        population = new_population
        best_history.append(fitness(min(population, key=fitness)))

    best_solution = min(population, key=fitness)

    # ------------------------------------------------------
    # Optimized Appliance Schedule
    # ------------------------------------------------------
    st.subheader("Optimized Appliance Schedule")
    result = []
    optimized_shiftable_cost = 0
    for i, start in enumerate(best_solution):
        row = shiftable.iloc[i]
        end_time = min(start+row["Duration"], 24)
        optimized_shiftable_cost += row["Avg_kWh"]*row["Duration"]*get_tariff(start)
        result.append({
            "Appliance": row["Appliance"],
            "Preferred Time": row["Preferred_Time"],
            "Scheduled Start": start,
            "Scheduled End": end_time,
            "Duration (h)": row["Duration"],
            "Avg Power (kW)": row["Avg_kWh"]
        })
    st.dataframe(pd.DataFrame(result))

    total_optimized_cost = fixed_cost + optimized_shiftable_cost

    # ------------------------------------------------------
    # Cost Comparison
    # ------------------------------------------------------
    baseline_cost = fixed_cost
    for _, row in shiftable.iterrows():
        baseline_cost += row["Avg_kWh"]*row["Duration"]*get_tariff(row["Start_Window"])

    st.metric("Baseline Cost (RM)", f"{baseline_cost:.2f}")
    st.metric("Optimized Cost (RM)", f"{total_optimized_cost:.2f}")
    st.metric("Cost Savings (RM)", f"{baseline_cost-total_optimized_cost:.2f}")

    # ------------------------------------------------------
    # GA Convergence Plot
    # ------------------------------------------------------
    st.subheader("GA Convergence Curve")
    ga_series = pd.Series(best_history)
    st.line_chart(ga_series)

    # ------------------------------------------------------
    # Hourly Power Consumption Plot
    # ------------------------------------------------------
    st.subheader("Hourly Power Consumption (kW)")

    hourly_power = [0.0]*24
    for _, row in non_shiftable.iterrows():
        for h in range(row["Start_Window"], min(row["Start_Window"]+row["Duration"], 24)):
            hourly_power[h] += row["Avg_kWh"]

    for i, start in enumerate(best_solution):
        row = shiftable.iloc[i]
        for h in range(start, min(start+row["Duration"], 24)):
            hourly_power[h] += row["Avg_kWh"]

    # Convert to Pandas Series for Streamlit
    hourly_series = pd.Series(hourly_power, index=[f"{h}:00" for h in range(24)])
    st.bar_chart(hourly_series)

    # Optional: show peak power reference line
    st.markdown(f"**Red line reference**: Peak power limit = {MAX_POWER} kW")
    st.line_chart(pd.Series([MAX_POWER]*24, index=hourly_series.index))
