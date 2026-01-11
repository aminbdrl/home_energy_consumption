# ==========================================================
# Smart Home Energy Scheduling using Genetic Algorithm
# Objectives:
# 1. Minimize Electricity Cost (RM) - Malaysia ToU
# 2. Minimize User Discomfort (Waiting Time)
# Constraints:
# - Non-shiftable appliances remain fixed
# - Shiftable appliances scheduled within time window
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

@st.cache_data
def load_data():
    return pd.read_csv("universal_tasks.csv")

df = load_data()
st.subheader("Appliance Dataset")
st.dataframe(df)

# Separate appliances
shiftable = df[df["Shiftable"] == 1].reset_index(drop=True)
non_shiftable = df[df["Shiftable"] == 0].reset_index(drop=True)

# ----------------------------------------------------------
# 2. Malaysia Time-of-Use Tariff
# ----------------------------------------------------------
TARIFF_PEAK = 0.50      # RM/kWh
TARIFF_OFFPEAK = 0.30  # RM/kWh

def get_tariff(hour):
    if 8 <= hour < 22:
        return TARIFF_PEAK
    return TARIFF_OFFPEAK

# ----------------------------------------------------------
# 3. Fixed Cost (Non-shiftable Appliances)
# ----------------------------------------------------------
fixed_cost = 0
for _, row in non_shiftable.iterrows():
    tariff = get_tariff(row["Start_Window"])
    fixed_cost += row["Avg_kWh"] * row["Duration"] * tariff

# ----------------------------------------------------------
# 4. GA Parameters (Streamlit Controls)
# ----------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 30)
GENERATIONS = st.sidebar.slider("Generations", 20, 300, 100)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.1, 0.95, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
ALPHA = st.sidebar.slider("Discomfort Weight (α)", 0.1, 2.0, 0.5)

# ----------------------------------------------------------
# 5. Genetic Algorithm Functions
# ----------------------------------------------------------
def create_individual():
    """Create a chromosome of start times for shiftable appliances."""
    individual = []
    for _, row in shiftable.iterrows():
        individual.append(random.randint(row["Start_Window"], row["End_Window"]))
    return individual

def fitness(individual):
    """Fitness = Cost + α * Discomfort + Penalty"""
    total_cost = fixed_cost
    discomfort = 0
    penalty = 0

    for i, start_time in enumerate(individual):
        row = shiftable.iloc[i]

        # Time window constraint
        if not (row["Start_Window"] <= start_time <= row["End_Window"]):
            penalty += 100

        tariff = get_tariff(start_time)
        total_cost += row["Avg_kWh"] * row["Duration"] * tariff
        discomfort += abs(start_time - row["Start_Window"])

    return total_cost + ALPHA * discomfort + penalty

def selection(population):
    """Tournament selection"""
    contenders = random.sample(population, 3)
    return min(contenders, key=fitness)

def crossover(parent1, parent2):
    """Single-point crossover"""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        return (
            parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:]
        )
    return parent1, parent2

def mutate(individual):
    """Random mutation within allowed window"""
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            row = shiftable.iloc[i]
            individual[i] = random.randint(row["Start_Window"], row["End_Window"])
    return individual

# ----------------------------------------------------------
# 6. Run Genetic Algorithm
# ----------------------------------------------------------
if st.button("Run Optimization"):

    population = [create_individual() for _ in range(POP_SIZE)]
    best_fitness_history = []

    for _ in range(GENERATIONS):
        new_population = []

        for _ in range(POP_SIZE // 2):
            p1 = selection(population)
            p2 = selection(population)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            new_population.append(mutate(c2))

        population = new_population
        best = min(population, key=fitness)
        best_fitness_history.append(fitness(best))

    best_solution = min(population, key=fitness)
    best_value = fitness(best_solution)

# ----------------------------------------------------------
# 7. Display Results (WITH END TIME)
# ----------------------------------------------------------
st.subheader("Optimized Appliance Schedule")

result = []

for i, start in enumerate(best_solution):
    row = shiftable.iloc[i]
    duration = row["Duration"]
    end_time = start + duration

    # Ensure schedule does not exceed 24 hours
    end_time = min(end_time, 24)

    result.append({
        "Appliance": row["Appliance"],
        "Start Window": row["Start_Window"],
        "End Window": row["End_Window"],
        "Scheduled Start Time (Hour)": start,
        "Scheduled End Time (Hour)": end_time,
        "Duration (Hour)": duration,
        "Avg Power (kWh)": row["Avg_kWh"]
    })

result_df = pd.DataFrame(result)
st.dataframe(result_df)

st.metric("Total Optimized Cost (RM)", f"{best_value:.2f}")


    # ----------------------------------------------------------
    # 8. Convergence Plot
    # ----------------------------------------------------------
    st.subheader("GA Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(best_fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    ax.set_title("Fitness Convergence")
    st.pyplot(fig)

# ----------------------------------------------------------
# End of Application
# ----------------------------------------------------------
