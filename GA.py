# ==========================================================
# Smart Home Energy Scheduling using Genetic Algorithm
# ==========================================================

import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
st.set_page_config(page_title="GA Energy Scheduler", layout="wide")
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

@st.cache_data
def load_data():
    # Ensure project_benchmark_data.csv is in the same folder
    try:
        return pd.read_csv("project_benchmark_data.csv")
    except FileNotFoundError:
        # Fallback dummy data if file is missing for testing
        data = {
            "Appliance": ["Washing Machine", "Dishwasher", "EV Charger", "Fridge", "Lights"],
            "Avg_kWh": [2.0, 1.5, 3.5, 0.5, 0.8],
            "Duration": [2, 2, 4, 24, 6],
            "Start_Window": [8, 9, 18, 0, 18],
            "End_Window": [22, 22, 23, 0, 18],
            "Shiftable": [1, 1, 1, 0, 0]
        }
        return pd.DataFrame(data)

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
    # Peak hours: 08:00 to 22:00
    return TARIFF_PEAK if 8 <= hour < 22 else TARIFF_OFFPEAK

# ----------------------------------------------------------
# 3. Fixed Cost Calculation
# ----------------------------------------------------------
fixed_cost = 0
for _, row in non_shiftable.iterrows():
    # For non-shiftables, we assume they run at their Start_Window
    # Fridge runs 24h, so we calculate total 24h cost
    if row["Duration"] == 24:
        for h in range(24):
            fixed_cost += row["Avg_kWh"] * get_tariff(h)
    else:
        for h in range(row["Start_Window"], min(row["Start_Window"]+row["Duration"], 24)):
            fixed_cost += row["Avg_kWh"] * get_tariff(h)

# ----------------------------------------------------------
# 4. Preferred Start Time (for Discomfort calculation)
# ----------------------------------------------------------
# This adds a "User Preference" start time within their allowed window
if 'preferred_times' not in st.session_state:
    st.session_state.preferred_times = [random.randint(row["Start_Window"], row["End_Window"]) for _, row in shiftable.iterrows()]

shiftable["Preferred_Time"] = st.session_state.preferred_times

# ----------------------------------------------------------
# 5. GA Parameters (Sidebar)
# ----------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")
POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 40)
GENERATIONS = st.sidebar.slider("Generations", 50, 500, 150)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.1, 0.95, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.15)
ALPHA = st.sidebar.slider("Discomfort Weight (α)", 0.1, 2.0, 0.5)
MAX_POWER = st.sidebar.number_input("Max Power Limit (kW)", 1.0, 20.0, 5.0)

# ----------------------------------------------------------
# 6. Genetic Algorithm Functions
# ----------------------------------------------------------
def create_individual():
    return [random.randint(row["Start_Window"], row["End_Window"]) for _, row in shiftable.iterrows()]

def fitness(individual):
    shiftable_cost = 0
    discomfort = 0
    penalty = 0
    hourly_power = [0.0]*24

    # Add Non-shiftable load
    for _, row in non_shiftable.iterrows():
        duration = int(row["Duration"])
        start = int(row["Start_Window"])
        for h in range(start, min(start + duration, 24)):
            hourly_power[h] += row["Avg_kWh"]

    # Add Shiftable load from GA individual
    for i, start in enumerate(individual):
        row = shiftable.iloc[i]
        duration = int(row["Duration"])
        
        # Cost and Discomfort
        shiftable_cost += row["Avg_kWh"] * duration * get_tariff(start)
        discomfort += abs(start - row["Preferred_Time"])
        
        # Load profile
        for h in range(start, min(start + duration, 24)):
            hourly_power[h] += row["Avg_kWh"]

    # Peak power penalty
    for power in hourly_power:
        if power > MAX_POWER:
            penalty += (power - MAX_POWER) * 500  # Strict penalty

    return shiftable_cost + (ALPHA * discomfort) + penalty

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
# 7. Execution and Visualization
# ----------------------------------------------------------
if st.button("Run Optimization"):
    # Initialize Population
    population = [create_individual() for _ in range(POP_SIZE)]
    best_history = []

    # Progress bar for UX
    progress_bar = st.progress(0)

    # GA Loop
    for gen in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE // 2):
            p1, p2 = selection(population), selection(population)
            c1, c2 = crossover(p1, p2)
            new_population.extend([mutate(c1), mutate(c2)])
        population = new_population
        best_ind = min(population, key=fitness)
        best_history.append(fitness(best_ind))
        progress_bar.progress((gen + 1) / GENERATIONS)

    best_solution = min(population, key=fitness)

    # --- Results Display ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimized Schedule")
        res_list = []
        opt_shift_cost = 0
        for i, start in enumerate(best_solution):
            row = shiftable.iloc[i]
            opt_shift_cost += row["Avg_kWh"] * row["Duration"] * get_tariff(start)
            res_list.append({
                "Appliance": row["Appliance"],
                "Preferred": f"{row['Preferred_Time']}:00",
                "Scheduled": f"{start}:00",
                "End": f"{min(start+row['Duration'], 24)}:00"
            })
        st.table(pd.DataFrame(res_list))

    with col2:
        st.subheader("Cost Analysis")
        baseline_shift_cost = sum(row["Avg_kWh"] * row["Duration"] * get_tariff(row["Start_Window"]) for _, row in shiftable.iterrows())
        total_base = baseline_shift_cost + fixed_cost
        total_opt = opt_shift_cost + fixed_cost
        
        st.metric("Baseline Cost", f"RM {total_base:.2f}")
        st.metric("Optimized Cost", f"RM {total_opt:.2f}", f"-RM {total_base-total_opt:.2f}")

    # --- POWER CHART ---
    st.subheader("24-Hour Load Profile vs Tariff")
    
    hourly_power = [0.0]*24
    # Calculate final power for plotting
    for _, row in non_shiftable.iterrows():
        for h in range(row["Start_Window"], min(row["Start_Window"]+int(row["Duration"]), 24)):
            hourly_power[h] += row["Avg_kWh"]
    for i, start in enumerate(best_solution):
        row = shiftable.iloc[i]
        for h in range(start, min(start+int(row["Duration"]), 24)):
            hourly_power[h] += row["Avg_kWh"]

    tariffs = [get_tariff(h) for h in range(24)]
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    colors = ['red' if p > MAX_POWER else 'skyblue' for p in hourly_power]
    
    # Left axis (Power)
    ax1.bar(range(24), hourly_power, color=colors, alpha=0.8, label="Power (kW)")
    ax1.axhline(MAX_POWER, color='red', linestyle='--', label="Limit")
    ax1.set_ylabel("Load (kW)")
    ax1.set_xlabel("Hour of Day")
    ax1.set_xticks(range(24))

    # Right axis (Tariff)
    ax2 = ax1.twinx()
    ax2.step(range(24), tariffs, where='post', color='orange', linewidth=2, label="Tariff RM")
    ax2.set_ylabel("Price (RM/kWh)", color='orange')
    
    plt.title("How the GA shifted loads to Off-Peak (Orange) while staying under Limit (Red)")
    st.pyplot(fig)

    # --- CONVERGENCE CHART ---
    st.subheader("GA Convergence (Fitness over Time)")
    st.line_chart(best_history)
