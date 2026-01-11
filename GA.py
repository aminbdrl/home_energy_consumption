# ==========================================================
# Smart Home Energy Scheduling using Genetic Algorithm
# Objectives:
# 1. Minimize Electricity Cost (RM) - Malaysia Peak vs Off-Peak
# 2. Minimize User Discomfort (Waiting Time)
# Constraints:
# - Non-shiftable appliances fixed
# - Shiftable appliances within time window
# - Peak Power Limit = 5.0 kW
# ==========================================================

import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load and Preprocess Dataset
# ----------------------------------------------------------
st.set_page_config(page_title="Smart Home Energy Scheduling", layout="wide")
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

@st.cache_data
def load_data():
    data = pd.read_csv("project_benchmark_data.csv")
    
    # Preprocessing to match GA logic variables
    data['Shiftable'] = data['Is_Shiftable'].astype(int)
    data['Avg_kWh'] = data['Avg_Power_kW']
    data['Duration'] = data['Duration_Hours']
    data['Preferred_Time'] = data['Preferred_Start_Hour']
    
    # Define search windows for GA
    # Shiftable can start anytime (0-23), Non-shiftable are fixed at preferred hour
    data['Start_Window'] = data.apply(lambda r: 0 if r['Is_Shiftable'] else r['Preferred_Start_Hour'], axis=1)
    data['End_Window'] = data.apply(lambda r: 23 if r['Is_Shiftable'] else r['Preferred_Start_Hour'], axis=1)
    
    return data

df = load_data()
st.subheader("Appliance Dataset")
st.dataframe(df[['Appliance', 'Avg_Power_kW', 'Preferred_Start_Hour', 'Duration_Hours', 'Is_Shiftable']])

# Separate appliances
shiftable = df[df["Shiftable"] == 1].reset_index(drop=True)
non_shiftable = df[df["Shiftable"] == 0].reset_index(drop=True)

# ----------------------------------------------------------
# 2. Malaysia Peak / Off-Peak Tariff
# ----------------------------------------------------------
TARIFF_PEAK = 0.50      # RM/kWh (8 AM - 10 PM)
TARIFF_OFFPEAK = 0.30   # RM/kWh

def get_tariff(hour):
    return TARIFF_PEAK if 8 <= hour < 22 else TARIFF_OFFPEAK

# ----------------------------------------------------------
# 3. Fixed Cost (Non-Shiftable Appliances)
# ----------------------------------------------------------
fixed_cost = 0
for _, row in non_shiftable.iterrows():
    # Calculate cost across the duration for non-shiftable
    for h in range(row["Preferred_Start_Hour"], row["Preferred_Start_Hour"] + row["Duration"]):
        hour_val = h % 24
        fixed_cost += row["Avg_kWh"] * get_tariff(hour_val)

# ----------------------------------------------------------
# 4. GA Parameters
# ----------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 40)
GENERATIONS = st.sidebar.slider("Generations", 50, 500, 200)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.1, 0.95, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.15)
ALPHA = st.sidebar.slider("Discomfort Weight (α)", 0.0, 5.0, 0.5)

MAX_POWER = 5.0  # kW limit

# ----------------------------------------------------------
# 5. Genetic Algorithm Functions
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

    # Initialize hourly power tracker
    hourly_power = [0.0] * 24

    # 1. Account for Non-shiftable power load
    for _, row in non_shiftable.iterrows():
        for h in range(row["Preferred_Start_Hour"], row["Preferred_Start_Hour"] + row["Duration"]):
            hourly_power[h % 24] += row["Avg_kWh"]

    # 2. Account for Shiftable appliances
    for i, start in enumerate(individual):
        row = shiftable.iloc[i]
        
        # Discomfort: difference from preferred start time
        discomfort += abs(start - row["Preferred_Time"])

        for h in range(start, start + row["Duration"]):
            hour_val = h % 24
            hourly_power[hour_val] += row["Avg_kWh"]
            shiftable_cost += row["Avg_kWh"] * get_tariff(hour_val)

    # 3. Peak power constraint penalty
    for power in hourly_power:
        if power > MAX_POWER:
            penalty += (power - MAX_POWER) * 500  # Strong penalty for exceeding limit

    return (fixed_cost + shiftable_cost) + (ALPHA * discomfort) + penalty

def selection(pop):
    return min(random.sample(pop, 3), key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(p1) - 1) if len(p1) > 1 else 0
        return (p1[:point] + p2[point:], p2[:point] + p1[point:])
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

    # Evolution Loop
    progress_bar = st.progress(0)
    for g in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE // 2):
            p1, p2 = selection(population), selection(population)
            c1, c2 = crossover(p1, p2)
            new_population.extend([mutate(c1), mutate(c2)])

        population = new_population
        best_in_gen = min(population, key=fitness)
        best_history.append(fitness(best_in_gen))
        progress_bar.progress((g + 1) / GENERATIONS)

    best_solution = min(population, key=fitness)

    # ------------------------------------------------------
    # 7. Results Display
    # ------------------------------------------------------
    st.subheader("Optimized Appliance Schedule")

    final_schedule = []
    
    # Fixed Appliances
    for _, row in non_shiftable.iterrows():
        final_schedule.append({
            "Appliance": row["Appliance"],
            "Type": "Non-Shiftable",
            "Preferred Start": row["Preferred_Time"],
            "Scheduled Start": row["Preferred_Time"],
            "End Time": (row["Preferred_Time"] + row["Duration"]) % 24,
            "Duration (h)": row["Duration"],
            "Power (kW)": row["Avg_kWh"]
        })

    # Optimized Shiftable Appliances
    for i, start in enumerate(best_solution):
        row = shiftable.iloc[i]
        final_schedule.append({
            "Appliance": row["Appliance"],
            "Type": "Shiftable",
            "Preferred Start": row["Preferred_Time"],
            "Scheduled Start": start,
            "End Time": (start + row["Duration"]) % 24,
            "Duration (h)": row["Duration"],
            "Power (kW)": row["Avg_kWh"]
        })

    result_df = pd.DataFrame(final_schedule)
    st.dataframe(result_df)

    # ------------------------------------------------------
    # 8. Metrics & Savings
    # ------------------------------------------------------
    # Baseline: Cost if everything started at Preferred Time
    baseline_cost = fitness([row["Preferred_Time"] for _, row in shiftable.iterrows()])
    optimized_cost = fitness(best_solution)

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Cost", f"RM {baseline_cost:.2f}")
    c2.metric("Optimized Cost", f"RM {optimized_cost:.2f}")
    c3.metric("Savings", f"RM {max(0, baseline_cost - optimized_cost):.2f}")

    # ------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.write("**Fitness Convergence**")
        fig1, ax1 = plt.subplots()
        ax1.plot(best_history, color='green')
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness Value")
        st.pyplot(fig1)

    with col_plot2:
        st.write("**Start Time Shift**")
        fig2, ax2 = plt.subplots()
        x = np.arange(len(result_df))
        ax2.bar(x - 0.2, result_df["Preferred Start"], 0.4, label='Preferred')
        ax2.bar(x + 0.2, result_df["Scheduled Start"], 0.4, label='Scheduled')
        ax2.set_xticks(x)
        ax2.set_xticklabels(result_df["Appliance"], rotation=45, ha='right')
        ax2.set_ylabel("Hour (0-23)")
        ax2.legend()
        st.pyplot(fig2)

# ----------------------------------------------------------
# End of Application
# ----------------------------------------------------------
