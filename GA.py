import streamlit as st
import pandas as pd
import numpy as np
import random

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Smart Home Energy GA", layout="wide")
st.title("⚡ Smart Home Energy Scheduling Optimization (Optimized)")

st.markdown("""
### ✅ Project Objectives & Constraints
* **Objective 1 (Cost):** Minimize RM using Daily Malaysia ToU Tariff (Peak: 08:00-22:00 @ RM0.50)
* **Objective 2 (Discomfort):** Minimize the time shift from user preferred hours
* **Constraint 1 (Fixed):** Fridge, TV, Lights, etc., stay at original times
* **Constraint 2 (Power):** **CRUCIAL - Total power must not exceed 5.0 kW at any hour**
""")

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("project_benchmark_data.csv")
    df['Shiftable'] = df['Is_Shiftable'].astype(bool)
    df['Power_kW'] = df['Avg_Power_kW']
    df['Duration'] = df['Duration_Hours']
    df['Preferred'] = df['Preferred_Start_Hour']
    return df

df = load_data()
shiftable_apps = df[df['Shiftable']].reset_index(drop=True)
fixed_apps = df[~df['Shiftable']].reset_index(drop=True)

# ----------------------------------------------------------
# 2. Tariff & Parameters
# ----------------------------------------------------------
st.sidebar.header("GA Settings")
pop_size = st.sidebar.slider("Population Size", 30, 150, 50)
generations = st.sidebar.slider("Generations", 50, 500, 200)
alpha = st.sidebar.slider("Discomfort Weight (α)", 0.0, 2.0, 0.5)

MAX_POWER_LIMIT = 5.0 
TARIFF_PEAK = 0.50
TARIFF_OFFPEAK = 0.30

# Precompute tariffs for all 24 hours
tariff_array = np.array([TARIFF_PEAK if 8 <= h < 22 else TARIFF_OFFPEAK for h in range(24)])

# ----------------------------------------------------------
# 3. Fitness Function (Vectorized)
# ----------------------------------------------------------
def fitness(individual):
    total_discomfort = 0
    hourly_power = np.zeros(24)

    # Fixed appliances
    for _, r in fixed_apps.iterrows():
        hours = np.arange(r['Preferred'], r['Preferred'] + r['Duration']) % 24
        hourly_power[hours] += r['Power_kW']

    # Shiftable appliances
    for i, start in enumerate(individual):
        r = shiftable_apps.iloc[i]
        total_discomfort += abs(start - r['Preferred'])
        hours = np.arange(start, start + r['Duration']) % 24
        hourly_power[hours] += r['Power_kW']

    # Cost
    total_cost = np.sum(hourly_power * tariff_array)

    # Penalty for exceeding max power
    peak_penalty = np.sum(np.clip(hourly_power - MAX_POWER_LIMIT, 0, None)) * 10000

    return total_cost + (alpha * total_discomfort) + peak_penalty

# ----------------------------------------------------------
# 4. Genetic Algorithm (Optimized)
# ----------------------------------------------------------
def solve():
    # Initialize population randomly
    population = [np.random.randint(0, 24, size=len(shiftable_apps)).tolist() for _ in range(pop_size)]
    best_history = []

    progress = st.progress(0)
    for gen in range(generations):
        # Compute fitness once per individual
        fitness_scores = [fitness(ind) for ind in population]

        # Store best
        best_idx = np.argmin(fitness_scores)
        best_history.append(fitness_scores[best_idx])

        # Selection (tournament)
        new_population = [population[best_idx]]  # Elitism: keep best

        while len(new_population) < pop_size:
            contenders = random.sample(range(pop_size), 5)
            p1 = population[min(contenders, key=lambda i: fitness_scores[i])]
            contenders = random.sample(range(pop_size), 5)
            p2 = population[min(contenders, key=lambda i: fitness_scores[i])]

            # Single-point crossover
            pt = random.randint(1, len(p1)-1) if len(p1) > 1 else 0
            child = p1[:pt] + p2[pt:]

            # Mutation
            if random.random() < 0.15:
                idx = random.randint(0, len(child)-1)
                child[idx] = random.randint(0, 23)

            new_population.append(child)

        population = new_population
        if gen % 10 == 0:  # Update progress every 10 gens
            progress.progress((gen + 1) / generations)

    # Return best individual
    fitness_scores = [fitness(ind) for ind in population]
    best_idx = np.argmin(fitness_scores)
    return population[best_idx], best_history

# ----------------------------------------------------------
# 5. Run Optimization
# ----------------------------------------------------------
if st.button("🚀 Calculate Optimized Schedule"):
    best_ind, hist = solve()

    # Compute final hourly load
    final_load = np.zeros(24)
    for _, r in fixed_apps.iterrows():
        hours = np.arange(r['Preferred'], r['Preferred'] + r['Duration']) % 24
        final_load[hours] += r['Power_kW']
    for i, start in enumerate(best_ind):
        r = shiftable_apps.iloc[i]
        hours = np.arange(start, start + r['Duration']) % 24
        final_load[hours] += r['Power_kW']

    # Daily cost
    daily_optimized = np.sum(final_load * tariff_array)
    baseline_load = np.zeros(24)
    for _, r in df.iterrows():
        hours = np.arange(r['Preferred'], r['Preferred'] + r['Duration']) % 24
        baseline_load[hours] += r['Power_kW']
    daily_baseline = np.sum(baseline_load * tariff_array)

    # Metrics
    st.subheader("Financial Impact (Daily)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Cost", f"RM {daily_baseline:.2f}")
    c2.metric("Optimized Cost", f"RM {daily_optimized:.2f}")
    c3.metric("Daily Savings", f"RM {daily_baseline - daily_optimized:.2f}")

    # Power chart (faster)
    st.write("**24-Hour Power Load Profile**")
    st.bar_chart(final_load)

    # Schedule Table
    st.subheader("Final Optimized Schedule Table")
    final_res = []

    for _, r in fixed_apps.iterrows():
        final_res.append({
            "Appliance": r['Appliance'], 
            "Type": "Non-Shiftable", 
            "Average Power (kW)": r['Power_kW'],
            "Start Time": f"{r['Preferred']}:00", 
            "Duration (h)": r['Duration'],
            "End Time": f"{(r['Preferred'] + r['Duration'])%24}:00"
        })
    for i, start in enumerate(best_ind):
        r = shiftable_apps.iloc[i]
        final_res.append({
            "Appliance": r['Appliance'], 
            "Type": "Shiftable", 
            "Average Power (kW)": r['Power_kW'],
            "Start Time": f"{start}:00", 
            "Duration (h)": r['Duration'],
            "End Time": f"{(start + r['Duration'])%24}:00"
        })

    st.table(pd.DataFrame(final_res))

    # Peak check
    max_load = np.max(final_load)
    if max_load > MAX_POWER_LIMIT:
        st.error(f"⚠️ Limit Exceeded: {max_load:.2f} kW. Increase generations to find valid slots.")
    else:
        st.success(f"✅ Peak Power kept under {MAX_POWER_LIMIT} kW! (Max: {max_load:.2f} kW)")
