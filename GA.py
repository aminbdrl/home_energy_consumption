import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Smart Home Energy GA", layout="wide")
st.title("⚡ Smart Home Energy Scheduling Optimization")

st.markdown("""
### ✅ Project Objectives & Constraints
* **Objective 1 (Cost):** Minimize RM using Daily Malaysia ToU Tariff (Peak: 08:00-22:00 @ RM0.50).
* **Objective 2 (Discomfort):** Minimize the time shift from user preferred hours.
* **Constraint 1 (Fixed):** Fridge, TV, Lights, etc., stay at original times.
* **Constraint 2 (Power):** **CRUCIAL - Total power must not exceed 5.0 kW at any hour.**
""")

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
@st.cache_data
def load_data():
    # Reading from project_benchmark_data.csv
    df = pd.read_csv("project_benchmark_data.csv")
    df['Shiftable'] = df['Is_Shiftable'].astype(bool)
    df['Power_kW'] = df['Avg_Power_kW']
    df['Duration'] = df['Duration_Hours']
    df['Preferred'] = df['Preferred_Start_Hour']
    return df

df = load_data()
shiftable_apps = df[df['Shiftable'] == True].reset_index(drop=True)
fixed_apps = df[df['Shiftable'] == False].reset_index(drop=True)

# ----------------------------------------------------------
# 2. Tariff & Parameters
# ----------------------------------------------------------
st.sidebar.header("GA Settings")
pop_size = st.sidebar.slider("Population Size", 50, 200, 100)
generations = st.sidebar.slider("Generations", 100, 1000, 500)
alpha = st.sidebar.slider("Discomfort Weight (α)", 0.0, 2.0, 0.5)

MAX_POWER_LIMIT = 5.0 
TARIFF_PEAK = 0.50
TARIFF_OFFPEAK = 0.30

def get_tariff(hour):
    h = hour % 24
    return TARIFF_PEAK if 8 <= h < 22 else TARIFF_OFFPEAK

# ----------------------------------------------------------
# 3. Fitness Function
# ----------------------------------------------------------
def fitness(individual):
    total_cost = 0
    total_discomfort = 0
    peak_penalty = 0
    hourly_power = [0.0] * 24

    # Load Fixed Appliances
    for _, row in fixed_apps.iterrows():
        for h in range(row['Preferred'], row['Preferred'] + row['Duration']):
            hourly_power[h % 24] += row['Power_kW']
            total_cost += row['Power_kW'] * get_tariff(h % 24)

    # Load Shiftable Appliances
    for i, start in enumerate(individual):
        row = shiftable_apps.iloc[i]
        total_discomfort += abs(start - row['Preferred'])
        for h in range(start, start + row['Duration']):
            hour_idx = h % 24
            hourly_power[hour_idx] += row['Power_kW']
            total_cost += row['Power_kW'] * get_tariff(hour_idx)

    # ENFORCE 5.0 kW RULE
    for p in hourly_power:
        if p > MAX_POWER_LIMIT:
            peak_penalty += (p - MAX_POWER_LIMIT) * 10000 

    return total_cost + (alpha * total_discomfort) + peak_penalty

# ----------------------------------------------------------
# 4. Genetic Algorithm Execution
# ----------------------------------------------------------
def solve():
    population = [[random.randint(0, 23) for _ in range(len(shiftable_apps))] for _ in range(pop_size)]
    best_history = []
    
    progress = st.progress(0)
    for gen in range(generations):
        population = sorted(population, key=fitness)
        best_history.append(fitness(population[0]))
        
        new_gen = population[:10] 
        while len(new_gen) < pop_size:
            p1 = min(random.sample(population, 5), key=fitness)
            p2 = min(random.sample(population, 5), key=fitness)
            pt = random.randint(1, len(p1)-1) if len(p1) > 1 else 0
            child = p1[:pt] + p2[pt:]
            if random.random() < 0.15:
                child[random.randint(0, len(child)-1)] = random.randint(0, 23)
            new_gen.append(child)
        population = new_gen
        progress.progress((gen + 1) / generations)
        
    return population[0], best_history

# ----------------------------------------------------------
# 5. Output and Reporting
# ----------------------------------------------------------
if st.button("🚀 Calculate Optimized Schedule"):
    best_ind, hist = solve()
    
    # Calculate Final Load
    final_load = [0.0] * 24
    for _, row in fixed_apps.iterrows():
        for h in range(row['Preferred'], row['Preferred'] + row['Duration']):
            final_load[h % 24] += row['Power_kW']
    for i, start in enumerate(best_ind):
        row = shiftable_apps.iloc[i]
        for h in range(start, start + row['Duration']):
            final_load[h % 24] += row['Power_kW']

    # Cost Analysis (Daily)
    daily_optimized = sum(p * get_tariff(h) for h, p in enumerate(final_load))
    
    # Baseline
    baseline_load = [0.0] * 24
    for _, row in df.iterrows():
        for h in range(row['Preferred'], row['Preferred'] + row['Duration']):
            baseline_load[h % 24] += row['Power_kW']
    daily_baseline = sum(p * get_tariff(h) for h, p in enumerate(baseline_load))

    # Metrics (Changed to Daily)
    st.subheader("Financial Impact (Daily)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Cost", f"RM {daily_baseline:.2f}")
    c2.metric("Optimized Cost", f"RM {daily_optimized:.2f}")
    c3.metric("Daily Savings", f"RM {daily_baseline - daily_optimized:.2f}")

    # Power Chart
    st.write("**24-Hour Power Load Profile**")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(24), final_load, color='skyblue', label='Actual Load')
    ax.axhline(y=5.0, color='red', linestyle='--', label='5.0 kW Threshold')
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    st.pyplot(fig)

    # Schedule Table (Added Average Power kW)
    st.subheader("Final Optimized Schedule Table")
    final_res = []
    
    # Add Non-Shiftable Appliances
    for _, r in fixed_apps.iterrows():
        final_res.append({
            "Appliance": r['Appliance'], 
            "Type": "Non-Shiftable", 
            "Average Power (kW)": r['Power_kW'],
            "Start Time": f"{r['Preferred']}:00", 
            "Duration (h)": r['Duration'],
            "End Time": f"{(r['Preferred'] + r['Duration'])%24}:00"
        })
        
    # Add Shifted Appliances
    for i, start in enumerate(best_ind):
        row = shiftable_apps.iloc[i]
        final_res.append({
            "Appliance": row['Appliance'], 
            "Type": "Shiftable", 
            "Average Power (kW)": row['Power_kW'],
            "Start Time": f"{start}:00", 
            "Duration (h)": row['Duration'],
            "End Time": f"{(start + row['Duration'])%24}:00"
        })
    
    st.table(pd.DataFrame(final_res))
    
    if max(final_load) > 5.0:
        st.error(f"⚠️ Limit Exceeded: {max(final_load):.2f} kW. Increase generations to find a valid slot.")
    else:
        st.success(f"✅ Peak Power kept under 5.0 kW! (Max: {max(final_load):.2f} kW)")
