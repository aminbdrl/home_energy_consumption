import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="Smart Home Energy GA", layout="wide")
st.title("⚡ Smart Home Energy Scheduling (Strict 5.0kW Constraint)")

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
shiftable_apps = df[df['Shiftable'] == True].reset_index(drop=True)
fixed_apps = df[df['Shiftable'] == False].reset_index(drop=True)

# ----------------------------------------------------------
# 2. GA Parameters (Adjusted for Stricter Enforcement)
# ----------------------------------------------------------
st.sidebar.header("Optimization Settings")
# Higher population and generations help find the narrow "legal" windows
pop_size = st.sidebar.slider("Population Size", 50, 200, 100)
generations = st.sidebar.slider("Generations", 100, 1000, 400)
alpha = st.sidebar.slider("Discomfort Weight (α)", 0.0, 2.0, 0.5)

MAX_POWER_LIMIT = 5.0 
TARIFF_PEAK = 0.50
TARIFF_OFFPEAK = 0.30

def get_tariff(hour):
    h = hour % 24
    return TARIFF_PEAK if 8 <= h < 22 else TARIFF_OFFPEAK

# ----------------------------------------------------------
# 3. Enhanced Fitness Function
# ----------------------------------------------------------
def fitness(individual):
    total_cost = 0
    total_discomfort = 0
    peak_penalty = 0
    hourly_power = [0.0] * 24

    # Add Fixed Loads
    for _, row in fixed_apps.iterrows():
        for h in range(row['Preferred'], row['Preferred'] + row['Duration']):
            hourly_power[h % 24] += row['Power_kW']
            total_cost += row['Power_kW'] * get_tariff(h % 24)

    # Add Shiftable Loads
    for i, start in enumerate(individual):
        row = shiftable_apps.iloc[i]
        total_discomfort += abs(start - row['Preferred'])
        for h in range(start, start + row['Duration']):
            hour_idx = h % 24
            hourly_power[hour_idx] += row['Power_kW']
            total_cost += row['Power_kW'] * get_tariff(hour_idx)

    # STRICT PENALTY: If power > 5kW, add a massive multiplier
    for p in hourly_power:
        if p > MAX_POWER_LIMIT:
            # We use a squared penalty to make small violations very "expensive"
            peak_penalty += (p - MAX_POWER_LIMIT) * 5000 

    return total_cost + (alpha * total_discomfort) + peak_penalty

# ----------------------------------------------------------
# 4. GA Core
# ----------------------------------------------------------
def run_optimization():
    population = [[random.randint(0, 23) for _ in range(len(shiftable_apps))] for _ in range(pop_size)]
    history = []
    
    progress_bar = st.progress(0)
    for gen in range(generations):
        population = sorted(population, key=fitness)
        history.append(fitness(population[0]))
        
        # Keep the top 5 (Elitism)
        new_population = population[:5] 
        
        while len(new_population) < pop_size:
            # Tournament selection
            p1 = min(random.sample(population, 5), key=fitness)
            p2 = min(random.sample(population, 5), key=fitness)
            
            # Crossover
            point = random.randint(1, len(p1)-1) if len(p1) > 1 else 0
            child = p1[:point] + p2[point:]
            
            # Mutation (Higher rate if population is stuck)
            if random.random() < 0.2:
                child[random.randint(0, len(child)-1)] = random.randint(0, 23)
                
            new_population.append(child)
        
        population = new_population
        progress_bar.progress((gen + 1) / generations)
        
    return population[0], history

# ----------------------------------------------------------
# 5. Results and Display
# ----------------------------------------------------------
if st.button("🚀 Optimize Schedule"):
    best_ind, hist = run_optimization()
    
    # Calculate Final Load
    final_load = [0.0] * 24
    for _, row in fixed_apps.iterrows():
        for h in range(row['Preferred'], row['Preferred'] + row['Duration']):
            final_load[h % 24] += row['Power_kW']
    for i, start in enumerate(best_ind):
        row = shiftable_apps.iloc[i]
        for h in range(start, start + row['Duration']):
            final_load[h % 24] += row['Power_kW']

    # Visualizing the Peak
    
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Hourly Power Usage**")
        fig, ax = plt.subplots()
        colors = ['red' if p > 5.0 else 'green' for p in final_load]
        ax.bar(range(24), final_load, color=colors)
        ax.axhline(y=5.0, color='black', linestyle='--', label='5.0 kW Limit')
        ax.set_xticks(range(24))
        ax.set_ylabel("kW")
        st.pyplot(fig)

    with col2:
        st.write("**Final Schedule Details**")
        res_data = []
        for _, r in fixed_apps.iterrows():
            res_data.append({"Appliance": r['Appliance'], "Start": r['Preferred'], "Duration": r['Duration'], "Type": "FIXED"})
        for i, start in enumerate(best_ind):
            row = shiftable_apps.iloc[i]
            res_data.append({"Appliance": row['Appliance'], "Start": start, "Duration": row['Duration'], "Type": "SHIFTED"})
        st.dataframe(pd.DataFrame(res_data))

    max_p = max(final_load)
    if max_p > 5.0:
        st.error(f"⚠️ Failed to stay under 5kW (Max: {max_p:.2f}kW). Try increasing Generations or Population.")
    else:
        st.success(f"✅ Success! Peak power is {max_p:.2f}kW.")
