import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Smart Home Energy GA", layout="wide")
st.title("⚡ Smart Home Energy Scheduling using Genetic Algorithm")

st.markdown("""
### Objectives & Rules:
1.  **Minimize Cost**: Malaysian ToU Tariff (Peak: RM 0.50 [08:00-22:00], Off-Peak: RM 0.30).
2.  **Minimize Discomfort**: Difference between Preferred and Scheduled start times.
3.  **Peak Power Constraint**: Total consumption must stay **below 5.0 kW** at any given hour.
4.  **Shiftable vs Contrains**: Appliances marked `FALSE` stay fixed; `TRUE` can be shifted.
""")

# ----------------------------------------------------------
# 1. Load & Prepare Dataset
# ----------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("project_benchmark_data.csv")
    
    # Standardize column names for the algorithm
    df['Shiftable'] = df['Is_Shiftable'].astype(bool)
    df['Power_kW'] = df['Avg_Power_kW']
    df['Duration'] = df['Duration_Hours']
    df['Preferred'] = df['Preferred_Start_Hour']
    
    return df

df = load_and_clean_data()

# Display the input data
st.subheader("Home Appliance Configuration")
st.dataframe(df[['Appliance', 'Avg_Power_kW', 'Preferred_Start_Hour', 'Duration_Hours', 'Is_Shiftable']])

# Split dataset
shiftable_apps = df[df['Shiftable'] == True].reset_index(drop=True)
fixed_apps = df[df['Shiftable'] == False].reset_index(drop=True)

# ----------------------------------------------------------
# 2. Optimization Parameters (Sidebar)
# ----------------------------------------------------------
st.sidebar.header("GA Settings")
pop_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 50, 500, 200)
alpha = st.sidebar.slider("Discomfort Weight (α)", 0.0, 2.0, 0.5)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.30, 0.1)

# Constants
MAX_POWER_LIMIT = 5.0  # 5.0 kW Constraint
TARIFF_PEAK = 0.50
TARIFF_OFFPEAK = 0.30

# ----------------------------------------------------------
# 3. Helper Functions
# ----------------------------------------------------------
def get_tariff(hour):
    """Returns RM/kWh based on hour of the day."""
    h = hour % 24
    return TARIFF_PEAK if 8 <= h < 22 else TARIFF_OFFPEAK

def fitness(individual):
    """
    Calculates the 'cost' of a schedule. 
    Lower is better. Includes Cost, Discomfort, and Power Penalties.
    """
    total_cost = 0
    total_discomfort = 0
    penalty = 0
    hourly_power = [0.0] * 24

    # 1. Calculate Fixed Appliances (Non-Shiftable)
    for _, row in fixed_apps.iterrows():
        start = row['Preferred']
        for h in range(start, start + row['Duration']):
            hour = h % 24
            hourly_power[hour] += row['Power_kW']
            total_cost += row['Power_kW'] * get_tariff(hour)

    # 2. Calculate Shiftable Appliances
    for i, scheduled_start in enumerate(individual):
        row = shiftable_apps.iloc[i]
        
        # Discomfort = difference from preferred time
        total_discomfort += abs(scheduled_start - row['Preferred'])

        for h in range(scheduled_start, scheduled_start + row['Duration']):
            hour = h % 24
            hourly_power[hour] += row['Power_kW']
            total_cost += row['Power_kW'] * get_tariff(hour)

    # 3. Apply Hard Penalty for Peak Power Violation
    for p in hourly_power:
        if p > MAX_POWER_LIMIT:
            # Add a massive penalty for every kW over the 5.0 limit
            penalty += (p - MAX_POWER_LIMIT) * 1000 

    return total_cost + (alpha * total_discomfort) + penalty

# ----------------------------------------------------------
# 4. Genetic Algorithm Core
# ----------------------------------------------------------
def create_individual():
    return [random.randint(0, 23) for _ in range(len(shiftable_apps))]

def run_ga():
    # Initialize Population
    population = [create_individual() for _ in range(pop_size)]
    best_fitness_history = []

    progress_bar = st.progress(0)
    
    for gen in range(generations):
        # Selection (Tournament)
        population = sorted(population, key=fitness)
        best_fitness_history.append(fitness(population[0]))
        
        new_population = population[:2] # Elitism: keep best 2
        
        while len(new_population) < pop_size:
            # Selection
            parent1 = min(random.sample(population, 3), key=fitness)
            parent2 = min(random.sample(population, 3), key=fitness)
            
            # Crossover
            if random.random() < 0.8:
                point = random.randint(1, len(parent1)-1) if len(parent1) > 1 else 0
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
            else:
                child1, child2 = parent1[:], parent2[:]
            
            # Mutation
            for child in [child1, child2]:
                if random.random() < mutation_rate:
                    idx = random.randint(0, len(child)-1)
                    child[idx] = random.randint(0, 23)
                new_population.append(child)
        
        population = new_population[:pop_size]
        progress_bar.progress((gen + 1) / generations)

    return population[0], best_fitness_history

# ----------------------------------------------------------
# 5. Execution & Results Visualization
# ----------------------------------------------------------
if st.button("🚀 Run Scheduling Optimization"):
    best_schedule, history = run_ga()

    # Calculate final power profile
    final_power = [0.0] * 24
    for _, row in fixed_apps.iterrows():
        for h in range(row['Preferred'], row['Preferred'] + row['Duration']):
            final_power[h % 24] += row['Power_kW']
    for i, start in enumerate(best_schedule):
        row = shiftable_apps.iloc[i]
        for h in range(start, start + row['Duration']):
            final_power[h % 24] += row['Power_kW']

    # --- Metrics Section ---
    st.subheader("Optimization Summary")
    
    # Calculate costs
    def get_raw_cost(power_profile):
        return sum(p * get_tariff(h) for h, p in enumerate(power_profile))
    
    # Baseline comparison (if shiftable apps ran at preferred time)
    baseline_power = [0.0] * 24
    for _, row in df.iterrows():
        for h in range(row['Preferred'], row['Preferred'] + row['Duration']):
            baseline_power[h % 24] += row['Power_kW']
    
    b_cost = get_raw_cost(baseline_power)
    o_cost = get_raw_cost(final_power)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Daily Baseline Cost", f"RM {b_cost:.2f}")
    m2.metric("Daily Optimized Cost", f"RM {o_cost:.2f}")
    m3.metric("Daily Savings", f"RM {b_cost - o_cost:.2f}")

    # --- Charts Section ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Power Load Profile (24h)**")
        fig, ax = plt.subplots()
        ax.bar(range(24), final_power, color='skyblue', label='Optimized Load')
        ax.axhline(y=MAX_POWER_LIMIT, color='red', linestyle='--', label='5.0 kW Peak Limit')
        ax.set_ylim(0, max(max(final_power) + 1, 6))
        ax.set_xticks(range(24))
        ax.set_xlabel("Hour")
        ax.set_ylabel("Power (kW)")
        ax.legend()
        st.pyplot(fig)

    with col_b:
        st.write("**GA Convergence (Fitness Improvement)**")
        fig2, ax2 = plt.subplots()
        ax2.plot(history, color='green')
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Fitness (lower is better)")
        st.pyplot(fig2)

    # --- Final Table ---
    st.subheader("Final Schedule Table")
    final_table = []
    for _, row in fixed_apps.iterrows():
        final_table.append({"Appliance": row['Appliance'], "Type": "Fixed", "Start Hour": row['Preferred']})
    for i, start in enumerate(best_schedule):
        row = shiftable_apps.iloc[i]
        final_table.append({"Appliance": row['Appliance'], "Type": "Shifted", "Start Hour": start})
    
    st.table(pd.DataFrame(final_table))
    
    if max(final_power) > MAX_POWER_LIMIT:
        st.warning(f"Note: Current configuration exceeds 5kW limit slightly ({max(final_power):.2f} kW). Try increasing Generations.")
    else:
        st.success("Successfully optimized within 5.0 kW peak power constraint!")
